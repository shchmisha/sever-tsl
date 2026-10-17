"""Model-facing tool contracts participate in the shared preflight allowance."""
import asyncio
import json
import math
from types import SimpleNamespace

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, ModelResponse, NativeOutput, ToolOutput
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ModelRequest, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits
import pytest

from sever.agents import (AgentConfig, BatchBudgetExceeded, _PageDeps,
                          _after_request, _before_request, _dump)


class SmallOutput(BaseModel):
    text: str


def config(**kwargs):
    return AgentConfig(model_name="offline-tool-budget", vision_model_name="offline-tool-budget",
        provider_kind="local", base_url="http://127.0.0.1:8090/v1",
        api_key="PRIVATE-CONFIG-KEY", max_output_tokens=64, max_requests=5,
        max_tokens=5000, **kwargs)


def run_agent(tmp_path, settings, output_type, calls, *, function_description=None):
    def respond(messages, info):
        calls.append(info.model_request_parameters)
        if info.output_tools:
            assert info.model_request_parameters.output_object is None
            parts = [ToolCallPart(info.output_tools[0].name, {"text": "literal"})]
        else:
            parts = [TextPart('{"text":"literal"}')]
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=17, output_tokens=3))
    model = FunctionModel(respond, model_name="offline-tool-budget")
    deps = _PageDeps({}, "synthetic-source", "synthetic-page", tmp_path / "trace.jsonl", settings)
    agent = Agent(model, output_type=output_type, deps_type=_PageDeps, retries=0,
        capabilities=[Hooks(before_model_request=_before_request, after_model_request=_after_request)])
    if function_description is not None:
        @agent.tool_plain(description=function_description)
        def zoom_source(x: int) -> int:
            return x
    return asyncio.run(agent.run("Read actual pixels", deps=deps, usage=settings.usage,
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=settings.max_tokens),
        model_settings={"max_tokens": 64, "temperature": 0}))


@pytest.mark.parametrize("location", ["parameters", "description"])
def test_large_actual_tool_output_schema_blocks_before_any_request(tmp_path, location):
    schema_type = (create_model("LargeOutput", text=(str, Field(description="pixel schema " * 4000)))
                   if location == "parameters" else SmallOutput)
    output = ToolOutput(schema_type, name="final_reading",
                        description="output contract " * 4000 if location == "description" else "final reading")
    calls, settings = [], config()
    # There is enough original batch allowance; only its small remainder is
    # available after prior actual usage. The oversized tool contract cannot fit.
    settings.max_tokens = 20000
    settings.usage.input_tokens = 15000
    with pytest.raises(BatchBudgetExceeded, match="Estimated next request"):
        run_agent(tmp_path, settings, output, calls)
    assert calls == [] and settings.usage.requests == settings.usage.attempted_requests == 0
    assert settings.usage.input_tokens == 15000 and settings.usage.output_tokens == 0
    assert settings.usage.actual_cost_rub == 0 and not (tmp_path / "trace.jsonl").exists()


def test_large_actual_function_tool_description_also_blocks_native_request(tmp_path):
    calls, settings = [], config()
    with pytest.raises(BatchBudgetExceeded, match="Estimated next request"):
        run_agent(tmp_path, settings, NativeOutput(SmallOutput, strict=False), calls,
                  function_description="source pixel tool contract " * 4000)
    assert calls == [] and settings.usage.requests == settings.usage.attempted_requests == 0


@pytest.mark.parametrize("output_type", [NativeOutput(SmallOutput, strict=False),
                                       ToolOutput(SmallOutput, name="final_reading")])
def test_small_native_and_tool_outputs_preserve_actual_shared_usage(tmp_path, output_type):
    calls, settings = [], config()
    result = run_agent(tmp_path, settings, output_type, calls)
    assert result.output.text == "literal" and len(calls) == 1
    assert settings.usage.requests == settings.usage.attempted_requests == 1
    assert settings.usage.input_tokens == 17 and settings.usage.output_tokens == 3
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["model_request_started", "model_response_received"]
    assert rows[-1]["billing"]["kind"] == "local_inference"
    assert "PRIVATE-CONFIG-KEY" not in (tmp_path / "trace.jsonl").read_text()


def preflight(tmp_path, settings, *, functions=(), outputs=()):
    deps = _PageDeps({}, "synthetic-source", "synthetic-page", tmp_path / "trace.jsonl", settings)
    request = ModelRequestContext(model=FunctionModel(lambda *args: None),
        messages=[ModelRequest(parts=[UserPromptPart("pixel-only")])],
        model_request_parameters=ModelRequestParameters(function_tools=list(functions), output_tools=list(outputs)),
        model_settings={"max_tokens": 64})
    returned = asyncio.run(_before_request(SimpleNamespace(deps=deps), request))
    assert returned is request
    return deps


def test_identical_actual_tool_definitions_are_not_counted_twice(tmp_path):
    tool = ToolDefinition(name="zoom_source", description="crop " * 700,
        parameters_json_schema={"type": "object", "properties": {"x": {"type": "number"}}})
    wire = _dump({"name": tool.name, "description": tool.description,
                  "parameters": tool.parameters_json_schema})
    settings = config()
    settings.max_tokens = math.ceil((len(_dump("pixel-only")) + len(wire)) / 1.5) + 512 + 64
    deps = preflight(tmp_path, settings, functions=[tool], outputs=[tool])
    assert deps.requests_started == settings.usage.attempted_requests == 1
    assert settings.usage.requests == 0  # This test exercises preflight only.


def test_different_tool_names_with_same_schema_both_consume_allowance(tmp_path):
    schema = {"type": "object", "description": "pixel geometry " * 450, "properties": {}}
    tools = [ToolDefinition(name=name, parameters_json_schema=schema) for name in ("zoom_left", "zoom_right")]
    calls, settings = [], config()
    with pytest.raises(BatchBudgetExceeded):
        preflight(tmp_path, settings, functions=tools)
    assert settings.usage.requests == settings.usage.attempted_requests == 0
    assert not (tmp_path / "trace.jsonl").exists()


def test_private_tool_metadata_is_not_serialized_or_counted(tmp_path):
    class PrivateRuntime:
        def __str__(self):
            raise AssertionError("Private runtime data must not be serialized")
    tool = ToolDefinition(name="zoom_source", metadata={
        "api_key": "PRIVATE-TOOL-KEY" * 4000, "runtime": PrivateRuntime()})
    settings = config()
    settings.max_tokens = 1000
    preflight(tmp_path, settings, functions=[tool])
    trace = (tmp_path / "trace.jsonl").read_text()
    assert "PRIVATE-TOOL-KEY" not in trace and "PRIVATE-CONFIG-KEY" not in trace


@pytest.mark.parametrize("include", [True, False])
def test_included_tool_return_schema_consumes_allowance(tmp_path, include):
    tool = ToolDefinition(name="zoom_source", return_schema={
        "type": "object", "description": "pixel panel geometry " * 4000, "properties": {}},
        include_return_schema=include)
    settings = config()
    if include:
        with pytest.raises(BatchBudgetExceeded):
            preflight(tmp_path, settings, functions=[tool])
        assert settings.usage.attempted_requests == 0
    else:
        preflight(tmp_path, settings, functions=[tool])
        assert settings.usage.attempted_requests == 1
