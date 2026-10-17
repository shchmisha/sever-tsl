import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, _after_request
from sever.cli import agent_config


@pytest.mark.parametrize("url", ["https://routerai.ru/api/v1", "https://127.0.0.1.evil.test/v1",
                                 "http://192.168.0.1/v1", "file:///tmp/model", "http://user@localhost/v1"])
def test_local_configuration_rejects_non_loopback_endpoints(url):
    with pytest.raises(ValueError, match="loopback"):
        AgentConfig(provider_kind="local", base_url=url)


def test_local_config_without_secret_selects_same_vision_model_and_long_timeout(monkeypatch):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "http://127.0.0.1:8090/v1")
    for name in ("HARNESS_MODEL_API_KEY", "HARNESS_VISION_MODEL", "HARNESS_MODEL_TIMEOUT", "HARNESS_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    config = agent_config("local-candidate", 20, 100000)
    assert config.model_name == config.vision_model_name == "local-candidate"
    assert config.provider_kind == "local" and config.timeout_seconds == 600
    assert config.document_reader == "native_image"
    assert config.usage.model_dump()["provider_cost_currency"] is None
    monkeypatch.setenv("HARNESS_MODEL_API_KEY", "remote-key-must-not-be-forwarded")
    assert agent_config("local-candidate",20,2000000).api_key == "local-no-key"
    monkeypatch.delenv("HARNESS_MODEL_BASE_URL")
    with pytest.raises(ValueError, match="loopback"):
        agent_config("local-candidate", 20, 100000)


def test_explicit_document_reader_mode_is_validated(monkeypatch):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "http://127.0.0.1:8090/v1")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "ocr_guided")
    assert agent_config("candidate", 20, 100000).document_reader == "ocr_guided"
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "typo")
    with pytest.raises(ValueError, match="document_reader"):
        agent_config("candidate", 20, 100000)


def test_local_trace_records_origin_and_no_provider_bill(tmp_path):
    config = AgentConfig(provider_kind="local", base_url="http://localhost:8090/v1")
    path = tmp_path / "trace.jsonl"
    ctx = SimpleNamespace(deps=SimpleNamespace(config=config, trace_path=path, source_id="source", page_id="page"))
    response = ModelResponse(parts=[TextPart("{}")], usage=RequestUsage(input_tokens=19, output_tokens=2),
                             provider_details={"routerai_cost_rub": 99})
    asyncio.run(_after_request(ctx, None, response))
    row = json.loads(path.read_text())
    assert row["provider_kind"] == "local" and row["model_base_url"] == config.base_url
    assert row["billing"] == {"kind": "local_inference", "value": 0, "currency": None}
    assert config.usage.actual_cost_rub == config.usage.costed_requests == 0


def test_llama_server_timings_are_kept_in_provider_details():
    from openai.types.chat import ChatCompletion
    from sever.agents import _RouterAIChatModel, _server_timings

    completion = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "local",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "{}"}}],
        "usage": {"prompt_tokens": 12000, "completion_tokens": 150, "total_tokens": 12150},
        "timings": {"cache_n": 9000, "prompt_n": 3000, "prompt_ms": 1100.5, "predicted_n": 150,
                    "predicted_ms": 2500.0, "predicted_per_second": 60.0, "unexpected": "dropped",
                    "prompt_per_second": float("nan")}})
    model = _RouterAIChatModel.__new__(_RouterAIChatModel)
    details = model._process_provider_details(completion)
    assert details["server_timings"] == {"cache_n": 9000, "prompt_n": 3000, "prompt_ms": 1100.5,
        "predicted_n": 150, "predicted_ms": 2500.0, "predicted_per_second": 60.0}
    assert _server_timings({}) is None and _server_timings({"timings": {"note": "x"}}) is None


def test_trace_records_wall_time_cache_reads_and_server_timings(tmp_path, monkeypatch):
    import sever.agents as agents

    config = AgentConfig(provider_kind="local", base_url="http://localhost:8090/v1")
    path = tmp_path / "trace.jsonl"
    deps = SimpleNamespace(config=config, trace_path=path, source_id="source", page_id="page",
                           request_started_monotonic=100.0)
    monkeypatch.setattr(agents.time, "monotonic", lambda: 104.25)
    timings = {"cache_n": 9000, "prompt_n": 3000, "predicted_n": 150, "predicted_ms": 2500.0}
    response = ModelResponse(parts=[TextPart("{}")], provider_details={"server_timings": timings},
                             usage=RequestUsage(input_tokens=12000, output_tokens=150, cache_read_tokens=9000))
    asyncio.run(_after_request(SimpleNamespace(deps=deps), None, response))
    row = json.loads(path.read_text())
    assert row["elapsed_seconds"] == 4.25 and row["server_timings"] == timings
    assert row["cache_read_tokens"] == 9000 and deps.request_started_monotonic is None
