"""Out-of-scope triage: one short call on the upright context, never field values."""
import asyncio
from copy import deepcopy
import json

from PIL import Image, ImageDraw
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever import zoom_image_reader as wrapper
from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_triage import TRIAGE_OUTPUT_LIMIT, TriageSchemaError, triage_frame
from sever.native_pipeline import process_video_images
from test_image_agent import last_panel, reading, response
from test_native_pipeline import setup_pages


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "private-source-name-not-input.png"
    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((7, 19, 177, 299), fill="navy")
    image.save(path)
    return {"frame_id": "source-frame-23", "path": str(path), "timestamp_s": 2.3}


def configuration(**options):
    return AgentConfig(model_name="triage-fixture", vision_model_name="triage-fixture",
        provider_kind="local", base_url="http://127.0.0.1:8090/v1",
        document_reader="native_image", max_output_tokens=8192, **options)


def triage_model(calls, answer=None, *, raw=None, fail=None):
    def respond(messages, info):
        calls.append(messages)
        assert info.output_tools == [] and info.function_tools == []
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_settings["temperature"] == 0
        assert info.model_settings["max_tokens"] <= TRIAGE_OUTPUT_LIMIT
        if fail:
            raise fail
        text = raw if raw is not None else json.dumps(answer)
        return ModelResponse(parts=[TextPart(text)], usage=RequestUsage(input_tokens=40, output_tokens=9))
    return FunctionModel(respond, model_name="offline-triage-model")


def run(frame, tmp_path, model, *, settings=None, rotation=0):
    return asyncio.run(triage_frame(frame, rotation=rotation, source_id="source-23", page_id="leaf-7",
        config=settings or configuration(), cache_dir=tmp_path / "evidence",
        trace_path=tmp_path / "trace.jsonl", model=model))


def images(messages):
    return [item for message in messages for part in message.parts
            if isinstance(getattr(part, "content", None), list)
            for item in part.content if isinstance(item, BinaryContent)]


def test_clear_out_of_scope_heading_becomes_an_unknown_reading_without_fields(frame, tmp_path):
    calls, settings = [], configuration()
    triage = run(frame, tmp_path, triage_model(calls, {"form_scope": "out_of_scope",
                                                       "observed_form": "АКТ сверки"}), settings=settings)
    assert len(calls) == 1 and len(images(calls[0])) == 1 and settings.usage.requests == 1
    result = triage["result"]
    document = result["document"]
    assert document.document_type == "unknown" and document.fields == {}
    assert [e.frame_id for e in document.classification_evidence] == ["source-frame-23"]
    recognition = result["recognition"]
    assert recognition["form_scope"] == "out_of_scope" and recognition["triage"] is True
    assert recognition["observed_form"] == recognition["visible_text"] == "АКТ сверки"
    assert recognition["pixel_proof"]["rotation_clockwise_degrees"] == 0
    assert len(recognition["pixel_proof"]["source_sha256"]) == 64
    record = json.loads(open(triage["evidence_path"]).read())
    assert record["status"] == "success" and record["short_circuit"] is True


@pytest.mark.parametrize("answer", [{"form_scope": "possible_supported", "observed_form": "Счет-фактура"},
                                    {"form_scope": "insufficient_evidence", "observed_form": ""},
                                    {"form_scope": "out_of_scope", "observed_form": "  "}])
def test_anything_but_a_named_out_of_scope_form_needs_the_full_reader(frame, tmp_path, answer):
    triage = run(frame, tmp_path, triage_model([], answer))
    assert triage["result"] is None and triage["form_scope"] == answer["form_scope"]


def test_schema_failure_and_provider_stops_have_distinct_errors(frame, tmp_path):
    with pytest.raises(TriageSchemaError):
        run(frame, tmp_path, triage_model([], raw='{"form_scope":"maybe"}'))
    with pytest.raises(FatalProviderError):
        run(frame, tmp_path, triage_model([], fail=ModelHTTPError(401, "offline", {"error": "denied"})))
    with pytest.raises(AnalysisError):
        run(frame, tmp_path, triage_model([], fail=TimeoutError("no response")))


def test_exhausted_token_budget_makes_no_triage_call(frame, tmp_path):
    calls, settings = [], configuration(input_tokens_limit=10)
    settings.usage.input_tokens = 10
    with pytest.raises(BatchBudgetExceeded):
        run(frame, tmp_path, triage_model(calls, {"form_scope": "out_of_scope", "observed_form": "CMR"}),
            settings=settings)
    assert calls == []


def orientation_model(calls):
    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('{"rotation_clockwise": 0}')],
                             usage=RequestUsage(input_tokens=12, output_tokens=3))
    return FunctionModel(respond)


def field_model(calls):
    def respond(messages, info):
        calls.append(messages)
        payload = reading(last_panel(messages))
        payload["reading"]["form_scope"] = "out_of_scope"
        return response(payload)
    return FunctionModel(respond)


def read(frame, tmp_path, **options):
    return asyncio.run(wrapper.read_frame_zoom(frame, source_id="source", page_id="leaf",
        config=configuration(), cache_dir=tmp_path / "proof", trace_path=tmp_path / "trace.jsonl", **options))


def test_out_of_scope_triage_skips_the_field_agent_and_keeps_triaging_the_leaf(frame, tmp_path):
    triage_calls, field_calls, orientation_calls, state = [], [], [], {}
    options = dict(orientation_model=orientation_model(orientation_calls), model=field_model(field_calls),
                   triage_model=triage_model(triage_calls, {"form_scope": "out_of_scope", "observed_form": "CMR"}),
                   leaf_orientation={}, leaf_triage=state)
    first = read(frame, tmp_path, **options)
    second = read(frame, tmp_path, **options)
    assert field_calls == [] and len(triage_calls) == 2 and len(orientation_calls) == 1
    assert state == {}
    assert first["recognition"]["image_preparation"]["triage"]["short_circuit"] is True
    assert second["document"].document_type == "unknown"


@pytest.mark.parametrize("schema_failure", [False, True])
def test_possible_target_switches_the_leaf_to_full_reads(frame, tmp_path, schema_failure):
    triage_calls, field_calls, state = [], [], {}
    answer = {"form_scope": "possible_supported", "observed_form": "Транспортная накладная"}
    options = dict(orientation_model=orientation_model([]), model=field_model(field_calls),
                   triage_model=triage_model(triage_calls, answer, raw='{"bad": 1}' if schema_failure else None),
                   leaf_orientation={}, leaf_triage=state)
    first = read(frame, tmp_path, **options)
    read(frame, tmp_path, **options)
    assert len(triage_calls) == 1 and len(field_calls) == 2
    assert state == {"full_read_required": True, "decided_by_frame_id": frame["frame_id"]}
    assert first["recognition"]["image_preparation"]["triage"] == {
        "form_scope": None if schema_failure else "possible_supported",
        "evidence_path": None if schema_failure else first["recognition"]["image_preparation"]["triage"]["evidence_path"],
        "short_circuit": False, "schema_failed": schema_failure}


def test_two_triaged_out_of_scope_views_confirm_the_leaf_without_field_reads(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    calls = []

    async def reader(frame, *, source_id, page_id, config, cache_dir, trace_path):
        triage = await triage_frame(frame, rotation=0, source_id=source_id, page_id=page_id,
            config=config, cache_dir=cache_dir, trace_path=trace_path,
            model=triage_model(calls, {"form_scope": "out_of_scope", "observed_form": "Путевой лист"}))
        return triage["result"]

    result = asyncio.run(process_video_images(source, tmp_path / "run", config=configuration(),
                                              reader=reader, additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    assert len(calls) == 2 and page["result"]["document_type"] == "unknown"
    assert page["image_recognition"]["stop_reason"] == "out_of_scope_form_confirmed"
    assert page["result"]["errors"] == []
