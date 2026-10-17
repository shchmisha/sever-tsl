"""Fresh orientation-only contracts using real local pixels and an offline model."""
import asyncio
from dataclasses import replace
import hashlib
from io import BytesIO
import json

from PIL import Image, ImageDraw
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_orientation import (ORIENTATION_OUTPUT_LIMIT, OrientationSchemaError,
                                     choose_reading_rotation)


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "private-source-name-not-input.png"
    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((7, 19, 177, 299), fill="navy")
    image.save(path)
    return {"frame_id": "source-frame-23", "path": str(path), "timestamp_s": 2.3}


def configuration(**options):
    return AgentConfig(model_name="orientation-fixture", vision_model_name="orientation-fixture",
        provider_kind="local", base_url="http://127.0.0.1:8090/v1",
        document_reader="native_image", max_output_tokens=8192, **options)


def offline_model(calls, response='{"rotation_clockwise":90}', *, fail=None, input_tokens=50):
    def respond(messages, info):
        calls.append((messages, info))
        assert info.output_tools == [] and info.function_tools == []
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_request_parameters.output_object.strict is False
        assert set(info.model_request_parameters.output_object.json_schema["properties"]) == {"rotation_clockwise"}
        assert info.model_settings["temperature"] == 0
        assert info.model_settings["max_tokens"] <= ORIENTATION_OUTPUT_LIMIT
        if fail:
            raise fail
        return ModelResponse(parts=[TextPart(response)],
                             usage=RequestUsage(input_tokens=input_tokens, output_tokens=12))
    return FunctionModel(respond, model_name="offline-orientation-model")


def run(frame, tmp_path, *, settings=None, model=None):
    return asyncio.run(choose_reading_rotation(frame, source_id="source-23", page_id="leaf-7",
        config=settings or configuration(), cache_dir=tmp_path / "evidence",
        trace_path=tmp_path / "trace.jsonl", model=model or offline_model([])))


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270, None])
def test_actual_four_context_images_and_exact_model_decision_bound_to_source(frame, tmp_path, rotation):
    calls = []
    decision = {"rotation_clockwise": rotation}
    result = run(frame, tmp_path, model=offline_model(calls, json.dumps(decision)))
    assert result["rotation_clockwise"] == rotation and result["raw_decision"] == decision
    proof = result["proof"]
    assert proof["frame_id"] == frame["frame_id"]
    assert proof["source_sha256"] == hashlib.sha256(open(frame["path"], "rb").read()).hexdigest()
    assert list(proof["panels"]) == [f"context_r{value}" for value in (0, 90, 180, 270)]
    assert len(calls) == 1
    content = [item for message in calls[0][0] for part in message.parts
               if isinstance(getattr(part, "content", None), list) for item in part.content]
    images = [item for item in content if isinstance(item, BinaryContent)]
    assert len(images) == 4
    for binary, panel in zip(images, proof["panels"].values(), strict=True):
        assert hashlib.sha256(binary.data).hexdigest() == panel["upload_sha256"]
        assert binary.data == open(panel["upload_path"], "rb").read()
        assert Image.open(BytesIO(binary.data)).size == tuple(panel["upload_pixel_size"])
        assert panel["frame_id"] == frame["frame_id"] and panel["bbox"] == [0, 0, 1, 1]
    text = "\n".join(item for item in content if isinstance(item, str))
    assert frame["path"] not in text and "source-23" not in text and "leaf-7" not in text
    assert "fields_by_type" not in text and "detail_" not in text
    record = json.loads(open(result["evidence_path"]).read())
    assert record["proof"] == proof and record["raw_decision"] == decision
    assert json.loads(record["raw_response_parts"][0]["content"]) == decision
    assert record["metadata"]["output_limit"] == 128
    assert record["usage_delta"] == {"requests": 1, "input_tokens": 50, "output_tokens": 12, "tool_calls": 0}
    trace = events(tmp_path)
    assert [row["event"] for row in trace] == ["image_orientation_observation", "model_request_started",
        "model_response_received", "image_orientation_reading"]
    assert trace[2]["billing"] == {"value": 0, "currency": None, "kind": "local_inference"}


def test_only_four_context_images_are_encoded_and_saved_without_legacy_details(frame, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The legacy twelve-panel preparation must not run")
    monkeypatch.setattr("sever.image_reader._prepare_panels", forbidden)
    encoded = []
    original_save = Image.Image.save
    def counted(self, destination, *args, **kwargs):
        if isinstance(destination, BytesIO):
            encoded.append(self.size)
        return original_save(self, destination, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "save", counted)
    result = run(frame, tmp_path)
    assert len(encoded) == 4
    assert len(list((tmp_path/"evidence"/"image-panels").glob("*.jpg"))) == 4
    assert all(identifier.startswith("context_r") for identifier in result["proof"]["panels"])
    assert result["proof"]["source_unchanged_during_preparation"] is True
    assert result["proof"]["rotations_clockwise_degrees"] == [0, 90, 180, 270]
    assert result["proof"]["recipe"] == "four-native-source-context-rotations-v1"


def test_identical_source_calls_are_fresh_unique_and_never_load_poisoned_answers(frame, tmp_path):
    calls = []
    settings = configuration(refresh_cache=False)
    first = run(frame, tmp_path, settings=settings, model=offline_model(calls))
    # A historically saved answer is not model input, even if it is malformed.
    open(first["evidence_path"], "w").write("PRIVATE-OLD-ANSWER-DO-NOT-REPLAY")
    second = run(frame, tmp_path, settings=settings,
                 model=offline_model(calls, '{"rotation_clockwise":270}'))
    assert first["evidence_path"] != second["evidence_path"]
    assert first["rotation_clockwise"] == 90 and second["rotation_clockwise"] == 270
    assert first["proof"]["source_sha256"] == second["proof"]["source_sha256"]
    assert len(calls) == 2 and settings.usage.requests == settings.usage.attempted_requests == 2
    assert len(list((tmp_path / "evidence").glob("orientation-reading-*.json"))) == 2
    assert "PRIVATE-OLD-ANSWER" not in str(calls[1][0])
    assert not any("cache_hit" in row["event"] for row in events(tmp_path))


def test_orientation_and_following_native_read_share_the_same_request_allowance(frame, tmp_path):
    from sever.image_reader import read_frame
    calls = []
    settings = configuration(max_requests=1)
    run(frame, tmp_path, settings=settings, model=offline_model(calls))
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(read_frame(frame, source_id="source-23", page_id="leaf-7", config=settings,
            cache_dir=tmp_path / "native-evidence", trace_path=tmp_path / "trace.jsonl",
            model=offline_model(calls)))
    assert len(calls) == 1 and settings.usage.requests == 1


@pytest.mark.parametrize("kind", ["requests", "attempts", "input", "output", "total"])
def test_shared_exhausted_budget_prevents_any_model_call(frame, tmp_path, kind):
    calls = []
    settings = configuration(max_requests=1) if kind in {"requests", "attempts"} else configuration(max_tokens=1) if kind == "total" else configuration()
    if kind == "requests":
        settings.usage.requests = 1
    elif kind == "attempts":
        settings.usage.attempted_requests = 1
    elif kind == "input":
        settings.usage.input_tokens = settings.input_tokens_limit
    elif kind == "output":
        settings.usage.output_tokens = settings.output_tokens_limit
    with pytest.raises(BatchBudgetExceeded):
        run(frame, tmp_path, settings=settings, model=offline_model(calls))
    assert calls == []
    record = json.loads(next((tmp_path / "evidence").glob("orientation-reading-*.json")).read_text())
    assert record["status"] == "budget_exhausted" and record["raw_response_parts"] == []


def test_actual_token_overrun_remains_batch_budget_failure_with_received_proof(frame, tmp_path):
    calls = []
    settings = configuration(input_tokens_limit=51)
    with pytest.raises(BatchBudgetExceeded):
        run(frame, tmp_path, settings=settings, model=offline_model(calls, input_tokens=52))
    assert len(calls) == 1 and settings.usage.input_tokens == 52 and settings.usage.requests == 1
    record = json.loads(next((tmp_path / "evidence").glob("orientation-reading-*.json")).read_text())
    assert record["status"] == "budget_exhausted" and record["raw_response_parts"]


@pytest.mark.parametrize("response", ['{"rotation_clockwise":45}', '{"rotation_clockwise":"90"}',
    '{"rotation_clockwise":false}', '{"rotation_clockwise":90.0}', '{}',
    '{"rotation_clockwise":90,"fields":{"manual":"answer"}}', 'invalid JSON'])
def test_bad_schema_is_distinct_and_retains_original_response_without_retry(frame, tmp_path, response):
    calls = []
    settings = configuration()
    with pytest.raises(OrientationSchemaError):
        run(frame, tmp_path, settings=settings, model=offline_model(calls, response))
    assert len(calls) == 1 and settings.usage.requests == 1
    record = json.loads(next((tmp_path / "evidence").glob("orientation-reading-*.json")).read_text())
    assert record["status"] == "schema_failed"
    assert record["raw_response_parts"][0]["content"] == response
    assert events(tmp_path)[-1]["failure_kind"] == "schema_failed"


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_fatal_provider_error_is_a_stop_and_private_body_is_not_recorded(frame, tmp_path, status):
    calls = []
    settings = configuration()
    with pytest.raises(FatalProviderError, match=f"HTTP {status}"):
        run(frame, tmp_path, settings=settings, model=offline_model(calls,
            fail=ModelHTTPError(status, "offline-model", "PRIVATE-PROVIDER-BODY")))
    assert len(calls) == 1 and settings.usage.requests == settings.usage.attempted_requests == 1
    record = next((tmp_path / "evidence").glob("orientation-reading-*.json"))
    assert json.loads(record.read_text())["status"] == "fatal"
    assert "PRIVATE-PROVIDER-BODY" not in record.read_text() and "PRIVATE-PROVIDER-BODY" not in (tmp_path / "trace.jsonl").read_text()


def test_ordinary_provider_failure_is_not_unknown_orientation_or_schema_fallback(frame, tmp_path):
    calls = []
    settings = configuration()
    with pytest.raises(AnalysisError, match="RuntimeError") as failure:
        run(frame, tmp_path, settings=settings, model=offline_model(calls, fail=RuntimeError("private body")))
    assert not isinstance(failure.value, OrientationSchemaError)
    assert len(calls) == 1 and settings.usage.requests == 1
    record = json.loads(next((tmp_path / "evidence").glob("orientation-reading-*.json")).read_text())
    assert record["status"] == "failed" and record["raw_response_parts"] == []


def test_changed_source_pixels_have_new_source_and_uploaded_proof(frame, tmp_path):
    calls = []
    first = run(frame, tmp_path, model=offline_model(calls))
    Image.new("RGB", (400, 600), "yellow").save(frame["path"])
    second = run(frame, tmp_path, model=offline_model(calls))
    assert len(calls) == 2 and first["proof"]["source_sha256"] != second["proof"]["source_sha256"]
    assert first["proof"]["panels"]["context_r0"]["upload_sha256"] != second["proof"]["panels"]["context_r0"]["upload_sha256"]


def test_unconfigured_provider_never_constructs_default_external_model(frame, tmp_path):
    with pytest.raises(AnalysisError, match="Explicit model and provider configuration"):
        asyncio.run(choose_reading_rotation(frame, source_id="source-23", page_id="leaf-7",
            config=replace(configuration(), base_url="", provider_kind="remote"),
            cache_dir=tmp_path/"evidence", trace_path=tmp_path/"trace.jsonl"))
    assert not (tmp_path / "trace.jsonl").exists()


@pytest.mark.parametrize("kind", ["budget", "fatal"])
def test_proof_write_failure_cannot_mask_budget_or_fatal_stop(frame, tmp_path, monkeypatch, kind):
    def failed_write(*args, **kwargs):
        raise OSError("Injected storage failure")
    monkeypatch.setattr("sever.image_orientation._write_evidence", failed_write)
    calls = []
    settings = configuration(max_requests=1)
    if kind == "budget":
        settings.usage.requests = 1
        expected = BatchBudgetExceeded
        model = offline_model(calls)
    else:
        expected = FatalProviderError
        model = offline_model(calls, fail=ModelHTTPError(400, "offline-model", "private body"))
    with pytest.raises(expected):
        run(frame, tmp_path, settings=settings, model=model)
    assert len(calls) == (0 if kind == "budget" else 1)


def test_schema_proof_write_failure_is_operational_not_schema_fallback(frame, tmp_path, monkeypatch):
    def failed_write(*args, **kwargs):
        raise OSError("Injected storage failure")
    monkeypatch.setattr("sever.image_orientation._write_evidence", failed_write)
    calls = []
    with pytest.raises(AnalysisError, match="Cannot preserve orientation failure proof") as failure:
        run(frame, tmp_path, model=offline_model(calls, '{"rotation_clockwise":45}'))
    assert not isinstance(failure.value, OrientationSchemaError) and len(calls) == 1


@pytest.mark.parametrize("storage_failure", [False, True])
def test_direct_fatal_provider_class_cannot_be_downgraded_or_masked(frame, tmp_path, monkeypatch, storage_failure):
    if storage_failure:
        def failed_write(*args, **kwargs):
            raise OSError("Injected storage failure")
        monkeypatch.setattr("sever.image_orientation._write_evidence", failed_write)
    calls = []
    settings = configuration()
    with pytest.raises(FatalProviderError, match="provider failure; stop batch"):
        run(frame, tmp_path, settings=settings, model=offline_model(calls,
            fail=FatalProviderError("PRIVATE-WRAPPER-PROVIDER-BODY")))
    assert len(calls) == 1 and settings.usage.requests == 1
    assert "PRIVATE-WRAPPER-PROVIDER-BODY" not in (tmp_path / "trace.jsonl").read_text()
    if not storage_failure:
        record = json.loads(next((tmp_path / "evidence").glob("orientation-reading-*.json")).read_text())
        assert record["status"] == "fatal" and record["http_status"] is None
        assert record["error_type"] == "FatalProviderError"


def owned_offline_client(monkeypatch, model, *, close_failure=False):
    clients = []
    class Client:
        def __init__(self, **kwargs):
            self.closed = 0
            clients.append(self)

        async def close(self):
            self.closed += 1
            if close_failure:
                raise OSError("PRIVATE-CLOSE-DETAIL")

    monkeypatch.setattr("sever.image_orientation.AsyncOpenAI", Client)
    monkeypatch.setattr("sever.image_orientation.OpenAIProvider", lambda **kwargs: object())
    monkeypatch.setattr("sever.image_orientation._RouterAIChatModel", lambda *args, **kwargs: model)
    return clients


def run_owned(frame, tmp_path, settings):
    return asyncio.run(choose_reading_rotation(frame, source_id="source-23", page_id="leaf-7",
        config=settings, cache_dir=tmp_path/"evidence", trace_path=tmp_path/"trace.jsonl"))


@pytest.mark.parametrize("kind", ["fatal", "budget_before", "budget_after", "ordinary"])
def test_client_cleanup_failure_preserves_existing_stop_or_operational_error(frame, tmp_path, monkeypatch, kind):
    calls = []
    settings = configuration(input_tokens_limit=51)
    if kind == "budget_before":
        settings.usage.input_tokens = 51
    error = FatalProviderError("PRIVATE-PROVIDER-BODY") if kind == "fatal" else RuntimeError("PRIVATE-PROVIDER-BODY") if kind == "ordinary" else None
    model = offline_model(calls, fail=error, input_tokens=52 if kind == "budget_after" else 50)
    clients = owned_offline_client(monkeypatch, model, close_failure=True)
    expected = FatalProviderError if kind == "fatal" else BatchBudgetExceeded if kind.startswith("budget") else AnalysisError
    match = "provider failure; stop batch" if kind == "fatal" else "budget exhausted" if kind.startswith("budget") else "RuntimeError"
    with pytest.raises(expected, match=match) as failure:
        run_owned(frame, tmp_path, settings)
    assert len(clients) == clients[0].closed == 1
    expected_requests = 0 if kind == "budget_before" else 1
    assert len(calls) == settings.usage.requests == expected_requests
    assert "PRIVATE" not in str(failure.value)
    record = json.loads(next((tmp_path/"evidence").glob("orientation-reading-*.json")).read_text())
    assert record["status"] == ("fatal" if kind == "fatal" else "budget_exhausted" if kind.startswith("budget") else "failed")
    if kind == "budget_after":
        assert record["raw_response_parts"] and settings.usage.input_tokens == 52


@pytest.mark.parametrize("schema_failure", [False, True])
def test_cleanup_failure_never_returns_success_or_permitted_schema_fallback(frame, tmp_path, monkeypatch, schema_failure):
    calls, settings = [], configuration()
    clients = owned_offline_client(monkeypatch,
        offline_model(calls, response='{}' if schema_failure else '{"rotation_clockwise":90}'),
        close_failure=True)
    with pytest.raises(AnalysisError, match="Cannot close orientation provider client") as failure:
        run_owned(frame, tmp_path, settings)
    assert not isinstance(failure.value, OrientationSchemaError)
    assert len(calls) == settings.usage.requests == 1
    assert len(clients) == clients[0].closed == 1
    assert "PRIVATE" not in str(failure.value)


@pytest.mark.parametrize("close_failure", [False, True])
def test_initial_trace_storage_failure_closes_owned_client_before_request(frame, tmp_path, monkeypatch, close_failure):
    calls = []
    clients = owned_offline_client(monkeypatch, offline_model(calls), close_failure=close_failure)
    def failed_trace(*args, **kwargs):
        raise OSError("PRIVATE-STORAGE-DETAIL")
    monkeypatch.setattr("sever.image_orientation._trace", failed_trace)
    with pytest.raises(AnalysisError, match="Cannot preserve orientation failure proof") as failure:
        run_owned(frame, tmp_path, configuration())
    assert calls == [] and len(clients) == clients[0].closed == 1
    assert "PRIVATE" not in str(failure.value)
