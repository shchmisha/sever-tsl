import asyncio
from dataclasses import replace
import hashlib
from io import BytesIO
import json

from PIL import Image, ImageDraw
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_reader import ImageReading, _prepare_panels, _validate_reading, read_frame
from sever.schema import FIELDS_BY_TYPE


@pytest.fixture
def source_frame(tmp_path):
    path = tmp_path / "private-name-not-model-input.png"
    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((7, 19, 177, 299), fill="navy")
    image.save(path)
    return {"frame_id": "real-frame-23", "path": str(path), "timestamp_s": 2.3}


def field(raw=None, *, status="unreadable", text="", panels=None, blank=False):
    return {"raw_value": raw, "status": status, "confidence": .88,
            "evidence_text": text, "panel_ids": panels or [],
            "explicit_blank": blank, "reason": None}


def reading(kind="unknown", *, heading=None, fields=None, complete=False):
    headings = {"unknown": "CMR", "upd": "Универсальный передаточный документ",
                "torg12": "Унифицированная форма ТОРГ-12", "ttn": "Товарно-транспортная накладная",
                "trn": "Транспортная накладная"}
    heading = headings[kind] if heading is None else heading
    defaults = {name: field() for name in FIELDS_BY_TYPE[kind]}
    defaults.update(fields or {})
    return {"document_type": kind, "type_confidence": .9,
            **({"form_scope": "insufficient_evidence"} if kind == "unknown" else {}),
            "observed_form": heading, "visible_text": heading + "\nВидимая строка документа",
            "heading_text": heading, "heading_panel_ids": ["context_r90"],
            "complete_page_visible": complete, "fields": defaults}


def config(**changes):
    return AgentConfig(model_name="qwen-test-local", vision_model_name="qwen-test-local",
                       provider_kind="local", base_url="http://127.0.0.1:8090/v1",
                       document_reader="native_image", max_output_tokens=4096,
                       **changes)


def model_for(payload, calls, *, model_name="pixel-test-model", fail=None, input_tokens=50):
    def respond(messages, info):
        calls.append((messages, info))
        assert info.output_tools == []
        assert info.function_tools == []
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_request_parameters.output_object.strict is False
        if fail:
            raise fail
        assert "reading" in info.model_request_parameters.output_object.json_schema["properties"]
        return ModelResponse(parts=[TextPart(json.dumps({"reading": payload}, ensure_ascii=False))],
                             usage=RequestUsage(input_tokens=input_tokens, output_tokens=20))
    return FunctionModel(respond, model_name=model_name)


def run(frame, tmp_path, payload=None, *, settings=None, model=None):
    return asyncio.run(read_frame(frame, source_id="source-23", page_id="leaf-7",
        config=settings or config(), model=model or model_for(payload or reading(), []),
        cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))


def validate(frame, tmp_path, payload):
    _, proof = _prepare_panels(frame, tmp_path / "panels")
    return _validate_reading(ImageReading.model_validate(payload), proof,
        source_id="source-23", page_id="leaf-7", model_identity="pixel-test-model")


def trace_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]


def test_unknown_still_reads_actual_pixels_and_persists_literal_response(source_frame, tmp_path):
    calls = []
    payload = reading()
    payload["visible_text"] = "CMR\nОтправитель: печатная строка"
    result = run(source_frame, tmp_path, payload, model=model_for(payload, calls))
    assert len(calls) == 1
    assert result["document"].document_type == "unknown"
    assert result["document"].fields == {}
    assert result["recognition"]["observed_form"] == "CMR"
    assert result["recognition"]["visible_text"] == payload["visible_text"]
    assert result["recognition"]["origin"] == "model_pixel_transcription"
    content = [item for message in calls[0][0] for part in message.parts
               if isinstance(getattr(part, "content", None), list) for item in part.content]
    images = [item for item in content if isinstance(item, BinaryContent)]
    assert len(images) == 12
    supplied = [item for item in content if isinstance(item, str)]
    assert source_frame["path"] not in "\n".join(supplied)
    assert "source-23" not in "\n".join(supplied)
    proof = result["recognition"]["pixel_proof"]
    assert proof["source_sha256"] == hashlib.sha256(open(source_frame["path"], "rb").read()).hexdigest()
    for actual, panel in zip(images, proof["panels"].values(), strict=True):
        assert hashlib.sha256(actual.data).hexdigest() == panel["upload_sha256"]
        assert actual.data == open(panel["upload_path"], "rb").read()
        assert Image.open(BytesIO(actual.data)).size == tuple(panel["upload_pixel_size"])
    cache = json.loads(next((tmp_path / "cache").glob("image-reading-*.json")).read_text())
    assert cache["decision"] == payload
    assert json.loads(cache["raw_response_parts"][0]["content"]) == {"reading": payload}
    events = [row["event"] for row in trace_rows(tmp_path)]
    assert events == ["image_reader_observation", "model_request_started", "model_response_received", "image_reader_reading"]
    response = next(row for row in trace_rows(tmp_path) if row["event"] == "model_response_received")
    assert response["billing"]["kind"] == "local_inference"


@pytest.mark.parametrize("kind", ["upd", "torg12", "ttn", "trn"])
def test_exact_contract_for_every_supported_form(source_frame, tmp_path, kind):
    result = run(source_frame, tmp_path, reading(kind))["document"]
    assert result.document_type == kind
    assert list(result.fields) == FIELDS_BY_TYPE[kind]
    assert all(value.status == "unreadable" and value.raw_value is None for value in result.fields.values())
    assert result.classification_evidence[0].frame_id == source_frame["frame_id"]
    assert result.classification_evidence[0].region_index is None
    assert result.classification_evidence[0].bbox == [0., 0., 1., 1.]


@pytest.mark.parametrize("scope", ["out_of_scope", "insufficient_evidence"])
def test_unknown_scope_is_a_model_decision_and_does_not_discard_transcription(source_frame, tmp_path, scope):
    payload = reading("unknown")
    payload["form_scope"] = scope
    result = run(source_frame, tmp_path, payload)
    assert result["recognition"]["form_scope"] == scope
    assert result["recognition"]["visible_text"] == payload["visible_text"]
    assert result["document"].fields == {}
    assert result["document"].errors == []


def test_recognized_value_retains_literal_quote_and_source_panel_box(source_frame, tmp_path):
    payload = reading("upd", fields={"invoice_number": field("А-0042", status="recognized",
        text="Счет-фактура № А-0042 от 15.02.2020", panels=["detail_0_r270"]),
        "invoice_date": field("15.02.2020", status="recognized",
        text="Счет-фактура № А-0042 от 15.02.2020", panels=["detail_0_r270"])})
    result = run(source_frame, tmp_path, payload)["document"]
    assert result.fields["invoice_number"].raw_value == "А-0042"
    assert result.fields["invoice_date"].normalized_value == "2020-02-15"
    evidence = result.fields["invoice_number"].evidence[0]
    assert evidence.bbox == [0., 0., 1., .6]
    assert evidence.region_index is None
    assert evidence.text == payload["fields"]["invoice_number"]["evidence_text"]


def test_every_call_reads_fresh_pixels_and_keeps_distinct_write_only_evidence(source_frame, tmp_path):
    calls = []
    settings = config()
    model = model_for(reading(), calls)
    first = run(source_frame, tmp_path, settings=settings, model=model)
    second = run(source_frame, tmp_path, settings=settings, model=model)
    assert first == second
    assert len(calls) == 2
    assert settings.usage.requests == 2
    assert settings.usage.attempted_requests == 2
    assert settings.usage.input_tokens == 100
    assert not any(row["event"] == "image_reader_cache_hit" for row in trace_rows(tmp_path))
    assert len(list((tmp_path / "cache").glob("image-reading-*.json"))) == 2
    run(source_frame, tmp_path, settings=replace(settings, refresh_cache=True), model=model)
    assert len(calls) == 3
    assert settings.usage.requests == 3
    assert settings.usage.attempted_requests == 3


def test_changed_source_pixels_and_model_have_distinct_evidence_proofs(source_frame, tmp_path):
    calls = []
    settings = config()
    run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls))
    Image.new("RGB", (400, 600), "yellow").save(source_frame["path"])
    run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls))
    run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls, model_name="different-model"))
    assert len(calls) == 3
    assert len(list((tmp_path / "cache").glob("image-reading-*.json"))) == 3


@pytest.mark.parametrize("limit_kind", ["requests", "input_tokens", "output_tokens", "max_tokens"])
def test_exhausted_shared_budget_stops_before_actual_call(source_frame, tmp_path, limit_kind):
    calls = []
    settings = config(max_requests=1) if limit_kind == "requests" else config(max_tokens=1) if limit_kind == "max_tokens" else config()
    if limit_kind == "requests":
        settings.usage.requests = 1
    elif limit_kind in {"input_tokens", "output_tokens"}:
        setattr(settings.usage, limit_kind, getattr(settings, limit_kind + "_limit"))
    with pytest.raises(BatchBudgetExceeded):
        run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls))
    assert calls == []
    assert not list((tmp_path / "cache").glob("image-reading-*.json"))


def test_provider_failure_is_charged_once_and_private_body_not_saved(source_frame, tmp_path):
    calls = []
    settings = config()
    model = model_for(reading(), calls, fail=RuntimeError("provider-private-body-marker"))
    with pytest.raises(AnalysisError, match="RuntimeError"):
        run(source_frame, tmp_path, settings=settings, model=model)
    assert settings.usage.requests == 1
    assert settings.usage.attempted_requests == 1
    with pytest.raises(AnalysisError, match="RuntimeError"):
        run(source_frame, tmp_path, settings=settings, model=model)
    assert len(calls) == 2
    assert settings.usage.requests == 2
    assert "provider-private-body-marker" not in (tmp_path / "trace.jsonl").read_text()
    assert "provider-private-body-marker" not in next((tmp_path / "cache").glob("image-reading-*.json")).read_text()


def test_fatal_provider_failure_stops_each_call_without_replaying_old_failure(source_frame, tmp_path):
    calls = []
    model = model_for(reading(), calls, fail=ModelHTTPError(400, "pixel-test-model", "private body"))
    with pytest.raises(FatalProviderError, match="HTTP 400"):
        run(source_frame, tmp_path, model=model)
    with pytest.raises(FatalProviderError, match="HTTP 400"):
        run(source_frame, tmp_path, model=model)
    assert len(calls) == 2


def _fail_reader_storage(monkeypatch, *, trace=False, evidence=False):
    from sever import image_reader
    original_trace = image_reader._trace
    if trace:
        def failed_error_trace(path, event, *args, **kwargs):
            if event == "image_reader_error":
                raise OSError("PRIVATE-STORAGE-DETAIL")
            return original_trace(path, event, *args, **kwargs)
        monkeypatch.setattr(image_reader, "_trace", failed_error_trace)
    if evidence:
        def failed_evidence(*args, **kwargs):
            raise OSError("PRIVATE-STORAGE-DETAIL")
        monkeypatch.setattr(image_reader, "_write_evidence", failed_evidence)


@pytest.mark.parametrize("storage", ["normal", "trace", "evidence", "both"])
def test_direct_fatal_class_stops_even_when_failure_proof_cannot_be_saved(source_frame, tmp_path, monkeypatch, storage):
    _fail_reader_storage(monkeypatch, trace=storage in {"trace", "both"},
                         evidence=storage in {"evidence", "both"})
    calls, settings = [], config()
    with pytest.raises(FatalProviderError, match="provider failure; stop batch") as failure:
        run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls,
            fail=FatalProviderError("PRIVATE-PROVIDER-BODY")))
    assert len(calls) == settings.usage.requests == settings.usage.attempted_requests == 1
    assert "PRIVATE" not in str(failure.value)
    assert "PRIVATE-PROVIDER-BODY" not in (tmp_path / "trace.jsonl").read_text()
    if storage in {"normal", "trace"}:
        record = json.loads(next((tmp_path / "cache").glob("image-reading-*.json")).read_text())
        assert record["status"] == "fatal" and record["http_status"] is None
        assert record["error_type"] == "FatalProviderError"


@pytest.mark.parametrize("status", [400, 401, 402, 403])
@pytest.mark.parametrize("storage", ["trace", "evidence", "both"])
def test_http_fatal_stop_survives_each_failure_storage_fault(source_frame, tmp_path, monkeypatch, status, storage):
    _fail_reader_storage(monkeypatch, trace=storage in {"trace", "both"},
                         evidence=storage in {"evidence", "both"})
    calls = []
    with pytest.raises(FatalProviderError, match=f"HTTP {status}"):
        run(source_frame, tmp_path, model=model_for(reading(), calls,
            fail=ModelHTTPError(status, "pixel-test-model", "PRIVATE-PROVIDER-BODY")))
    assert len(calls) == 1


@pytest.mark.parametrize("stage", ["requests", "tokens", "received"])
def test_budget_stop_survives_failure_trace_storage_fault(source_frame, tmp_path, monkeypatch, stage):
    _fail_reader_storage(monkeypatch, trace=True, evidence=True)
    calls = []
    settings = config(max_requests=1) if stage == "requests" else config(input_tokens_limit=51) if stage == "received" else config()
    if stage == "requests":
        settings.usage.requests = 1
    elif stage == "tokens":
        settings.usage.input_tokens = settings.input_tokens_limit
    with pytest.raises(BatchBudgetExceeded):
        run(source_frame, tmp_path, settings=settings,
            model=model_for(reading(), calls, input_tokens=52))
    assert len(calls) == (1 if stage == "received" else 0)
    assert not list((tmp_path / "cache").glob("image-reading-*.json"))
    if stage == "received":
        assert settings.usage.requests == 1 and settings.usage.input_tokens == 52


@pytest.mark.parametrize("storage", ["trace", "evidence", "both"])
def test_ordinary_failure_proof_storage_fault_is_visible_without_retry(source_frame, tmp_path, monkeypatch, storage):
    _fail_reader_storage(monkeypatch, trace=storage in {"trace", "both"},
                         evidence=storage in {"evidence", "both"})
    calls, settings = [], config()
    with pytest.raises(AnalysisError, match="Cannot preserve image-reading failure proof") as failure:
        run(source_frame, tmp_path, settings=settings,
            model=model_for(reading(), calls, fail=RuntimeError("PRIVATE-PROVIDER-BODY")))
    assert not isinstance(failure.value, (FatalProviderError, BatchBudgetExceeded))
    assert "PRIVATE" not in str(failure.value)
    assert len(calls) == settings.usage.requests == 1


def _owned_offline_client(monkeypatch, model, *, close_failure=False):
    clients = []
    class Client:
        def __init__(self, **kwargs):
            self.closed = 0
            clients.append(self)
        async def close(self):
            self.closed += 1
            if close_failure:
                raise OSError("PRIVATE-CLOSE-DETAIL")
    monkeypatch.setattr("sever.image_reader.AsyncOpenAI", Client)
    monkeypatch.setattr("sever.image_reader.OpenAIProvider", lambda **kwargs: object())
    monkeypatch.setattr("sever.image_reader._RouterAIChatModel", lambda *args, **kwargs: model)
    return clients


@pytest.mark.parametrize("close_failure", [False, True])
def test_initial_observation_storage_failure_closes_owned_client_before_any_request(source_frame, tmp_path, monkeypatch, close_failure):
    calls = []
    clients = _owned_offline_client(monkeypatch, model_for(reading(), calls), close_failure=close_failure)
    def failed_trace(*args, **kwargs):
        raise OSError("PRIVATE-STORAGE-DETAIL")
    monkeypatch.setattr("sever.image_reader._trace", failed_trace)
    with pytest.raises(AnalysisError, match="Cannot preserve image-reading failure proof") as failure:
        asyncio.run(read_frame(source_frame, source_id="s", page_id="p", config=config(),
            cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))
    assert calls == [] and len(clients) == 1 and clients[0].closed == 1
    assert "PRIVATE" not in str(failure.value)


@pytest.mark.parametrize("kind", ["fatal", "budget", "ordinary", "success"])
def test_client_cleanup_failure_never_masks_an_existing_stop(source_frame, tmp_path, monkeypatch, kind):
    calls, settings = [], config()
    error = FatalProviderError("PRIVATE-PROVIDER-BODY") if kind == "fatal" else RuntimeError("PRIVATE-PROVIDER-BODY") if kind == "ordinary" else None
    clients = _owned_offline_client(monkeypatch, model_for(reading(), calls, fail=error), close_failure=True)
    if kind == "budget":
        settings.usage.input_tokens = settings.input_tokens_limit
    expected = FatalProviderError if kind == "fatal" else BatchBudgetExceeded if kind == "budget" else AnalysisError
    match = "provider failure; stop batch" if kind == "fatal" else "budget exhausted" if kind == "budget" else "RuntimeError" if kind == "ordinary" else "Cannot close image provider client"
    with pytest.raises(expected, match=match) as failure:
        asyncio.run(read_frame(source_frame, source_id="s", page_id="p", config=settings,
            cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))
    assert len(clients) == 1 and clients[0].closed == 1
    assert len(calls) == (0 if kind == "budget" else 1)
    assert "PRIVATE" not in str(failure.value)


def test_alternative_pixel_preparer_keeps_native_model_contract_and_source_binding(source_frame, tmp_path):
    calls, preparations = [], []
    digest = hashlib.sha256(open(source_frame["path"], "rb").read()).hexdigest()
    def context_only(frame, cache):
        preparations.append((frame, cache))
        uploads, proof = _prepare_panels(frame, cache)
        selected = {"context_r90"}
        return [(name, data) for name, data in uploads if name in selected], {
            **proof, "panels": {name: panel for name, panel in proof["panels"].items() if name in selected}}
    payload = reading()
    result = asyncio.run(read_frame(source_frame, source_id="s", page_id="p", config=config(),
        cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl",
        panel_preparer=context_only, expected_source_sha256=digest, model=model_for(payload, calls)))
    assert len(preparations) == len(calls) == 1
    content = [item for message in calls[0][0] for part in message.parts
               if isinstance(getattr(part, "content", None), list) for item in part.content]
    assert len([item for item in content if isinstance(item, BinaryContent)]) == 1
    assert digest not in "\n".join(item for item in content if isinstance(item, str))
    assert result["recognition"]["reading"] == payload
    assert result["recognition"]["pixel_proof"]["source_sha256"] == digest


@pytest.mark.parametrize("changed", ["expected_sha", "proof_sha", "frame_id", "source_path", "source_during_prepare"])
def test_source_or_frame_mismatch_prevents_field_model_call(source_frame, tmp_path, changed):
    calls = []
    expected = hashlib.sha256(open(source_frame["path"], "rb").read()).hexdigest()
    def prepare(frame, cache):
        uploads, proof = _prepare_panels(frame, cache)
        if changed == "proof_sha":
            proof["source_sha256"] = "0" * 64
        elif changed == "frame_id":
            proof["frame_id"] = "foreign-frame"
        elif changed == "source_path":
            proof["source_path"] = str(tmp_path / "foreign-source.png")
        elif changed == "source_during_prepare":
            Image.new("RGB", (400, 600), "yellow").save(frame["path"])
        return uploads, proof
    with pytest.raises(AnalysisError, match="differs from the expected source"):
        asyncio.run(read_frame(source_frame, source_id="s", page_id="p", config=config(),
            cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl",
            panel_preparer=prepare, expected_source_sha256="0" * 64 if changed == "expected_sha" else expected,
            model=model_for(reading(), calls)))
    assert calls == []
    assert not list((tmp_path / "cache").glob("image-reading-*.json"))


def test_model_budget_overrun_is_a_batch_stop_not_an_ordinary_page_failure(source_frame, tmp_path):
    settings = config(input_tokens_limit=51)
    calls = []
    with pytest.raises(BatchBudgetExceeded):
        run(source_frame, tmp_path, settings=settings, model=model_for(reading(), calls, input_tokens=52))
    assert len(calls) == 1
    assert settings.usage.input_tokens == 52
    assert settings.usage.requests == 1


def test_malformed_native_response_retains_received_raw_response_no_retry(source_frame, tmp_path):
    calls = []
    payload = reading("upd")
    del payload["fields"]["invoice_date"]
    with pytest.raises(AnalysisError, match="UnexpectedModelBehavior"):
        run(source_frame, tmp_path, model=model_for(payload, calls))
    assert len(calls) == 1
    rows = trace_rows(tmp_path)
    assert any(row["event"] == "model_response_received" and row["response_parts"][0]["content"] for row in rows)


def test_missing_source_makes_no_model_call_and_no_input_proof(tmp_path):
    calls = []
    with pytest.raises(AnalysisError, match="no model call made"):
        run({"frame_id": "f", "path": str(tmp_path / "missing.png")}, tmp_path, model=model_for(reading(), calls))
    assert calls == []


@pytest.mark.parametrize("size", [(400, 600), (600, 400)])
def test_panel_transforms_round_trip_source_corners_and_preserve_real_crop(tmp_path, size):
    path = tmp_path / "frame.png"
    Image.new("RGB", size, "white").save(path)
    _, proof = _prepare_panels({"path": str(path), "frame_id": "f"}, tmp_path / "cache")
    for panel in proof["panels"].values():
        left, top, right, bottom = panel["source_bbox_pixels_top_left"]
        forward, reverse = panel["source_to_upload_pixels"], panel["upload_to_source_pixels"]
        points = []
        for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]:
            u, v = [row[0] * x + row[1] * y + row[2] for row in forward[:2]]
            points.append((u, v))
            assert [row[0] * u + row[1] * v + row[2] for row in reverse[:2]] == pytest.approx([x, y])
        assert min(p[0] for p in points) == pytest.approx(0)
        assert min(p[1] for p in points) == pytest.approx(0)
        assert max(p[0] for p in points) == pytest.approx(panel["upload_pixel_size"][0])
        assert max(p[1] for p in points) == pytest.approx(panel["upload_pixel_size"][1])


def test_native_schema_requires_all_field_properties_and_rejects_extra_contract_values():
    payload = reading("torg12")
    del payload["fields"]["supplier"]["explicit_blank"]
    with pytest.raises(ValidationError):
        ImageReading.model_validate(payload)
    payload = reading("unknown")
    payload["fields"]["invoice_number"] = field()
    with pytest.raises(ValidationError):
        ImageReading.model_validate(payload)


def test_panel_recipe_preserves_new_detail_limit_in_provenance(source_frame, tmp_path):
    _, proof = _prepare_panels(source_frame, tmp_path / "cache")
    assert {panel["prep"]["max_edge"] for key, panel in proof["panels"].items() if key.startswith("context_")} == {1024}
    assert {panel["prep"]["max_edge"] for key, panel in proof["panels"].items() if key.startswith("detail_")} == {2400}


def test_prompt_guides_literal_form_codes_and_distinct_own_header_without_source_hints(source_frame, tmp_path):
    from sever.image_reader import IMAGE_READER_VERSION, IMAGE_SCHEMA_VERSION, INSTRUCTIONS
    assert "ОКУД" in INSTRUCTIONS and "Коды block BEFORE" in INSTRUCTIONS.replace("\n", " ")
    assert "standalone СЧЕТ-ФАКТУРА" in INSTRUCTIONS
    assert "observed_form quotes the literal" in INSTRUCTIONS
    assert source_frame["path"] not in INSTRUCTIONS
    result = run(source_frame, tmp_path, reading())
    cache = json.loads(next((tmp_path / "cache").glob("image-reading-*.json")).read_text())
    assert cache["metadata"]["version"] == IMAGE_READER_VERSION
    assert cache["metadata"]["response_schema_version"] == IMAGE_SCHEMA_VERSION
    assert result["recognition"]["reading"]["document_type"] == "unknown"


@pytest.mark.parametrize("stored_kind", ["valid", "altered", "malformed", "fatal"])
def test_saved_answers_cannot_skip_model_or_supply_current_recognition(source_frame, tmp_path, stored_kind):
    calls = []
    settings = config(refresh_cache=False)
    previous = reading("upd", fields={"invoice_number": field("014", status="recognized",
        text="Счет-фактура № 014 от 11.03.2023", panels=["context_r0"])})
    run(source_frame, tmp_path, settings=settings, model=model_for(previous, calls))
    old_record = next((tmp_path / "cache").glob("image-reading-*.json"))
    original_bytes = old_record.read_bytes()
    saved = json.loads(original_bytes)
    if stored_kind == "altered":
        saved["decision"]["fields"]["invoice_number"]["raw_value"] = "999"
        saved["decision"]["fields"]["invoice_number"]["evidence_text"] = "Счет-фактура № 999"
    elif stored_kind == "fatal":
        saved["status"] = "fatal"
    stored_text = "malformed historical answer" if stored_kind == "malformed" else json.dumps(saved, ensure_ascii=False)
    # Reproduce both historical deterministic answer names and the new output
    # record. Neither may become an input to the next actual model call.
    legacy_path = old_record.with_name(old_record.stem.rsplit("-", 1)[0] + ".json")
    legacy_path.write_text(stored_text)
    old_record.write_text(stored_text)
    fresh = reading("upd", fields={"invoice_number": field("025", status="recognized",
        text="Счет-фактура № 025 от 11.03.2023", panels=["context_r0"])})
    result = run(source_frame, tmp_path, settings=settings, model=model_for(fresh, calls))
    assert len(calls) == 2 and settings.usage.requests == 2
    assert result["document"].fields["invoice_number"].raw_value == "025"
    assert result["recognition"]["reading"] == fresh
    assert legacy_path.read_text() == old_record.read_text() == stored_text
    assert len(list((tmp_path / "cache").glob("image-reading-*.json"))) == 3
    assert not any(row["event"] == "image_reader_cache_hit" for row in trace_rows(tmp_path))
    for messages, _ in calls:
        images = [item for message in messages for part in message.parts
                  if isinstance(getattr(part, "content", None), list) for item in part.content
                  if isinstance(item, BinaryContent)]
        assert len(images) == 12


@pytest.mark.parametrize("kind,heading", [
    ("upd", "СЧЕТ-ФАКТУРА"), ("torg12", "Товарная накладная"),
    ("ttn", "ТН-2"), ("trn", "CMR"),
])
def test_form_semantics_belong_to_model_even_when_literal_heading_disagrees(source_frame, tmp_path, kind, heading):
    payload = reading(kind, heading=heading)
    payload["visible_text"] = "Другой текст: модель не повторила заголовок"
    result = validate(source_frame, tmp_path, payload)
    assert result["document"].document_type == kind
    assert result["document"].type_confidence == payload["type_confidence"]
    assert result["document"].errors == []
    assert result["recognition"]["reading"] == payload
    assert result["recognition"]["semantic_validation"] == "model"


@pytest.mark.parametrize("kind,name,raw,quote", [
    ("upd", "invoice_number", "2020", "Счет-фактура № 0042 от 15.02.2020"),
    ("upd", "invoice_number", "A-42", "Счет-фактура № А-42"),
    ("upd", "invoice_number", "123", "Счет-фактура № 12/3"),
    ("upd", "invoice_date", "16.02.2020", "Счет-фактура № 0042 от 15.02.2020"),
    ("upd", "consignor", "он же", "Грузоотправитель: он же"),
    ("upd", "consignor_address", 'ООО «Сосна»', "Покупатель: ООО «Сосна»"),
    ("upd", "consignee_address", "Москва, ул. Иная, д. 4", "Адрес: Москва, ул. Иная, д. 4"),
    ("trn", "document_number", "77", "Заказ (заявка) № 77"),
    ("ttn", "document_number", "17", "Утверждена постановлением № 17"),
    ("torg12", "supplier", 'ООО «Береза»', "Грузополучатель: ООО «Ива»"),
    ("trn", "customer", 'ООО «Кедр»', "Перевозчик: ООО «Кедр»"),
    ("upd", "invoice_date", "31.02.2020", "Дата 31.02.2020"),
    ("ttn", "document_date", "02.06.99", "02.06.99"),
    ("trn", "document_number", "№", "№"),
    ("upd", "consignee", "  ООО «Ива»\n", ""),
])
def test_model_recognized_raw_is_never_filtered_or_repaired_using_caption_semantics(source_frame, tmp_path, kind, name, raw, quote):
    proposal = field(raw, status="recognized", text=quote, panels=["detail_1_r270"])
    proposal["reason"] = "Решение модели по изображению"
    payload = reading(kind, fields={name: proposal})
    payload["visible_text"] = "Разметка заголовка отличается от поля\nПродавец: ООО «Другой»"
    result = validate(source_frame, tmp_path, payload)
    value = result["document"].fields[name]
    assert value.status == "recognized"
    assert value.raw_value == raw
    assert value.confidence == proposal["confidence"] and value.reason == proposal["reason"]
    assert value.evidence[0].text == quote
    assert value.evidence[0].bbox == [0., .4, 1., .6]
    assert result["document"].errors == []
    assert result["recognition"]["reading"] == payload
    assert result["recognition"]["literal_recoveries"] == []


@pytest.mark.parametrize("name,raw", [("invoice_date", "31.02.2020"), ("invoice_date", "02.06.99"), ("invoice_number", "№")])
def test_failed_format_normalization_cannot_reject_model_recognition(source_frame, tmp_path, name, raw):
    payload = reading("upd", fields={name: field(raw, status="recognized", text="", panels=["context_r0"])})
    value = validate(source_frame, tmp_path, payload)["document"].fields[name]
    assert value.status == "recognized" and value.raw_value == raw
    assert value.normalized_value is None


def test_no_reference_recovery_or_raw_value_completion_occurs(source_frame, tmp_path):
    payload = reading("upd", fields={
        "consignor": field("он же", status="recognized", text="он же", panels=["context_r90"]),
        "consignor_address": field(status="unreadable", text="Адрес не прочитан", panels=["context_r90"]),
    })
    payload["visible_text"] = ('Продавец: ООО «Сосна»\nАдрес: Москва, ул. Зеленая, д. 2\n'
        'Грузоотправитель и его адрес: он же')
    result = run(source_frame, tmp_path, payload)
    assert result["document"].fields["consignor"].raw_value == "он же"
    assert result["document"].fields["consignor_address"].status == "unreadable"
    assert result["document"].fields["consignor_address"].raw_value is None
    assert result["recognition"]["reading"] == payload
    assert result["recognition"]["literal_recoveries"] == []
    assert result["recognition"]["semantic_validation"] == "model"
    assert result["recognition"]["response_reuse"] is False
    record = json.loads(next((tmp_path / "cache").glob("image-reading-*.json")).read_text())
    assert record["decision"] == payload
    assert json.loads(record["raw_response_parts"][0]["content"]) == {"reading": payload}
    assert record["literal_recoveries"] == []
    assert record["metadata"]["semantic_validation"] == "model"


@pytest.mark.parametrize("heading_ids", [[], ["foreign-frame"]])
def test_heading_panel_defect_is_reported_without_reclassifying_model_type(source_frame, tmp_path, heading_ids):
    payload = reading("trn", heading="", fields={"document_number": field("77", status="recognized",
        text="", panels=["context_r0"])})
    payload["heading_panel_ids"] = heading_ids
    result = validate(source_frame, tmp_path, payload)
    assert result["document"].document_type == "trn"
    assert result["document"].classification_evidence == []
    assert result["document"].fields["document_number"].status == "recognized"
    assert result["document"].errors
    assert result["recognition"]["reading"] == payload


def test_empty_model_heading_and_quotes_with_real_panels_do_not_change_type_or_values(source_frame, tmp_path):
    payload = reading("torg12", heading="", fields={"supplier": field('ООО «Ива»', status="recognized",
        text="", panels=["detail_0_r0"])})
    payload["visible_text"] = ""
    result = validate(source_frame, tmp_path, payload)
    assert result["document"].document_type == "torg12"
    assert result["document"].classification_evidence[0].text == ""
    assert result["document"].fields["supplier"].raw_value == 'ООО «Ива»'
    assert result["document"].errors == []


@pytest.mark.parametrize("panels", [[], ["foreign-frame"], ["context_r0", "foreign-frame"]])
def test_recognized_field_requires_only_real_supplied_panel_provenance(source_frame, tmp_path, panels):
    payload = reading("upd", fields={"invoice_number": field("77", status="recognized", text="", panels=panels)})
    result = validate(source_frame, tmp_path, payload)
    value = result["document"].fields["invoice_number"]
    assert value.status == "unreadable" and value.raw_value is None
    assert value.evidence == []
    assert result["document"].errors
    assert result["recognition"]["reading"] == payload


@pytest.mark.parametrize("raw,status,blank", [
    (None, "recognized", False), ("", "recognized", False), ("   ", "recognized", False),
    ("77", "recognized", True), ("77", "not_present", True),
    (None, "not_present", False), ("77", "unreadable", False), ("77", "conflict", False),
])
def test_status_and_raw_value_structure_is_enforced_without_semantic_rules(source_frame, tmp_path, raw, status, blank):
    payload = reading("trn", fields={"document_number": field(raw, status=status, text="",
        panels=["context_r0"], blank=blank)})
    result = validate(source_frame, tmp_path, payload)
    value = result["document"].fields["document_number"]
    assert value.status == "unreadable" and value.raw_value is None
    assert result["document"].errors
    assert result["recognition"]["reading"] == payload


@pytest.mark.parametrize("status", ["unreadable", "conflict"])
def test_unresolved_model_status_is_preserved_with_null_value(source_frame, tmp_path, status):
    payload = reading("trn", fields={"document_date": field(status=status, text="", panels=[])})
    result = validate(source_frame, tmp_path, payload)
    assert result["document"].fields["document_date"].status == status
    assert result["document"].errors == []


def test_model_blank_decision_needs_no_caption_regex_or_complete_page_filter(source_frame, tmp_path):
    payload = reading("trn", complete=False, fields={"document_number": field(status="not_present",
        text="", panels=["detail_1_r90"], blank=True)})
    result = validate(source_frame, tmp_path, payload)
    value = result["document"].fields["document_number"]
    assert value.status == "not_present" and value.raw_value is None
    assert value.evidence[0].bbox == [0., .4, 1., .6]
    assert result["document"].errors == []
    assert result["recognition"]["complete_page_visible"] is False


def test_supplied_panel_id_cannot_claim_a_different_source_frame(source_frame, tmp_path):
    payload = reading("ttn", fields={"document_number": field("77", status="recognized",
        text="", panels=["detail_0_r90"])})
    _, proof = _prepare_panels(source_frame, tmp_path / "panels")
    proof["panels"]["detail_0_r90"]["frame_id"] = "foreign-frame"
    result = _validate_reading(ImageReading.model_validate(payload), proof,
        source_id="source-23", page_id="leaf-7", model_identity="pixel-test-model")
    assert result["document"].fields["document_number"].status == "unreadable"
    assert result["document"].fields["document_number"].evidence == []
