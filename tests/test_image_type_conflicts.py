"""Fresh joint form decisions use all actual source frames without prior types."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from sever.agents import AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_conflicts import read_leaf_conflicts
from test_image_conflicts import (config, content, frames, panel_ids, records,
                                 response, zooms)


def type_decision(messages, *, kind="ttn"):
    return {"resolution_reason": "Fresh own headings were read on every actual image",
            "readings": [{"frame_id": label, "document_type": kind,
                          "heading_text": "Literal own heading Ω — exact text", "panel_ids": [panel]}
                         for label, panel in panel_ids(messages).items()]}


def run(frames, tmp_path, fn, *, settings=None, document_type="unknown", requested=()):
    return asyncio.run(read_leaf_conflicts(frames, source_id="private-source-id", page_id="private-page-id",
        document_type=document_type, unresolved_fields=list(requested), config=settings or config(),
        cache_dir=tmp_path / "evidence", trace_path=tmp_path / "trace.jsonl",
        model=FunctionModel(fn, model_name="offline-type-review")))


@pytest.mark.parametrize("kind", ["upd", "torg12", "ttn", "trn"])
@pytest.mark.parametrize("count", [2, 3, 4])
def test_every_actual_frame_agrees_type_and_literal_headings_reach_export_exactly(frames, tmp_path, kind, count):
    for index in range(2, count):
        path = tmp_path / f"extra-source-{index}.png"
        Image.new("RGB", (160, 100), (index * 20, 30, 40)).save(path)
        frames.append({"frame_id": f"actual-source-{index}", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timestamp_s": index + .5, "rotation_clockwise_degrees": 90})
    sent = []
    def answer(messages, info):
        assert [tool.name for tool in info.output_tools] == ["submit_conflict_review"]
        schema = info.output_tools[0].parameters_json_schema
        assert set(schema["properties"]) == {"resolution_reason", "readings"}
        assert "fields" not in schema["properties"]
        payload = type_decision(messages, kind=kind)
        for index, reading in enumerate(payload["readings"]):
            reading["heading_text"] += f" {index}"
        sent.append(deepcopy(payload))
        return response(payload)
    result = run(frames, tmp_path, answer)
    assert result["document_type"] == kind and result["fields"] == {}
    assert result["adjudication"]["status"] == "resolved"
    assert result["adjudication"]["resolved_document_type"] == kind
    assert result["adjudication"]["errors"] == []
    assert result["recognition"]["raw_decision"] == sent[0]
    assert result["recognition"]["metadata"]["review_mode"] == "document_type"
    assert {item["frame_id"] for item in result["classification_evidence"]} == {frame["frame_id"] for frame in frames}
    assert [item["text"] for item in result["classification_evidence"]] == [item["heading_text"] for item in sent[0]["readings"]]
    record = records(tmp_path)[0]
    assert record["document_type"] == kind
    assert record["classification_evidence"] == result["classification_evidence"]
    assert record["raw_decision"] == sent[0]


@pytest.mark.parametrize("case", ["different_types", "minority", "one_unknown", "all_unknown"])
def test_disagreement_or_unknown_never_resolves_by_majority_or_ignoring_frames(frames, tmp_path, case):
    if case == "minority":
        path = tmp_path / "third.png"
        Image.new("RGB", (160, 100), "yellow").save(path)
        frames.append({"frame_id": "actual-third", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timestamp_s": 2.5, "rotation_clockwise_degrees": 0})
    sent = []
    def answer(messages, info):
        payload = type_decision(messages)
        if case in {"different_types", "minority"}:
            payload["readings"][-1]["document_type"] = "trn"
        elif case == "one_unknown":
            payload["readings"][-1].update(document_type="unknown", heading_text="")
        else:
            for reading in payload["readings"]:
                reading.update(document_type="unknown", heading_text="Other unsupported form")
        sent.append(deepcopy(payload))
        return response(payload)
    result = run(frames, tmp_path, answer)
    assert result["document_type"] == "unknown"
    assert result["fields"] == {} and result["adjudication"]["status"] == "unresolved"
    assert result["adjudication"]["resolved_document_type"] is None
    assert result["recognition"]["raw_decision"] == sent[0]
    assert len(result["classification_evidence"]) == len(frames)


@pytest.mark.parametrize("case", ["other_panel", "unsupplied_panel", "missing_panel", "blank_heading", "wrong_frame", "duplicate_frame", "omitted_frame"])
def test_type_resolution_requires_own_real_panel_heading_and_every_frame_once(frames, tmp_path, case):
    if case == "omitted_frame":
        path = tmp_path / "third.png"
        Image.new("RGB", (160, 100), "yellow").save(path)
        frames.append({"frame_id": "actual-third", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timestamp_s": 2.5, "rotation_clockwise_degrees": 0})
    sent = []
    def answer(messages, info):
        payload = type_decision(messages)
        first, second = payload["readings"][:2]
        if case == "other_panel": second["panel_ids"] = first["panel_ids"]
        elif case == "unsupplied_panel": second["panel_ids"] = ["not-actually-supplied"]
        elif case == "missing_panel": second["panel_ids"] = []
        elif case == "blank_heading": second["heading_text"] = "  "
        elif case == "wrong_frame": second["frame_id"] = "frame_99"
        elif case == "duplicate_frame": second["frame_id"] = first["frame_id"]
        else: payload["readings"].pop()
        sent.append(deepcopy(payload))
        return response(payload)
    result = run(frames, tmp_path, answer)
    assert result["document_type"] == "unknown" and result["adjudication"]["status"] == "unresolved"
    assert result["adjudication"]["errors"]
    assert result["recognition"]["raw_decision"] == sent[0]


@pytest.mark.parametrize("same", ["id", "sha", "pts", "triangle"])
def test_type_corroboration_requires_jointly_distinct_actual_id_sha_and_pts(frames, tmp_path, same):
    if same == "sha":
        Path(frames[1]["path"]).write_bytes(Path(frames[0]["path"]).read_bytes())
        frames[1]["source_sha256"] = frames[0]["source_sha256"]
    elif same == "pts": frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
    else:
        frames[1]["frame_id"] = frames[0]["frame_id"]
        if same == "triangle":
            frames.append({**frames[0], "frame_id": "third-id", "timestamp_s": frames[1]["timestamp_s"]})
    result = run(frames, tmp_path, lambda messages, _: response(type_decision(messages)))
    assert result["document_type"] == "unknown"
    assert any("distinct actual frame IDs" in error for error in result["adjudication"]["errors"])


def test_type_review_uses_actual_native_zoom_and_shared_provider_usage(frames, tmp_path):
    settings, sent, calls = config(max_requests=2), [], []
    def answer(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return zooms()
        assert info.function_tools == []
        payload = type_decision(messages)
        sent.append(deepcopy(payload))
        return response(payload)
    result = run(frames, tmp_path, answer, settings=settings)
    assert result["document_type"] == "ttn"
    assert settings.usage.requests == settings.usage.attempted_requests == 2
    assert (settings.usage.input_tokens, settings.usage.output_tokens, settings.usage.tool_calls) == (80, 30, 2)
    assert result["recognition"]["usage_delta"] == {"requests": 2, "input_tokens": 80, "output_tokens": 30, "tool_calls": 2}
    for reading in sent[0]["readings"]:
        assert "zoom_01" in reading["panel_ids"][0]
    for panel in result["panels"].values():
        assert hashlib.sha256(Path(panel["upload_path"]).read_bytes()).hexdigest() == panel["upload_sha256"]
        if panel["prep"]["format"] == "PNG":
            source = next(frame for frame in frames if frame["frame_id"] == panel["frame_id"])
            with Image.open(source["path"]) as original, Image.open(panel["upload_path"]) as actual:
                expected = original.crop(panel["upright_bbox_pixels_top_left"])
                assert actual.tobytes() == expected.tobytes()


def test_no_prior_types_answers_identifiers_or_historical_evidence_enter_model(frames, tmp_path, monkeypatch):
    historical = tmp_path / "evidence" / "image-conflicts-old.json"
    historical.parent.mkdir()
    historical.write_text('{"type":"OLD-CANDIDATE-TYPE","answer":"OLD-ANSWER"}')
    for frame in frames:
        frame.update(document_type="OLD-CANDIDATE-TYPE", raw_value="OLD-ANSWER")
    actual_paths = {Path(frame["path"]) for frame in frames}
    read_bytes, read_text = Path.read_bytes, Path.read_text
    def checked_bytes(path, *args, **kwargs):
        assert path in actual_paths, "Historical response/crop entered model input"
        return read_bytes(path, *args, **kwargs)
    def forbidden_text(*args, **kwargs):
        raise AssertionError("Historical response text read")
    def answer(messages, info):
        sent = "\n".join(item for item in content(messages) if isinstance(item, str))
        assert "Requested form=" not in sent
        for forbidden in ("OLD-CANDIDATE-TYPE", "OLD-ANSWER", "private-source-id", "private-page-id",
                          *(frame["frame_id"] for frame in frames), *(frame["path"] for frame in frames)):
            assert forbidden not in str(messages)
        assert sum(isinstance(item, BinaryContent) for item in content(messages)) == 2
        return response(type_decision(messages))
    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(Path, "read_text", forbidden_text)
    result = run(frames, tmp_path, answer)
    assert result["document_type"] == "ttn" and result["recognition"]["response_reuse"] is False
    assert "OLD-ANSWER" in read_text(historical)


@pytest.mark.parametrize("case", ["requests", "preflight", "received_tokens", "provider"])
def test_type_review_budget_provider_stop_is_shared_and_preserves_actual_response(frames, tmp_path, case, monkeypatch):
    settings = config(max_requests=1) if case == "requests" else config(max_tokens=1) if case == "preflight" else config(input_tokens_limit=49) if case == "received_tokens" else config()
    if case == "requests":
        settings.usage.requests = settings.usage.attempted_requests = 1
    calls = []
    def answer(messages, info):
        calls.append(1)
        if case == "provider":
            raise ModelHTTPError(401, "offline", body={"secret": "NEVER-EXPOSE"})
        return response(type_decision(messages))
    expected = FatalProviderError if case == "provider" else BatchBudgetExceeded
    with pytest.raises(expected) as caught:
        run(frames, tmp_path, answer, settings=settings)
    assert len(calls) == (0 if case in {"requests", "preflight"} else 1)
    assert "NEVER-EXPOSE" not in str(caught.value)
    actual = records(tmp_path)
    if calls:
        assert actual[0]["status"] == ("fatal" if case == "provider" else "budget_exhausted")
        assert len(actual[0]["raw_responses"]) == (0 if case == "provider" else 1)
    else:
        assert actual == []


def test_changed_type_source_after_response_cannot_publish_success(frames, tmp_path):
    settings = config()
    def answer(messages, info):
        Image.new("RGB", (160, 100), "yellow").save(frames[1]["path"])
        return response(type_decision(messages))
    with pytest.raises(AnalysisError):
        run(frames, tmp_path, answer, settings=settings)
    assert settings.usage.requests == 1
    record = records(tmp_path)[0]
    assert record["status"] == "failed" and len(record["raw_responses"]) == 1
    assert "document_type" not in record


@pytest.mark.parametrize("document_type,requested", [("unknown", ["consignor"]), ("upd", []), ("private", [])])
def test_only_explicit_unknown_empty_request_enables_type_mode(frames, tmp_path, document_type, requested):
    with pytest.raises(ValueError):
        run(frames, tmp_path, lambda *_: pytest.fail("Invalid mode invoked model"),
            document_type=document_type, requested=requested)
    assert records(tmp_path) == []
