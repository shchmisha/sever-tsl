import asyncio
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_conflicts import read_leaf_conflicts
from sever.image_detail import _rotate, _source_box


@pytest.fixture
def frames(tmp_path):
    result = []
    for index in range(2):
        path = tmp_path / f"private-source-{index}.png"
        image = Image.new("RGB", (160, 100))
        image.putdata([((x + index * 13) % 256, y % 256, (x + y) % 256)
                       for y in range(100) for x in range(160)])
        image.save(path)
        result.append({"frame_id": f"private-frame-{index}", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timestamp_s": index + .5, "rotation_clockwise_degrees": 0})
    return result


def config(**kwargs):
    return AgentConfig(model_name="offline-test", vision_model_name="offline-test", provider_kind="local",
        base_url="http://127.0.0.1:8090/v1", document_reader="native_image", **kwargs)


def content(messages):
    return [item for message in messages for part in message.parts
            if isinstance(getattr(part, "content", None), list) for item in part.content]


def panel_ids(messages):
    result = {}
    for item in content(messages):
        if isinstance(item, str) and item.startswith("Actual frame_id="):
            label = item.split("=", 1)[1].split(";", 1)[0]
            result[label] = item.split("panel_id=", 1)[1].split(";", 1)[0]
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and isinstance(part.content, dict) and "panel_id" in part.content:
                result[part.content["frame_id"]] = part.content["panel_id"]
    return result


def decision(messages, *, requested=("consignor",), raw="  ООО «Тест Ω»  ", kind="upd", statuses=None):
    panels = panel_ids(messages)
    return {"document_type": kind, "fields": {name: {"resolution_reason": "Свежие реальные строки на каждом кадре",
        "readings": [{"frame_id": label, "status": (statuses or {}).get(label, "recognized"),
            "raw_value": raw if (statuses or {}).get(label, "recognized") == "recognized" else None,
            "explicit_blank": (statuses or {}).get(label) == "not_present", "evidence_text": "",
            "complete_page_visible": True,
            "panel_ids": [identifier]} for label, identifier in panels.items()]}
        for name in requested}}


def response(payload, *, tool="submit_conflict_review", input_tokens=50, output_tokens=20):
    return ModelResponse(parts=[ToolCallPart(tool, payload, tool_call_id="submit")],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens))


def zooms(labels=("frame_01", "frame_02"), *, count=None):
    labels = list(labels) if count is None else [labels[index % len(labels)] for index in range(count)]
    return ModelResponse(parts=[ToolCallPart("zoom_source", {"frame_id": label,
        "left": .203, "top": .157, "right": .553, "bottom": .457}, tool_call_id=f"crop-{index}")
        for index, label in enumerate(labels)], usage=RequestUsage(input_tokens=30, output_tokens=10))


def run(frames, tmp_path, fn, *, settings=None, requested=("consignor",)):
    return asyncio.run(read_leaf_conflicts(frames, source_id="private-video-sha", page_id="private-leaf-id",
        document_type="upd", unresolved_fields=list(requested), config=settings or config(),
        cache_dir=tmp_path / "evidence", trace_path=tmp_path / "trace.jsonl",
        model=FunctionModel(fn, model_name="offline-function-model")))


def records(tmp_path):
    return [json.loads(path.read_text()) for path in (tmp_path / "evidence").glob("image-conflicts-*.json")]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_two_sources_actual_native_crops_all_rotations_and_raw_export(frames, tmp_path, rotation):
    for frame in frames:
        frame["rotation_clockwise_degrees"] = rotation
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(deepcopy(messages))
        assert [tool.name for tool in info.output_tools] == ["submit_conflict_review"]
        assert info.model_request_parameters.output_mode == "tool"
        assert info.model_request_parameters.output_object is None
        assert info.output_tools[0].strict is False
        if len(calls) == 1:
            return zooms()
        return response(decision(messages))
    result = run(frames, tmp_path, respond, settings=settings)
    assert settings.usage.requests == settings.usage.attempted_requests == len(calls) == 2
    assert settings.usage.tool_calls == 2
    assert (settings.usage.input_tokens, settings.usage.output_tokens) == (80, 30)
    assert result["adjudication"]["status"] == "resolved"
    assert result["fields"]["consignor"].raw_value == "  ООО «Тест Ω»  "
    assert result["fields"]["consignor"].status == "recognized"
    proof = result["recognition"]["pixel_proof"]
    assert set(proof["sources"]) == {"frame_01", "frame_02"}
    images = [item for item in content(calls[-1]) if isinstance(item, BinaryContent)]
    assert len(images) == len(proof["panels"]) == 4
    for image, panel in zip(images, proof["panels"].values()):
        # Aggregate dictionary is source-grouped, model history is context-first;
        # bind bytes by SHA rather than relying on that presentation order.
        actual = next(item for item in images if hashlib.sha256(item.data).hexdigest() == panel["upload_sha256"])
        assert actual.data == Path(panel["upload_path"]).read_bytes()
        if panel["prep"]["format"] == "PNG":
            source = next(frame for frame in frames if frame["frame_id"] == panel["frame_id"])
            with Image.open(source["path"]) as original:
                box = tuple(panel["upright_bbox_pixels_top_left"])
                expected = _rotate(original.crop(_source_box(box, original.size, rotation)), rotation)
            with Image.open(BytesIO(actual.data)) as crop:
                assert crop.size == expected.size and crop.tobytes() == expected.tobytes()
        assert panel["panel_id"].startswith(panel["model_frame_id"] + "_")
    record = records(tmp_path)[0]
    assert record["raw_decision"] == result["recognition"]["raw_decision"]
    assert record["raw_responses"] == result["recognition"]["raw_responses"]
    assert record["fields"]["consignor"] == result["fields"]["consignor"].model_dump(mode="json")
    assert len(record["raw_responses"]) == 2 and len(record["tool_proofs"]) == 2
    for messages in calls:
        sent = "\n".join(item for item in content(messages) if isinstance(item, str))
        for private in ["private-video-sha", "private-leaf-id", *(frame["path"] for frame in frames),
                        *(frame["frame_id"] for frame in frames), *(frame["source_sha256"] for frame in frames)]:
            assert private not in sent


def test_only_requested_fields_schema_no_prior_values_or_answer_replay(frames, tmp_path, monkeypatch):
    old = tmp_path / "evidence" / "image-conflicts-historical.json"
    old.parent.mkdir()
    old.write_text('{"raw_decision":"OLD-SECRET-VALUE"}')
    actual_paths = {Path(frame["path"]) for frame in frames}
    read_bytes, read_text = Path.read_bytes, Path.read_text
    def checked_bytes(path, *args, **kwargs):
        assert path in actual_paths, "Historical answer/crop was read as model input"
        return read_bytes(path, *args, **kwargs)
    def forbidden_text(*args, **kwargs):
        raise AssertionError("Historical answer text read")
    calls = []
    def respond(messages, info):
        calls.append(deepcopy(messages))
        assert "OLD-SECRET-VALUE" not in str(messages)
        fields = info.output_tools[0].parameters_json_schema["$defs"]["RequestedConflictFields"]["properties"]
        assert set(fields) == {"consignor_address"}
        return response(decision(messages, requested=("consignor_address",), raw=f"Fresh {len(calls)}"))
    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(Path, "read_text", forbidden_text)
    first = run(frames, tmp_path, respond, requested=("consignor_address",))
    second = run(frames, tmp_path, respond, requested=("consignor_address",))
    assert len(calls) == 2
    assert first["fields"]["consignor_address"].raw_value == "Fresh 1"
    assert second["fields"]["consignor_address"].raw_value == "Fresh 2"
    assert first["recognition"]["evidence_path"] != second["recognition"]["evidence_path"]
    assert "OLD-SECRET-VALUE" in read_text(old)


@pytest.mark.parametrize("case", ["different_values", "mixed_blank", "one_unreadable", "all_blank", "different_form"])
def test_truthful_disagreement_blank_unreadable_and_other_form(frames, tmp_path, case):
    def respond(messages, info):
        payload = decision(messages)
        readings = payload["fields"]["consignor"]["readings"]
        if case == "different_values":
            readings[1]["raw_value"] = "Different literal organization"
        elif case in {"mixed_blank", "all_blank"}:
            for reading in readings[1:] if case == "mixed_blank" else readings:
                reading.update(status="not_present", raw_value=None, explicit_blank=True)
        elif case == "one_unreadable":
            readings[1].update(status="unreadable", raw_value=None)
        else:
            payload["document_type"] = "ttn"
        return response(payload)
    result = run(frames, tmp_path, respond)
    expected = {"different_values": "conflict", "mixed_blank": "conflict", "one_unreadable": "unreadable",
                "all_blank": "not_present", "different_form": "unreadable"}[case]
    assert result["fields"]["consignor"].status == expected
    assert result["fields"]["consignor"].raw_value is None
    assert result["adjudication"]["status"] == ("resolved" if case == "all_blank" else "unresolved")
    assert records(tmp_path)[0]["raw_decision"] == result["recognition"]["raw_decision"]


@pytest.mark.parametrize("count", [3, 4])
def test_every_supplied_frame_is_required_and_no_majority_can_resolve(frames, tmp_path, count):
    for index in range(2, count):
        path = tmp_path / f"extra-{index}.png"
        Image.new("RGB", (160, 100), (index * 20, 30, 40)).save(path)
        frames.append({"frame_id": f"source-{index}", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timestamp_s": index + .5, "rotation_clockwise_degrees": 0})
    def disagree(messages, info):
        payload = decision(messages)
        payload["fields"]["consignor"]["readings"][-1]["raw_value"] = "Materially different"
        return response(payload)
    result = run(frames, tmp_path, disagree)
    assert result["fields"]["consignor"].status == "conflict"
    assert len(result["recognition"]["raw_decision"]["fields"]["consignor"]["readings"]) == count
    def omitted(messages, info):
        payload = decision(messages)
        payload["fields"]["consignor"]["readings"].pop()
        return response(payload)
    result = run(frames, tmp_path, omitted)
    assert result["fields"]["consignor"].status == "unreadable"
    assert "each supplied frame exactly once" in " ".join(result["recognition"]["errors"])


def test_visible_blank_does_not_resolve_with_incomplete_page_claim(frames, tmp_path):
    def respond(messages, info):
        payload = decision(messages, statuses={"frame_01": "not_present", "frame_02": "not_present"})
        payload["fields"]["consignor"]["readings"][1]["complete_page_visible"] = False
        return response(payload)
    result = run(frames, tmp_path, respond)
    assert result["fields"]["consignor"].status == "unreadable"
    assert result["adjudication"]["resolved_fields"] == []
    assert "complete page" in " ".join(result["recognition"]["errors"])


def test_partial_resolutions_are_explicit_requested_fields_only(frames, tmp_path):
    def respond(messages, info):
        payload = decision(messages, requested=("consignor", "consignee"))
        payload["fields"]["consignee"]["readings"][1]["raw_value"] = "Other recipient"
        return response(payload)
    result = run(frames, tmp_path, respond, requested=("consignor", "consignee"))
    assert set(result["fields"]) == {"consignor", "consignee"}
    assert result["adjudication"]["status"] == "unresolved"
    assert result["adjudication"]["resolved_fields"] == ["consignor"]
    assert result["adjudication"]["unresolved_fields"] == ["consignee"]


@pytest.mark.parametrize("case", ["other_panel", "forged_panel", "missing_panel", "wrong_frame", "duplicate_frame", "bad_raw", "bad_blank"])
def test_structural_provenance_failures_never_resolve_and_raw_is_retained(frames, tmp_path, case):
    sent = []
    def respond(messages, info):
        payload = decision(messages)
        first, second = payload["fields"]["consignor"]["readings"]
        if case == "other_panel": second["panel_ids"] = first["panel_ids"]
        elif case == "forged_panel": second["panel_ids"] = ["unsupplied-panel"]
        elif case == "missing_panel": second["panel_ids"] = []
        elif case == "wrong_frame": second["frame_id"] = "frame_99"
        elif case == "duplicate_frame": second["frame_id"] = first["frame_id"]
        elif case == "bad_raw": second["raw_value"] = None
        else: second["explicit_blank"] = True
        sent.append(deepcopy(payload))
        return response(payload)
    result = run(frames, tmp_path, respond)
    assert result["adjudication"]["status"] == "unresolved"
    assert result["fields"]["consignor"].status == "unreadable"
    assert result["recognition"]["errors"]
    assert result["recognition"]["raw_decision"] == sent[0]


@pytest.mark.parametrize("same", ["sha", "pts", "frame_id"])
def test_source_identity_independence_not_just_two_model_labels(frames, tmp_path, same):
    if same == "sha":
        Path(frames[1]["path"]).write_bytes(Path(frames[0]["path"]).read_bytes())
        frames[1]["source_sha256"] = frames[0]["source_sha256"]
    elif same == "pts": frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
    else: frames[1]["frame_id"] = frames[0]["frame_id"]
    result = run(frames, tmp_path, lambda messages, info: response(decision(messages)))
    assert result["fields"]["consignor"].status == "unreadable"
    assert result["adjudication"]["resolved_fields"] == []
    assert "distinct actual frame IDs" in " ".join(result["recognition"]["errors"])


def test_independence_requires_one_pair_distinct_in_all_three_identifiers(frames, tmp_path):
    # Separate set cardinalities alone would falsely accept this triangle:
    # (IDa,SHAa,PTSa), (IDa,SHAb,PTSb), (IDb,SHAa,PTSb).
    frames[1]["frame_id"] = frames[0]["frame_id"]
    third = {**frames[0], "frame_id": "third-actual-id", "timestamp_s": frames[1]["timestamp_s"]}
    frames.append(third)
    result = run(frames, tmp_path, lambda messages, info: response(decision(messages)))
    independence = result["adjudication"]["field_reviews"]["consignor"]["independence"]
    assert all(len(values) == 2 for values in independence.values())
    assert result["fields"]["consignor"].status == "unreadable"


@pytest.mark.parametrize("stage", ["expected_sha", "after_first", "after_final"])
def test_changed_source_never_resolves_and_received_usage_is_retained(frames, tmp_path, stage):
    calls, settings = [], config()
    if stage == "expected_sha": frames[1]["source_sha256"] = "0" * 64
    def respond(messages, info):
        calls.append(1)
        Image.new("RGB", (160, 100), "yellow").save(frames[1]["path"])
        return zooms() if stage == "after_first" else response(decision(messages))
    with pytest.raises(AnalysisError):
        run(frames, tmp_path, respond, settings=settings)
    assert len(calls) == (0 if stage == "expected_sha" else 1)
    assert settings.usage.requests == len(calls)
    if calls:
        assert records(tmp_path)[0]["status"] == "failed"
        assert len(records(tmp_path)[0]["raw_responses"]) == 1


@pytest.mark.parametrize("bad", ["too_many_frames", "missing_pts", "bool_pts", "duplicate_field", "wrong_field"])
def test_invalid_review_request_rejected_before_model(frames, tmp_path, bad):
    requested = ("consignor",)
    if bad == "too_many_frames": frames = frames * 3
    elif bad == "missing_pts": frames[0].pop("timestamp_s")
    elif bad == "bool_pts": frames[0]["timestamp_s"] = True
    elif bad == "duplicate_field": requested = ("consignor", "consignor")
    else: requested = ("unrequested_private_field",)
    with pytest.raises(ValueError):
        run(frames, tmp_path, lambda *a: pytest.fail("Model called"), requested=requested)
    assert records(tmp_path) == []


def test_parallel_excess_crops_declined_truthfully_then_only_final_tool(frames, tmp_path):
    calls = []
    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return zooms(count=14)
        assert info.function_tools == []
        images = [item for item in content(messages) if isinstance(item, BinaryContent)]
        assert len(images) == 14  # Two contexts plus exactly twelve crops.
        return response(decision(messages))
    result = run(frames, tmp_path, respond)
    record = records(tmp_path)[0]
    assert len(calls) == 2 and result["recognition"]["zoom_images"] == 12
    assert len(record["tool_proofs"]) == 12 and len(record["tool_declines"]) == 2
    assert all(row["no_image_returned"] for row in record["tool_declines"])


@pytest.mark.parametrize("case", ["preflight", "shared_requests", "response_tokens", "own_request_limit"])
def test_shared_and_own_budgets_keep_real_responses_without_retry(frames, tmp_path, case):
    settings = config(**{"preflight": {"max_tokens": 1}, "shared_requests": {"max_requests": 1},
                         "response_tokens": {"input_tokens_limit": 29}, "own_request_limit": {}}[case])
    calls = []
    def respond(messages, info):
        calls.append(1)
        # Ignoring tool availability cannot create an unbounded extra request.
        return zooms(labels=("frame_01",))
    # Returning a tool that the final-request prepare hook has hidden is a
    # model schema/tool fault, not a fictitious extra budgeted request.
    error = AnalysisError if case in {"shared_requests", "own_request_limit"} else BatchBudgetExceeded
    with pytest.raises(error):
        run(frames, tmp_path, respond, settings=settings)
    expected = {"preflight": 0, "shared_requests": 1, "response_tokens": 1, "own_request_limit": 12}[case]
    assert len(calls) == settings.usage.requests == expected
    assert settings.usage.input_tokens == 30 * expected and settings.usage.output_tokens == 10 * expected
    if expected:
        assert records(tmp_path)[0]["status"] == ("failed" if case in {"shared_requests", "own_request_limit"} else "budget_exhausted")
        assert len(records(tmp_path)[0]["raw_responses"]) == expected
    else:
        assert records(tmp_path) == []


@pytest.mark.parametrize("request_limit", [1, 12])
def test_final_request_reserved_and_actual_shared_limits(frames, tmp_path, request_limit):
    calls, settings = [], config(max_requests=request_limit)
    def respond(messages, info):
        calls.append(1)
        if len(calls) < request_limit:
            assert [tool.name for tool in info.function_tools] == ["zoom_source", "zoom_regions"]
            return zooms(labels=("frame_01",))
        assert info.function_tools == []
        return response(decision(messages))
    result = run(frames, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == request_limit
    assert result["adjudication"]["status"] == "resolved"
    assert result["recognition"]["zoom_images"] == request_limit - 1


def test_already_exhausted_shared_budget_makes_no_call(frames, tmp_path):
    settings = config(max_requests=1)
    settings.usage.requests = settings.usage.attempted_requests = 1
    with pytest.raises(BatchBudgetExceeded):
        run(frames, tmp_path, lambda *a: pytest.fail("Exceeded shared budget"), settings=settings)
    assert settings.usage.requests == settings.usage.attempted_requests == 1
    assert records(tmp_path) == []


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_fatal_provider_attempts_charged_once_and_bodies_hidden(frames, tmp_path, status):
    calls, settings = [], config()
    def fail(messages, info):
        calls.append(1)
        raise ModelHTTPError(status, "offline", body={"secret": "DO-NOT-EXPOSE"})
    with pytest.raises(FatalProviderError) as caught:
        run(frames, tmp_path, fail, settings=settings)
    assert len(calls) == settings.usage.requests == 1
    assert "DO-NOT-EXPOSE" not in str(caught.value)
    assert records(tmp_path)[0]["status"] == "fatal"
    assert "DO-NOT-EXPOSE" not in json.dumps(records(tmp_path))


def test_partial_provider_failure_preserves_received_crop_and_usage(frames, tmp_path):
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1: return zooms(labels=("frame_01",))
        raise RuntimeError("Sensitive provider body")
    with pytest.raises(AnalysisError):
        run(frames, tmp_path, respond, settings=settings)
    record = records(tmp_path)[0]
    assert len(calls) == settings.usage.requests == 2
    assert settings.usage.input_tokens == 30 and settings.usage.output_tokens == 10
    assert record["status"] == "failed"
    assert len(record["raw_responses"]) == len(record["tool_proofs"]) == 1
    assert "Sensitive provider body" not in json.dumps(record)


@pytest.mark.parametrize("failure", ["ordinary", "fatal", "budget"])
def test_proof_failure_does_not_mask_fatal_or_received_budget(frames, tmp_path, monkeypatch, failure):
    def storage(*a, **kw): raise OSError("disk full")
    monkeypatch.setattr("sever.image_conflicts._write_evidence", storage)
    settings = config(input_tokens_limit=29) if failure == "budget" else config()
    def respond(messages, info):
        if failure == "fatal": raise FatalProviderError("safe fatal")
        return response(decision(messages), input_tokens=30)
    expected = {"ordinary": AnalysisError, "fatal": FatalProviderError, "budget": BatchBudgetExceeded}[failure]
    with pytest.raises(expected):
        run(frames, tmp_path, respond, settings=settings)
    assert settings.usage.requests == 1


@pytest.mark.parametrize("failure", ["success", "fatal", "budget"])
def test_owned_client_cleanup_preserves_primary_error(frames, tmp_path, monkeypatch, failure):
    from sever import image_conflicts
    closed = []
    class Client:
        def __init__(self, **kw): assert kw["max_retries"] == 0
        async def close(self):
            closed.append(True)
            raise OSError("close failed")
    monkeypatch.setattr(image_conflicts, "AsyncOpenAI", Client)
    monkeypatch.setattr(image_conflicts, "OpenAIProvider", lambda **kw: object())
    def respond(messages, info):
        if failure == "fatal": raise FatalProviderError("safe fatal")
        return response(decision(messages), input_tokens=30)
    monkeypatch.setattr(image_conflicts, "_RouterAIChatModel", lambda *a, **kw: FunctionModel(respond))
    settings = config(input_tokens_limit=29) if failure == "budget" else config()
    expected = {"success": AnalysisError, "fatal": FatalProviderError, "budget": BatchBudgetExceeded}[failure]
    with pytest.raises(expected):
        asyncio.run(read_leaf_conflicts(frames, source_id="s", page_id="p", document_type="upd",
            unresolved_fields=["consignor"], config=settings, cache_dir=tmp_path / "evidence",
            trace_path=tmp_path / "trace.jsonl"))
    assert closed == [True]


@pytest.mark.parametrize("case", ["unreadable_first", "unreadable_last", "blank_last", "different_last", "duplicate_support"])
def test_two_readable_views_with_an_extra_view_require_actual_support_without_majority(frames, tmp_path, case):
    path = tmp_path / "third-source.png"
    Image.new("RGB", (160, 100), "yellow").save(path)
    frames.append({"frame_id": "third", "path": str(path), "timestamp_s": 2.5,
                   "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rotation_clockwise_degrees": 0})
    if case == "duplicate_support":
        frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
    def respond(messages, info):
        payload = decision(messages)
        readings = payload["fields"]["consignor"]["readings"]
        if case == "different_last":
            readings[-1]["raw_value"] = "A different actually read organization"
        elif case == "blank_last":
            readings[-1].update(status="not_present", raw_value=None, explicit_blank=True)
        else:
            readings[0 if case == "unreadable_first" else -1].update(status="unreadable", raw_value=None)
        return response(payload)
    result = run(frames, tmp_path, respond)
    expected = "conflict" if case in {"blank_last", "different_last"} else "unreadable" if case == "duplicate_support" else "recognized"
    field = result["fields"]["consignor"]
    assert field.status == expected
    if expected == "recognized":
        assert field.raw_value == "  ООО «Тест Ω»  "
        assert len(result["adjudication"]["field_reviews"]["consignor"]["independence"]["supporting_frame_ids"]) == 2
    assert len(result["recognition"]["raw_decision"]["fields"]["consignor"]["readings"]) == 3


def test_joint_review_lost_proof_has_a_persistent_error_type(frames, tmp_path, monkeypatch):
    from sever import image_conflicts
    from sever.image_agent import EvidencePersistenceError
    def unavailable(*args, **kwargs):
        raise OSError("private proof storage failure")
    monkeypatch.setattr(image_conflicts, "_write_evidence", unavailable)
    settings = config()
    with pytest.raises(EvidencePersistenceError, match="Cannot preserve") as caught:
        run(frames, tmp_path, lambda messages, _: response(decision(messages)), settings=settings)
    assert "private" not in str(caught.value)
    assert settings.usage.requests == 1


def test_batched_two_source_crops_preserve_native_pixels_and_use_one_tool_roundtrip(frames, tmp_path):
    import math
    frames[1]["rotation_clockwise_degrees"] = 90
    regions = [{"frame_id": "frame_01", "left": .13, "top": .21, "right": .63, "bottom": .79},
               {"frame_id": "frame_02", "left": .17, "top": .31, "right": .67, "bottom": .89}]
    settings, calls = config(max_requests=2), []
    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            assert [tool.name for tool in info.function_tools] == ["zoom_source", "zoom_regions"]
            return ModelResponse(parts=[ToolCallPart("zoom_regions", {"regions": regions}, tool_call_id="batch")],
                usage=RequestUsage(input_tokens=30, output_tokens=10))
        assert info.function_tools == []
        returned = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)][-1]
        panels = returned["panels"]
        assert len(panels) == 2 and {panel["frame_id"] for panel in panels} == {"frame_01", "frame_02"}
        images = [item for item in content(messages) if isinstance(item, BinaryContent)]
        assert len(images) == 4
        for actual, frame, region in zip(images[2:], frames, regions, strict=True):
            with Image.open(frame["path"]) as original:
                upright = _rotate(original, frame["rotation_clockwise_degrees"])
                box = (math.floor(region["left"] * upright.width), math.floor(region["top"] * upright.height),
                       math.ceil(region["right"] * upright.width), math.ceil(region["bottom"] * upright.height))
                assert Image.open(BytesIO(actual.data)).tobytes() == upright.crop(box).tobytes()
        payload = decision(messages)
        by_frame = {panel["frame_id"]: panel["panel_id"] for panel in panels}
        for reading in payload["fields"]["consignor"]["readings"]:
            reading["panel_ids"] = [by_frame[reading["frame_id"]]]
        return response(payload)
    result = run(frames, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == settings.usage.attempted_requests == 2
    assert settings.usage.tool_calls == 1 and result["recognition"]["zoom_images"] == 2
    assert result["fields"]["consignor"].status == "recognized"
    record = records(tmp_path)[0]
    assert len(record["tool_proofs"]) == 2 and record["tool_declines"] == []
    assert len(record["proof"]["panels"]) == 4
    assert {proof["frame_id"] for proof in record["tool_proofs"]} == {frame["frame_id"] for frame in frames}
    for proof in record["tool_proofs"]:
        assert hashlib.sha256(Path(proof["upload_path"]).read_bytes()).hexdigest() == proof["upload_sha256"]


@pytest.mark.parametrize("bad", ["frame_id", "geometry"])
def test_batched_review_validates_every_region_before_any_crop_is_written(frames, tmp_path, bad):
    regions = [{"frame_id": "frame_01", "left": .1, "top": .2, "right": .5, "bottom": .8},
               {"frame_id": "frame_02", "left": .1, "top": .2, "right": .5, "bottom": .8}]
    if bad == "frame_id":
        regions[1]["frame_id"] = "not-supplied"
    else:
        regions[1]["right"] = regions[1]["left"]
    calls = []
    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("zoom_regions", {"regions": regions}, tool_call_id="bad-batch")],
                usage=RequestUsage(input_tokens=30, output_tokens=10))
        return response(decision(messages))
    result = run(frames, tmp_path, respond)
    assert len(calls) == 2 and result["recognition"]["zoom_images"] == 0
    assert len(records(tmp_path)[0]["proof"]["panels"]) == 2
    assert records(tmp_path)[0]["tool_proofs"] == []


def test_batched_review_overflow_returns_explicit_nonimage_declines(frames, tmp_path):
    calls = []
    regions = [{"frame_id": f"frame_{index % 2 + 1:02d}", "left": .1, "top": .2, "right": .5, "bottom": .8}
               for index in range(12)]
    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            parts = zooms().parts + [ToolCallPart("zoom_regions", {"regions": regions}, tool_call_id="batch")]
            return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=30, output_tokens=10))
        assert info.function_tools == []
        returns = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        panels = returns[-1]["panels"]
        assert len(panels) == 12
        assert all(panel["no_image_returned"] and "panel_id" not in panel for panel in panels[-2:])
        assert len([item for item in content(messages) if isinstance(item, BinaryContent)]) == 14
        return response(decision(messages))
    result = run(frames, tmp_path, respond)
    assert len(calls) == 2 and result["recognition"]["zoom_images"] == 12
    record = records(tmp_path)[0]
    assert len(record["tool_proofs"]) == 12 and len(record["tool_declines"]) == 2


def test_joint_instructions_forbid_reference_markers_as_values():
    from sever.image_conflicts import IMAGE_CONFLICTS_VERSION, INSTRUCTIONS, TYPE_INSTRUCTIONS
    assert "raw_value НИКОГДА не содержит слов «он же»" in INSTRUCTIONS
    assert "без префикса «он же;»" in INSTRUCTIONS
    assert "ttn-continuation" in IMAGE_CONFLICTS_VERSION and "имеет продолжение" in TYPE_INSTRUCTIONS
