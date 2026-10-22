import asyncio
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, BatchBudgetExceeded, FatalProviderError
from sever.field_verification import (DETAIL_EDGE, DETAIL_OVERLAP, _padded_regions,
    _tile_boxes, verify_field_views)
from sever.image_agent import CropRegion, _Source
from sever.image_detail import _rotate, _source_box


@pytest.fixture
def frames(tmp_path):
    result = []
    for index in range(2):
        path = tmp_path / f"private-source-{index}.png"
        image = Image.new("RGB", (2048, 1280))
        # Spatially varying real pixels detect wrong boxes and invented scaling.
        image.putdata([((x + index * 13) % 256, y % 256, (x + y) % 256)
            for y in range(1280) for x in range(2048)])
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


def texts(messages):
    return [item for item in content(messages) if isinstance(item, str)]


def own_label(messages):
    return next(item.split("=", 1)[1].split(";", 1)[0] for item in texts(messages)
        if item.startswith("Actual frame_id="))


def detail_ids(messages):
    return [item.split("panel_id=", 1)[1].split(";", 1)[0] for item in texts(messages)
        if "native complete-field tile;" in item]


def response(tool, payload):
    return ModelResponse(parts=[ToolCallPart(tool, payload, tool_call_id="submit")],
        usage=RequestUsage(input_tokens=50, output_tokens=20))


def location(regions=None, status="located"):
    return {"status": status, "regions": regions if regions is not None else
        [{"left": .1, "top": .2, "right": .95, "bottom": .4}]}


def transcription(messages, raw="ООО «Тест», ИНН 1234567890, Москва, дом 15/16, стр. 8", status="recognized", **extra):
    return {"document_type": "upd", "status": status, "raw_value": raw if status == "recognized" else None,
        "explicit_blank": status == "not_present", "complete_page_visible": True,
        "evidence_text": "Грузополучатель и его адрес", "panel_ids": detail_ids(messages),
        "visibility_cause": "explicit_blank" if status == "not_present" else "readable" if status == "recognized" else "blur_or_small_text",
        "resolution_reason": "Actual own caption and full literal block on this frame", **extra}


def run(frames, tmp_path, respond, *, settings=None):
    return asyncio.run(verify_field_views(frames, source_id="private-video-sha", page_id="private-leaf-id",
        document_type="upd", field_name="consignee_address", config=settings or config(),
        cache_dir=tmp_path / "evidence", trace_path=tmp_path / "trace.jsonl",
        model=FunctionModel(respond, model_name="offline-function-model")))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_isolated_native_tiles_are_exact_complete_and_never_upscaled(frames, tmp_path, rotation):
    for frame in frames:
        frame["rotation_clockwise_degrees"] = rotation
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(deepcopy(messages))
        tool = info.output_tools[0].name
        assert info.model_request_parameters.output_mode == "tool"
        if tool == "submit_field_location":
            return response(tool, location())
        assert tool == "submit_field_transcription"
        assert len(messages) == 1, "Transcription must start with a fresh history"
        assert len(detail_ids(messages)) >= 2
        return response(tool, transcription(messages))
    result = run(frames, tmp_path, respond, settings=settings)
    assert result["fields"]["consignee_address"].status == "recognized"
    assert settings.usage.requests == settings.usage.attempted_requests == 4
    assert (settings.usage.input_tokens, settings.usage.output_tokens) == (200, 80)
    proof = result["recognition"]["pixel_proof"]
    images = [item for messages in calls for item in content(messages) if isinstance(item, BinaryContent)]
    for panel in proof["panels"].values():
        assert panel["panel_id"].startswith(panel["model_frame_id"])
        if panel["verification_purpose"] != "transcription_detail":
            continue
        assert max(panel["upload_pixel_size"]) <= DETAIL_EDGE
        assert panel["prep"]["resize"] == "none" and panel["prep"]["native_source_detail"]
        actual = next(item for item in images if hashlib.sha256(item.data).hexdigest() == panel["upload_sha256"])
        assert Path(panel["upload_path"]).read_bytes() == actual.data
        frame = next(frame for frame in frames if frame["frame_id"] == panel["frame_id"])
        with Image.open(frame["path"]) as original:
            expected = _rotate(original.crop(_source_box(tuple(panel["upright_bbox_pixels_top_left"]),
                original.size, rotation)), rotation)
        with Image.open(BytesIO(actual.data)) as crop:
            assert crop.size == expected.size and crop.tobytes() == expected.tobytes()
    record = json.loads(Path(result["recognition"]["evidence_path"]).read_text())
    assert record["raw_decision"] == result["recognition"]["raw_decision"]
    assert record["raw_responses"] == result["recognition"]["raw_responses"]
    assert record["fields"]["consignee_address"] == result["fields"]["consignee_address"].model_dump(mode="json")
    assert all(panel["source_sha256"] in {frame["source_sha256"] for frame in frames}
        for panel in proof["panels"].values())
    for messages in calls:
        sent = "\n".join(texts(messages))
        for private in ["private-video-sha", "private-leaf-id", *(frame["path"] for frame in frames),
                        *(frame["frame_id"] for frame in frames), *(frame["source_sha256"] for frame in frames)]:
            assert private not in sent


def test_full_selected_rectangle_and_ending_have_overlapping_pixel_coverage(frames, tmp_path):
    source = _Source(frames[0], tmp_path, 0)
    selected = CropRegion(left=.1, top=.2, right=.95, bottom=.9)
    boxes = _tile_boxes(source, [selected])
    assert all(right-left <= DETAIL_EDGE and bottom-top <= DETAIL_EDGE for left, top, right, bottom in boxes)
    width, height = source.upright_size
    x0, y0 = int(selected.left*width), int(selected.top*height)
    x1, y1 = int(-(-selected.right*width//1)), int(-(-selected.bottom*height//1))
    mask = Image.new("1", source.upright_size)
    for left, top, right, bottom in boxes:
        mask.paste(1, (left, top, right, bottom))
    assert mask.crop((x0, y0, x1, y1)).getextrema() == (1, 1)
    assert min(box[0] for box in boxes) == x0 and max(box[2] for box in boxes) == x1


def test_localization_zoom_then_fresh_transcription_has_no_previous_outputs(frames, tmp_path):
    calls, localization_calls = [], {}
    secret = "PREVIOUS_PRIVATE_READING_DO_NOT_REPLAY"
    def respond(messages, info):
        calls.append(deepcopy(messages))
        tool, label = info.output_tools[0].name, own_label(messages)
        if tool == "submit_field_location":
            assert secret not in str(messages)
            localization_calls[label] = localization_calls.get(label, 0) + 1
            if localization_calls[label] == 1:
                return response("zoom_regions", {"regions": location()["regions"]})
            assert "zoom_regions" not in [tool.name for tool in info.function_tools]
            return response(tool, location())
        assert len(messages) == 1
        assert secret not in str(messages)
        assert "submit_field_location" not in str(messages)
        assert "native localization crop" not in str(messages)
        assert "zoom_regions" not in str(messages)
        return response(tool, transcription(messages, raw=secret))
    settings = config()
    result = run(frames, tmp_path, respond, settings=settings)
    assert result["fields"]["consignee_address"].raw_value == secret
    assert len(calls) == settings.usage.requests == settings.usage.attempted_requests == 6
    assert settings.usage.tool_calls == 2


@pytest.mark.parametrize("suffix", ["В", "8", "5", "6", "B"])
def test_true_literal_characters_and_entire_address_are_preserved(frames, tmp_path, suffix):
    raw = f'ООО «Склад», ИНН/КПП 1234567890/123401001, РЦ Запад, Москва, дом 15/16, стр. {suffix}'
    def respond(messages, info):
        tool = info.output_tools[0].name
        return response(tool, location() if tool == "submit_field_location" else transcription(messages, raw=raw))
    result = run(frames, tmp_path, respond)
    assert result["fields"]["consignee_address"].raw_value == raw


@pytest.mark.parametrize("case,expected", [("letter_digit", "conflict"), ("one_unreadable", "unreadable"),
    ("all_blank", "not_present"), ("missing_block", "unreadable"), ("mixed_blank", "conflict"),
    ("blank_incomplete", "unreadable"), ("ignored_tile", "unreadable"), ("wrong_form", "unreadable"),
    ("no_caption", "unreadable"), ("blurred_recognized", "unreadable")])
def test_disagreement_blanks_unreadability_and_provenance(frames, tmp_path, case, expected):
    def respond(messages, info):
        tool, label = info.output_tools[0].name, own_label(messages)
        if tool == "submit_field_location":
            return response(tool, location([], status="unreadable") if case == "missing_block" else location())
        payload = transcription(messages)
        if case == "letter_digit" and label == "frame_02":
            payload["raw_value"] = payload["raw_value"][:-1] + "В"
        if case == "one_unreadable" and label == "frame_02":
            payload = transcription(messages, status="unreadable")
        if case in {"all_blank", "blank_incomplete"} or (case == "mixed_blank" and label == "frame_02"):
            payload = transcription(messages, status="not_present")
            if case == "blank_incomplete":
                payload["complete_page_visible"] = False
        if case == "ignored_tile":
            payload["panel_ids"].pop()
        if case == "wrong_form":
            payload["document_type"] = "ttn"
        if case == "no_caption":
            payload["evidence_text"] = ""
        if case == "blurred_recognized":
            payload["visibility_cause"] = "blur_or_small_text"
        return response(tool, payload)
    result = run(frames, tmp_path, respond)
    field = result["fields"]["consignee_address"]
    assert field.status == expected
    assert field.raw_value is None
    assert result["adjudication"]["status"] == ("resolved" if case == "all_blank" else "unresolved")
    if case == "missing_block":
        assert len(result["recognition"]["view_errors"]) == 2
        assert result["recognition"]["model_requests"] == 2


def test_duplicate_source_cannot_independently_confirm(frames, tmp_path):
    frames[1].update(path=frames[0]["path"], source_sha256=frames[0]["source_sha256"])
    def respond(messages, info):
        tool = info.output_tools[0].name
        return response(tool, location() if tool == "submit_field_location" else transcription(messages))
    result = run(frames, tmp_path, respond)
    assert result["fields"]["consignee_address"].status == "unreadable"
    assert "distinct actual frame" in " ".join(result["recognition"]["errors"])


def test_history_directory_is_write_only_and_responses_never_reused(frames, tmp_path, monkeypatch):
    old = tmp_path / "evidence" / "field-verification-old.json"
    old.parent.mkdir()
    old.write_text('{"raw_value":"OLD_SECRET"}')
    original_read = Path.read_bytes
    allowed = {Path(frame["path"]) for frame in frames}
    def source_only(path, *args, **kwargs):
        assert path in allowed, "Historical crops/evidence must never be loaded"
        return original_read(path, *args, **kwargs)
    def no_text(*args, **kwargs):
        raise AssertionError("Historical evidence text read")
    monkeypatch.setattr(Path, "read_bytes", source_only)
    monkeypatch.setattr(Path, "read_text", no_text)
    calls = []
    def respond(messages, info):
        calls.append(info.output_tools[0].name)
        assert "OLD_SECRET" not in str(messages)
        tool = info.output_tools[0].name
        return response(tool, location() if tool == "submit_field_location" else transcription(messages))
    first, second = run(frames, tmp_path, respond), run(frames, tmp_path, respond)
    assert len(calls) == 8
    assert first["recognition"]["evidence_path"] != second["recognition"]["evidence_path"]


@pytest.mark.parametrize("failure", ["request_budget", "token_budget", "provider"])
def test_shared_budget_and_fatal_provider_failures_propagate(frames, tmp_path, failure):
    calls = []
    def respond(messages, info):
        calls.append(info.output_tools[0].name)
        if failure == "provider":
            raise ModelHTTPError(status_code=401, model_name="offline-test", body={"error": "denied"})
        tool = info.output_tools[0].name
        return response(tool, location() if tool == "submit_field_location" else transcription(messages))
    settings = config(max_requests=1) if failure == "request_budget" else config(input_tokens_limit=1) if failure == "token_budget" else config()
    with pytest.raises(FatalProviderError if failure == "provider" else BatchBudgetExceeded):
        run(frames, tmp_path, respond, settings=settings)
    assert len(calls) == 1
    assert settings.usage.requests == settings.usage.attempted_requests == 1


def test_oversized_block_is_never_partially_transcribed(tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"large-{index}.png"
        Image.new("RGB", (3200, 4000), (index, 0, 0)).save(path)
        frames.append({"path": str(path), "frame_id": f"real-{index}", "timestamp_s": index+.5,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rotation_clockwise_degrees": 0})
    def respond(messages, info):
        assert info.output_tools[0].name == "submit_field_location", "Partial native detail must not be transcribed"
        return response("submit_field_location", location([{"left": 0, "top": 0, "right": 1, "bottom": 1}]))
    result = run(frames, tmp_path, respond)
    assert result["fields"]["consignee_address"].status == "unreadable"
    assert result["recognition"]["model_requests"] == 2
    assert all("no partial transcription" in error for error in result["recognition"]["view_errors"].values())
    assert len(result["panels"]) == 2


@pytest.mark.parametrize("third_status,expected", [("recognized", "conflict"), ("unreadable", "recognized")])
def test_third_view_disagreement_is_kept_and_unreadable_is_not_a_vote(frames, tmp_path, third_status, expected):
    path = tmp_path / "third-source.png"
    Image.new("RGB", (2048, 1280), (80, 90, 100)).save(path)
    frames.append({"path": str(path), "frame_id": "actual-third-frame", "timestamp_s": 2.5,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rotation_clockwise_degrees": 0})
    def respond(messages, info):
        tool = info.output_tools[0].name
        if tool == "submit_field_location":
            return response(tool, location())
        payload = transcription(messages)
        if own_label(messages) == "frame_03":
            payload = transcription(messages, raw="Different literal suffix", status=third_status)
        return response(tool, payload)
    result = run(frames, tmp_path, respond)
    assert result["fields"]["consignee_address"].status == expected
    assert result["recognition"]["model_requests"] == 6
    review = result["adjudication"]["field_reviews"]["consignee_address"]
    assert len(review["model_review"]["readings"]) == 3
    assert len(review["independence"]["supporting_frame_ids"]) == (3 if third_status == "recognized" else 2)


def test_padding_recovers_nearby_own_row_with_exact_native_pixels(tmp_path):
    path = tmp_path / "row-offset.png"
    image = Image.new("RGB", (2160, 3840), "white")
    ImageDraw.Draw(image).rectangle((300, 790, 1700, 820), fill=(15, 30, 45))
    image.save(path)
    source = _Source({"path": str(path), "frame_id": "real-row-frame", "timestamp_s": .5}, tmp_path, 0)
    region = CropRegion(left=.1, top=.222, right=.95, bottom=.238)
    padded, padding = _padded_regions(source, [region])
    assert padding["horizontal_pixels"] == 48 and padding["vertical_pixels"] == 96
    details = padding["regions"][0]
    raw = details["raw_localized_upright_bbox_pixels_xyxy"]
    actual = details["padded_upright_bbox_pixels_xyxy"]
    assert raw[1] > 820, "Unpadded model rectangle misses the adjacent own row"
    assert actual[1] <= 790 < 820 <= actual[3]
    assert actual[0] <= raw[0] - 48 and actual[2] >= raw[2] + 48
    assert details["raw_localized_upright_bbox_xyxy"] == [.1, .222, .95, .238]
    assert details["padded_upright_bbox_xyxy"] == list(padded[0].model_dump().values())
    covered = Image.new("1", image.size)
    for box in _tile_boxes(source, padded):
        identifier, data, proof = source.panel(box, context=False)
        assert proof["upload_pixel_size"] == [box[2]-box[0], box[3]-box[1]]
        assert max(proof["upload_pixel_size"]) <= DETAIL_EDGE
        assert proof["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert proof["prep"]["resize"] == "none"
        assert proof["source_bbox_pixels_top_left"] == list(box)
        with Image.open(BytesIO(data)) as actual_crop:
            assert actual_crop.tobytes() == image.crop(box).tobytes()
        covered.paste(1, box)
    assert covered.crop((300, 790, 1701, 821)).getextrema() == (1, 1)


@pytest.mark.parametrize("size,expected", [((1080, 1920), (48, 96)),
    ((2160, 3840), (48, 96)), ((3840, 2160), (48, 96)), ((7680, 4320), (64, 128))])
def test_padding_clips_to_real_source_and_stays_bounded(size, expected):
    padded, proof = _padded_regions(SimpleNamespace(upright_size=size),
        [CropRegion(left=.001, top=.001, right=.999, bottom=.999)])
    assert (proof["horizontal_pixels"], proof["vertical_pixels"]) == expected
    assert padded[0].model_dump() == {"left": 0, "top": 0, "right": 1, "bottom": 1}
    assert proof["regions"][0]["padded_upright_bbox_pixels_xyxy"] == [0, 0, *size]
    assert proof["resize"] == "none"


def test_balanced_overlap_contains_word_crossing_old_boundary_without_near_duplicate_end_tiles():
    source = SimpleNamespace(upright_size=(2160, 3840))
    region = CropRegion(left=189/2160, top=741/3840, right=1993/2160, bottom=999/3840)
    boxes = _tile_boxes(source, [region])
    assert DETAIL_OVERLAP == 256
    assert len(boxes) == 5
    starts = [box[0] for box in boxes]
    assert starts[0] == 189 and boxes[-1][2] == 1993
    # A complete200px word crossing the old first tile's right boundary829px
    # remains intact in a supplied crop. No text is reconstructed or upscaled.
    word_left, word_right = 760, 960
    assert word_left < 829 < word_right
    assert any(left <= word_left and right >= word_right for left, _, right, _ in boxes)
    strides = [right-left for left, right in zip(starts, starts[1:])]
    assert max(strides) - min(strides) <= 1
    assert min(strides) > 12, "Do not append an almost identical final crop"
    assert all(first[2] - second[0] >= DETAIL_OVERLAP for first, second in zip(boxes, boxes[1:]))
    assert all(right-left == DETAIL_EDGE and bottom-top == 258 for left, top, right, bottom in boxes)
