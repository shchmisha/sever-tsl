import asyncio
from copy import deepcopy
import hashlib
import json

from PIL import Image, ImageDraw
import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded, FatalProviderError
from sever.native_pipeline import process_video_images, _type_conflict_review_frames
from sever.schema import Evidence, FieldValue
from test_native_pipeline import setup_pages, answer


def prepared(tmp_path, monkeypatch, kinds, *, unreadable=None, modify=None):
    source, frames = setup_pages(tmp_path, monkeypatch, count=len(kinds))
    for index, frame in enumerate(frames):
        with Image.open(frame["path"]) as image:
            ImageDraw.Draw(image).point((100 + index, 101), fill=(index, 40, 80))
            image.save(frame["path"])
    seen = []

    async def reader(frame, *, source_id, page_id, **kwargs):
        index = int(frame["frame_id"].split("-")[-1])
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, kinds[index], label="Own heading " + kinds[index])
        if unreadable and unreadable in result["document"].fields:
            result["document"].fields[unreadable] = FieldValue(status="unreadable", reason="Unreadable source pixels")
        result["recognition"]["pixel_proof"] = {
            "source_sha256": hashlib.sha256(open(frame["path"], "rb").read()).hexdigest(),
            "rotation_clockwise_degrees": 90,
        }
        if modify:
            modify(index, result)
        return result
    return source, frames, reader, seen


def type_answer(frames, kind="ttn", *, resolved=True, foreign=False):
    evidence = [Evidence(frame_id="foreign" if foreign else frame["frame_id"],
                         text="Fresh own heading", bbox=[0, 0, 1, 1]).model_dump(mode="json")
                for frame in frames]
    return {"document_type": kind if resolved else "unknown", "fields": {},
            "classification_evidence": evidence, "panels": [],
            "recognition": {"raw_decision": {"fresh_joint_type": kind, "resolved": resolved}},
            "adjudication": {"status": "resolved" if resolved else "unresolved",
                             "resolved_document_type": kind if resolved else None}}


def unreadable_fields(frames, names):
    """Fresh field-review fixture; a form-only answer cannot confirm a field."""
    return {"fields": {name: FieldValue(status="unreadable", reason="Own field unreadable on supplied images")
                        for name in names}, "recognition": {"fresh_field_review": True},
            "adjudication": {"status": "unresolved"}, "panels": []}


def run(source, out, reader, reviewer):
    return asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
        conflict_reader=reviewer, additional_frames=lambda *args: iter(())))


def test_fresh_joint_type_keeps_all_raw_answers_and_does_not_fabricate_field_reads(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "trn"])
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append((deepcopy(chosen), set(kwargs)))
        if kwargs["document_type"] != "unknown":
            return unreadable_fields(chosen, kwargs["unresolved_fields"])
        assert kwargs["unresolved_fields"] == []
        assert "Own heading" not in json.dumps(chosen)
        assert "raw_value" not in json.dumps(chosen) and "document_type" not in json.dumps(chosen)
        return type_answer(chosen)

    result = run(source, tmp_path / "run", reader, reviewer)
    page = result["pages"][0]
    assert len(seen) == 2 and len(calls) == 2
    assert calls[0][1] == {"source_id", "page_id", "document_type", "unresolved_fields", "config", "cache_dir", "trace_path"}
    assert page["result"]["document_type"] == "ttn"
    assert all(field["status"] == "unreadable" and field["raw_value"] is None for field in page["result"]["fields"].values())
    fusion = page["image_recognition"]["fusion"]
    assert [reading["document_type"] for reading in fusion["readings"]] == ["ttn", "trn"]
    assert len(fusion["readings"]) == fusion["coverage"]["reading_count"] == 2
    assert fusion["type_confirmation"]["joint_review_frame_ids"] == [frame["frame_id"] for frame in frames]
    assert fusion["type_confirmation"]["original_frame_ids_by_type"] == {"ttn": ["frame-0"], "trn": ["frame-1"]}
    assert fusion["model_type_adjudication"]["fields_read"] is False
    assert [view["document"]["document_type"] for view in page["image_recognition"]["views"]] == ["ttn", "trn"]
    review = page["image_recognition"]["conflict_reviews"][0]
    assert review["kind"] == "document_type" and review["field_names"] == []
    assert review["status"] == "resolved" and review["answer"]["recognition"]["raw_decision"]["fresh_joint_type"] == "ttn"


def test_resolved_type_accepts_later_same_form_fields_without_rediscovering_old_tags(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "trn", "ttn", "trn"])
    calls = []

    async def reviewer(chosen, **kwargs):
        if kwargs["document_type"] != "unknown":
            return unreadable_fields(chosen, kwargs["unresolved_fields"])
        calls.append(chosen)
        return type_answer(chosen)

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert seen == [frame["frame_id"] for frame in frames[:3]] and len(calls) == 1
    assert page["image_recognition"]["stop_reason"] == "all_required_fields_confirmed"
    assert all(field["status"] == "recognized" for field in page["result"]["fields"].values())
    assert len(page["image_recognition"]["fusion"]["readings"]) == 3
    # Different-form field names never leak into the selected contract.
    assert "customer" not in page["result"]["fields"]
    assert page["image_recognition"]["views"][1]["document"]["fields"]["customer"]["raw_value"] == "example-customer"


def test_later_grounded_contradiction_invalidates_resolution_until_fresh_final_review(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch,
        ["ttn", "trn", "torg12", "ttn"], unreadable="document_date")
    calls, snapshots = [], []

    async def reviewer(chosen, **kwargs):
        if kwargs["document_type"] != "unknown":
            return unreadable_fields(chosen, kwargs["unresolved_fields"])
        calls.append([frame["frame_id"] for frame in chosen])
        checkpoint = json.loads((tmp_path / "run/partial-results.json").read_text())
        snapshots.append(checkpoint["pages"][0])
        return type_answer(chosen, "ttn")

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert seen == [frame["frame_id"] for frame in frames]
    assert len(calls) == 2 and set(calls[1]) - set(calls[0])
    assert snapshots[1]["result"]["document_type"] == "unknown"
    assert {"frame-0", "frame-1", "frame-2"} <= set(calls[1])
    reviews = [entry for entry in page["image_recognition"]["conflict_reviews"] if entry["kind"] == "document_type"]
    assert [review["phase"] for review in reviews] == ["initial", "final"]
    assert reviews[0]["invalidated_by"]["frame_id"] == "frame-2"
    assert page["result"]["document_type"] == "ttn"
    assert page["result"]["fields"]["document_date"]["status"] == "unreadable"
    assert [item["document_type"] for item in page["image_recognition"]["fusion"]["readings"]] == ["ttn", "trn", "torg12", "ttn"]


def test_unresolved_type_review_does_not_cap_frame_search_or_choose_a_majority(tmp_path, monkeypatch):
    kinds = ["ttn", "trn", "ttn", "ttn", "torg12", "upd", "ttn"]
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, kinds)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append([frame["frame_id"] for frame in chosen])
        return type_answer(chosen, resolved=False)

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == len(frames) and len(calls) == 2
    assert len(calls[1]) == 4 and {"frame-0", "frame-1", "frame-4", "frame-5"} == set(calls[1])
    assert page["result"]["document_type"] == "unknown" and page["result"]["fields"] == {}
    assert len(page["image_recognition"]["fusion"]["readings"]) == len(kinds)
    assert all(review["status"] == "unresolved" for review in page["image_recognition"]["conflict_reviews"])


def test_invalidated_joint_type_never_falls_back_to_its_old_resolved_answer(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch,
        ["ttn", "trn", "upd"], unreadable="document_date")
    calls = []

    async def reviewer(chosen, **kwargs):
        if kwargs["document_type"] != "unknown":
            return unreadable_fields(chosen, kwargs["unresolved_fields"])
        calls.append(chosen)
        return type_answer(chosen, resolved=len(calls) == 1)

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(calls) == 2 and len(seen) == 3
    assert page["result"]["document_type"] == "unknown" and page["result"]["fields"] == {}
    reviews = [entry for entry in page["image_recognition"]["conflict_reviews"] if entry["kind"] == "document_type"]
    assert reviews[0]["status"] == "resolved" and reviews[0]["invalidated_by"]["frame_id"] == "frame-2"
    assert reviews[1]["status"] == "unresolved"
    assert "model_type_adjudication" not in page["image_recognition"]["fusion"]
    assert [reading["document_type"] for reading in page["image_recognition"]["fusion"]["readings"]] == ["ttn", "trn", "upd"]


def selection_fixture(kinds):
    frames = [{"frame_id": str(index), "timestamp_s": index + 1,
               "path": str(index), "quality": {"relative_selection_score": 100-index}}
              for index in range(len(kinds))]
    readings = [answer(frame, "source", "page", kind)["document"] for frame, kind in zip(frames, kinds)]
    views = [{"frame_id": frame["frame_id"], "pixel_proof": {
        "source_sha256": f"{index:064x}", "rotation_clockwise_degrees": 90}}
             for index, frame in enumerate(frames)]
    return {"frames": frames}, readings, views


def test_type_selection_includes_single_minority_form_among_eighteen_other_views():
    page, readings, views = selection_fixture(["ttn"] * 18 + ["trn"])
    chosen = _type_conflict_review_frames(page, readings, views)
    assert {"0", "18"} <= {frame["frame_id"] for frame in chosen}
    assert len(chosen) == 4
    assert all("document_type" not in frame for frame in chosen)


def test_final_selection_with_four_type_groups_keeps_every_group_and_new_pixels():
    page, readings, views = selection_fixture(["ttn", "trn", "torg12", "upd", "ttn"])
    chosen = _type_conflict_review_frames(page, readings, views, previous_frame_ids={"0", "1", "2", "3"})
    assert {frame["frame_id"] for frame in chosen} == {"1", "2", "3", "4"}


@pytest.mark.parametrize("duplicate", ["sha", "pts", "missing_proof"])
def test_type_selection_requires_distinct_source_sha_and_pts_from_every_group(duplicate):
    page, readings, views = selection_fixture(["ttn", "trn"])
    if duplicate == "sha":
        views[1]["pixel_proof"]["source_sha256"] = views[0]["pixel_proof"]["source_sha256"]
    elif duplicate == "pts":
        page["frames"][1]["timestamp_s"] = page["frames"][0]["timestamp_s"]
    else:
        views[1]["pixel_proof"] = {}
    assert _type_conflict_review_frames(page, readings, views) == []


def test_selection_can_choose_lower_quality_independent_group_representative():
    page, readings, views = selection_fixture(["ttn", "trn", "ttn"])
    views[0]["pixel_proof"]["source_sha256"] = views[1]["pixel_proof"]["source_sha256"]
    chosen = _type_conflict_review_frames(page, readings, views)
    assert {frame["frame_id"] for frame in chosen} == {"1", "2"}


@pytest.mark.parametrize("failure", ["foreign", "one_cite", "fabricated_fields", "invalid_type"])
def test_invalid_type_review_cannot_select_form_or_replace_original_answers(tmp_path, monkeypatch, failure):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "trn"])

    async def reviewer(chosen, **kwargs):
        result = type_answer(chosen, foreign=failure == "foreign")
        if failure == "one_cite":
            result["classification_evidence"] = result["classification_evidence"][:1]
        elif failure == "fabricated_fields":
            result["fields"] = {"document_number": {"status": "recognized", "raw_value": "not a field read"}}
        elif failure == "invalid_type":
            result["document_type"] = "invented"
        return result

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert page["result"]["document_type"] == "unknown"
    assert page["image_recognition"]["conflict_reviews"][0]["status"] == "failed"
    assert len(page["image_recognition"]["fusion"]["readings"]) == 2
    assert any("conflict_review_failed:ValueError" in error for error in page["result"]["errors"])


@pytest.mark.parametrize("error", [BatchBudgetExceeded, FatalProviderError])
@pytest.mark.parametrize("storage_failure", [False, True])
def test_type_review_fatal_budget_primary_survives_checkpoint_storage_failure(tmp_path, monkeypatch, error, storage_failure):
    import sever.native_pipeline as pipeline
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "trn", "ttn"])
    original_save = pipeline.save_json

    def save(path, data):
        if storage_failure and isinstance(data, dict) and data.get("status") == "failed":
            raise OSError("fatal checkpoint disk failure")
        return original_save(path, data)

    monkeypatch.setattr(pipeline, "save_json", save)

    async def reviewer(*args, **kwargs):
        raise error("original type review failure")

    with pytest.raises(error, match="original type review failure"):
        run(source, tmp_path / "run", reader, reviewer)
    assert len(seen) == 2
    if not storage_failure:
        checkpoint = json.loads((tmp_path / "run/partial-results.json").read_text())
        assert checkpoint["status"] == "failed"
        review = checkpoint["pages"][0]["image_recognition"]["conflict_reviews"][0]
        assert review["status"] == "failed" and review["error_type"] == error.__name__


def test_nonselected_errors_preserved_and_type_entry_does_not_suppress_initial_field_review(tmp_path, monkeypatch):
    def modify(index, result):
        if index == 1:
            result["document"].errors.append("Original trn field structural error")
        if index == 2:
            result["document"].fields["consignee"].raw_value = "Different actual proposal"

    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "trn", "ttn"], modify=modify)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["document_type"])
        if kwargs["document_type"] == "unknown":
            return type_answer(chosen)
        evidence = [Evidence(frame_id=frame["frame_id"], text="Own field", bbox=[0, 0, 1, 1]) for frame in chosen]
        return {"fields": {name: FieldValue(status="recognized", raw_value="Fresh model field", evidence=evidence)
                           for name in kwargs["unresolved_fields"]}, "recognition": {},
                "adjudication": {"status": "resolved"}, "panels": []}

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert calls == ["unknown", "ttn"]
    assert [review["kind"] for review in page["image_recognition"]["conflict_reviews"]] == ["document_type", "fields"]
    assert [review["phase"] for review in page["image_recognition"]["conflict_reviews"]] == ["initial", "initial"]
    assert page["result"]["errors"] == []
    assert {"error": "Original trn field structural error", "recovered": True} in page["image_recognition"]["error_history"]
    assert page["result"]["fields"]["consignee"]["raw_value"] == "Fresh model field"
    assert len(page["image_recognition"]["fusion"]["readings"]) == 2
    assert len(seen) == 2  # No third full read once the new pixel review recovered every field.


def test_pending_own_heading_gets_fresh_type_review_without_discarding_read_fields(tmp_path, monkeypatch):
    def remove_heading(index, result):
        result["document"].classification_evidence = []
        result["document"].errors = ["Model document type has no supplied-panel heading evidence"]
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, ["ttn", "ttn", "ttn"], modify=remove_heading)
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["document_type"])
        assert kwargs["unresolved_fields"] == []
        return type_answer(chosen)
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert calls == ["unknown"] and len(seen) == 2
    assert page["result"]["document_type"] == "ttn"
    assert page["result"]["errors"] == []
    assert all(field["status"] == "recognized" for field in page["result"]["fields"].values())
    fusion = page["image_recognition"]["fusion"]
    assert fusion["model_type_adjudication"]["fields_read"] is False
    assert fusion["type_confirmation"]["original_frame_ids_by_type"] == {}
    assert fusion["field_confirmations"]["document_number"]["value_groups"][0]["frame_ids"] == ["frame-0", "frame-1"]


@pytest.mark.parametrize("final_resolved", [False, True])
def test_joint_heading_evidence_survives_invalidation_until_a_fresh_type_review(tmp_path, monkeypatch, final_resolved):
    def initial_missing_heading(index, value):
        if index < 2:
            value["document"].classification_evidence = []
            value["document"].fields["consignee"] = FieldValue(status="unreadable")
    source, frames, reader, seen = prepared(tmp_path, monkeypatch,
        ["ttn", "ttn", "trn", "trn"], modify=initial_missing_heading)
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["document_type"])
        if kwargs["document_type"] != "unknown":
            return unreadable_fields(chosen, kwargs["unresolved_fields"])
        if len(calls) == 1:
            return type_answer(chosen, "ttn")
        checkpoint = json.loads((tmp_path / "run/partial-results.json").read_text())
        assert checkpoint["pages"][0]["result"]["document_type"] == "unknown"
        assert {"frame-0", "frame-2"} <= {frame["frame_id"] for frame in chosen}
        return type_answer(chosen, "trn", resolved=final_resolved)
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert calls == ["unknown", "ttn", "unknown"] and len(seen) == 4
    assert page["result"]["document_type"] == ("trn" if final_resolved else "unknown")
    reviews = page["image_recognition"]["conflict_reviews"]
    assert reviews[0]["answer"]["document_type"] == "ttn"
    assert reviews[0]["invalidated_by"]["frame_id"] == "frame-2"
    assert ("pending_model_type_conflict" in page["image_recognition"]["fusion"]) is not final_resolved
