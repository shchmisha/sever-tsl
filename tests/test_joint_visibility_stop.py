import json

import pytest

from sever.schema import Evidence, FieldValue
from test_image_conflicts import frames, decision, response, run as run_joint
from test_native_conflict_pipeline import prepared, reviewed, run as run_pipeline


@pytest.mark.parametrize("case", ["other_side", "occluded", "blur_or_small_text", "needs_more_zoom",
    "heading_uncertain", "unknown", "mixed_visibility", "incomplete_page", "conflict", "readable",
    "missing_panel", "different_form", "no_explicit_action", "same_timestamp", "empty_context", "nonblank_absence"])
def test_only_explicit_complete_other_side_pixel_assessment_is_terminal(frames, tmp_path, case):
    if case == "same_timestamp":
        frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
    def respond(messages, info):
        payload = decision(messages, statuses={"frame_01": "unreadable", "frame_02": "unreadable"})
        review = payload["fields"]["consignor"]
        review["next_evidence_action"] = "obtain_other_document_side"
        for reading in review["readings"]:
            reading["visibility_cause"] = "other_document_side"
            reading["evidence_text"] = "Дословный видимый контекст продолжения документа"
        if case in {"occluded", "blur_or_small_text", "needs_more_zoom", "heading_uncertain", "unknown"}:
            for reading in review["readings"]:
                reading["visibility_cause"] = case
        elif case == "mixed_visibility":
            review["readings"][1]["visibility_cause"] = "occluded"
        elif case == "incomplete_page":
            review["readings"][1]["complete_page_visible"] = False
        elif case == "conflict":
            review["readings"][1]["status"] = "conflict"
        elif case == "readable":
            review["readings"][1].update(status="recognized", raw_value="Actually readable", visibility_cause="readable")
        elif case == "missing_panel":
            review["readings"][1]["panel_ids"] = []
        elif case == "different_form":
            payload["document_type"] = "ttn"
        elif case == "no_explicit_action":
            del review["next_evidence_action"]
        elif case == "empty_context":
            review["readings"][1]["evidence_text"] = ""
        elif case == "nonblank_absence":
            for reading in review["readings"]:
                reading["status"] = "not_present"
        return response(payload)

    result = run_joint(frames, tmp_path, respond)
    field = result["fields"]["consignor"]
    assessment = result["adjudication"]["field_reviews"]["consignor"]["additional_evidence"]
    assert assessment["terminal_for_current_leaf"] is (case == "other_side")
    assert field.status in {"unreadable", "conflict"}
    assert field.raw_value is None
    assert result["adjudication"]["status"] == "unresolved"
    if case == "other_side":
        assert set(assessment["frame_ids"]) == {frame["frame_id"] for frame in frames}
        assert len(field.evidence) == 2


def other_side_review(chosen, names):
    result = reviewed(chosen, names, resolved=False)
    result["adjudication"]["field_reviews"] = {}
    for name, field in result["fields"].items():
        field.status = "unreadable"
        result["adjudication"]["field_reviews"][name] = {"additional_evidence": {
            "action": "obtain_other_document_side", "terminal_for_current_leaf": True,
            "frame_ids": [frame["frame_id"] for frame in chosen],
            "method": "Fresh joint source-pixel assessment, not independent model calls"}}
    return result


@pytest.mark.parametrize("case", ["other_side", "raw_conflict", "prior_readable_value", "source_error", "missing_evidence",
    "source_attempt_failure", "review_failure"])
def test_joint_other_side_stop_keeps_unreadable_and_never_hides_existing_evidence(tmp_path, monkeypatch, case):
    count = 4 if case == "raw_conflict" else 6
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=count)
    async def reader(frame, **kwargs):
        result = await base_reader(frame, **kwargs)
        if case == "source_attempt_failure" and len(seen) == 1:
            raise ValueError("Deliberate source attempt failure")
        result["document"].fields["consignee"] = FieldValue(status="unreadable")
        if case == "raw_conflict":
            result["document"].fields["consignee"] = FieldValue(status="conflict",
                evidence=[Evidence(frame_id=frame["frame_id"], text="Actually contradictory source line")])
        elif case == "prior_readable_value" and len(seen) == 1:
            result["document"].fields["consignee"] = FieldValue(status="recognized", raw_value="A visible proposal",
                evidence=[Evidence(frame_id=frame["frame_id"], text="A visible proposal")])
        elif case == "source_error":
            result["document"].errors.append("Unrecovered source evidence error")
        return result

    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        if case == "review_failure" and len(calls) == 1:
            raise ValueError("Deliberate failed fresh joint review")
        result = other_side_review(chosen, kwargs["unresolved_fields"])
        if case == "missing_evidence":
            result["fields"]["consignee"].evidence = []
        return result

    out = tmp_path / "run"
    page = run_pipeline(source, out, reader, reviewer)["pages"][0]
    assert len(seen) == (2 if case == "other_side" else count)
    assert page["result"]["fields"]["consignee"]["status"] in {"unreadable", "conflict"}
    assert page["result"]["fields"]["consignee"]["raw_value"] is None
    assert not page["image_recognition"]["stop_reason"].startswith("all_required_fields_confirmed")
    if case == "raw_conflict":
        assert page["result"]["fields"]["consignee"]["status"] == "conflict"
        assert page["image_recognition"]["fusion"]["field_confirmations"]["consignee"]["status"] == "conflict"
    if case in {"source_attempt_failure", "review_failure"}:
        assert page["image_recognition"]["stop_reason"] == "source_views_exhausted"
        assert page["result"]["errors"]
    if case == "other_side":
        assert len(calls) == 1
        assert page["image_recognition"]["stop_reason"] == "requires_other_document_side"
        assessment = page["image_recognition"]["leaf_visibility_assessments"]["consignee"]
        assert assessment["remaining_source_views_unread"] is True
        assert assessment["field_status"] == "unreadable"
        assert len(page["image_recognition"]["views"]) == 2
        saved = json.loads((out / "results.json").read_text())["pages"][0]
        assert saved["result"]["fields"]["consignee"]["status"] == "unreadable"
        assert saved["image_recognition"]["leaf_visibility_assessments"] == page["image_recognition"]["leaf_visibility_assessments"]
