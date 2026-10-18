import asyncio
from copy import deepcopy
import hashlib
import json

from PIL import Image, ImageDraw
import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded, FatalProviderError
from sever.image_fusion import _field_comparison
from sever.native_pipeline import process_video_images, _conflict_review_frames
from sever.schema import Evidence, FieldValue
from test_native_pipeline import setup_pages, answer


def prepared(tmp_path, monkeypatch, count=4):
    source, frames = setup_pages(tmp_path, monkeypatch, count=count)
    for index, frame in enumerate(frames):
        with Image.open(frame["path"]) as image:
            ImageDraw.Draw(image).point((100 + index, 101), fill=(index, 40, 80))
            image.save(frame["path"])
    seen = []

    async def reader(frame, *, source_id, page_id, **kwargs):
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, "upd")
        # Alternate two readings so no value reaches strong agreement; the
        # dispute can only be settled by a joint review.
        result["document"].fields["consignee"].raw_value = "Organization A" if len(seen) % 2 else "Organization A; auxiliary identifier"
        result["recognition"]["pixel_proof"] = {
            "source_sha256": hashlib.sha256(open(frame["path"], "rb").read()).hexdigest(),
            "rotation_clockwise_degrees": 90,
        }
        return result
    return source, frames, reader, seen


def reviewed(frames, names, *, resolved=True, foreign=False):
    evidence = [Evidence(frame_id="foreign" if foreign else frame["frame_id"],
                        text="Грузополучатель: Organization A", bbox=[0, 0, 1, 1]) for frame in frames]
    fields = {name: FieldValue(status="recognized" if resolved else "conflict",
                              raw_value="Organization A" if resolved else None,
                              normalized_value="Organization A" if resolved else None,
                              reason="Fresh model separation of the own role from adjacent identifiers",
                              evidence=evidence) for name in names}
    return {"fields": fields, "recognition": {"raw_decision": {"fresh": True}},
            "adjudication": {"status": "resolved" if resolved else "unresolved"}, "panels": []}


def run(source, out, reader, reviewer, **options):
    # These scenarios exercise review coverage across every supplied frame;
    # the no-progress leaf stop has its own tests and is disabled by default here.
    options.setdefault("no_progress_views", None)
    return asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
        conflict_reader=reviewer, additional_frames=lambda *args: iter(()), **options))


def test_joint_review_uses_pixels_and_field_names_without_previous_answers(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append((deepcopy(chosen), set(kwargs)))
        assert kwargs["unresolved_fields"] == ["consignee"]
        assert all("raw_value" not in json.dumps(frame) for frame in chosen)
        assert "Organization A" not in json.dumps(chosen)
        return reviewed(chosen, kwargs["unresolved_fields"])

    result = run(source, tmp_path / "run", reader, reviewer)
    page = result["pages"][0]
    assert len(seen) == 2 and len(calls) == 1
    assert calls[0][1] == {"source_id", "page_id", "document_type", "unresolved_fields", "config", "cache_dir", "trace_path"}
    assert page["result"]["fields"]["consignee"]["raw_value"] == "Organization A"
    originals = [view["document"]["fields"]["consignee"]["raw_value"] for view in page["image_recognition"]["views"]]
    assert originals == ["Organization A", "Organization A; auxiliary identifier"]
    review = page["image_recognition"]["conflict_reviews"][0]
    assert review["status"] == "resolved" and review["phase"] == "initial"
    assert review["answer"]["recognition"]["raw_decision"] == {"fresh": True}
    assert page["image_recognition"]["fusion"]["field_confirmations"]["consignee"]["resolution"] == "fresh_joint_model_review"
    assert "model adjudication" in " ".join(page["result"]["notes"])


def test_unresolved_review_keeps_reading_all_frames_then_reviews_new_pixels(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=7)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append([frame["frame_id"] for frame in chosen])
        return reviewed(chosen, kwargs["unresolved_fields"], resolved=len(calls) > 1)

    result = run(source, tmp_path / "run", reader, reviewer)
    page = result["pages"][0]
    assert seen == [frame["frame_id"] for frame in frames]
    assert len(calls) == 2 and set(calls[1]) - set(calls[0])
    assert all(len(call) <= 4 for call in calls)
    assert page["result"]["fields"]["consignee"]["status"] == "recognized"
    assert page["image_recognition"]["stop_reason"] == "all_required_fields_confirmed_after_joint_review"
    assert [item["status"] for item in page["image_recognition"]["conflict_reviews"]] == ["unresolved", "resolved"]


def test_same_pixels_are_not_resubmitted_until_review_happens_to_agree(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=2)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append(chosen)
        return reviewed(chosen, kwargs["unresolved_fields"], resolved=False)

    result = run(source, tmp_path / "run", reader, reviewer)
    assert len(calls) == 1 and len(seen) == 2
    assert result["pages"][0]["result"]["fields"]["consignee"]["status"] == "conflict"


@pytest.mark.parametrize("error", [BatchBudgetExceeded, FatalProviderError])
def test_joint_review_fatal_or_budget_stop_preserves_both_originals(tmp_path, monkeypatch, error):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch)

    async def reviewer(*args, **kwargs):
        raise error("offline deliberate failure")

    out = tmp_path / "run"
    with pytest.raises(error):
        run(source, out, reader, reviewer)
    checkpoint = json.loads((out / "partial-results.json").read_text())
    assert checkpoint["status"] == "failed"
    page = checkpoint["pages"][0]
    assert len(page["image_recognition"]["views"]) == 2
    assert page["image_recognition"]["conflict_reviews"][0]["status"] == "failed"
    assert page["result"]["fields"]["consignee"]["status"] == "conflict"
    assert not (out / "results.json").exists()


def test_foreign_frame_review_cannot_replace_a_conflicting_field(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=2)

    async def reviewer(chosen, **kwargs):
        return reviewed(chosen, kwargs["unresolved_fields"], foreign=True)

    result = run(source, tmp_path / "run", reader, reviewer)
    page = result["pages"][0]
    assert page["result"]["fields"]["consignee"]["status"] == "conflict"
    assert page["image_recognition"]["conflict_reviews"][0]["status"] == "failed"


def test_final_review_failure_keeps_latest_partial_report_and_all_source_reads(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=5)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append(chosen)
        if len(calls) == 2:
            raise BatchBudgetExceeded("offline final review stopped")
        return reviewed(chosen, kwargs["unresolved_fields"], resolved=False)

    out = tmp_path / "run"
    with pytest.raises(BatchBudgetExceeded):
        run(source, out, reader, reviewer)
    checkpoint = json.loads((out / "partial-results.json").read_text())
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["pages"][0]["image_recognition"]["views"]) == 5
    assert len(checkpoint["pages"][0]["image_recognition"]["conflict_reviews"]) == 2
    assert 'http-equiv="refresh"' not in (out / "report.html").read_text()


def test_review_selection_requires_new_distinct_source_identity_not_value_majority():
    frames = [{"frame_id": str(i), "timestamp_s": i, "quality": {"relative_selection_score": 10-i}, "path": str(i)} for i in range(6)]
    views = [{"frame_id": str(i), "pixel_proof": {"source_sha256": f"{i:064x}", "rotation_clockwise_degrees": 90}} for i in range(6)]
    selected = _conflict_review_frames({"frames": frames}, views, previous_frame_ids={"0", "1", "2", "3"})
    assert selected[0]["frame_id"] == "4" and len(selected) == 4
    views[4]["pixel_proof"]["source_sha256"] = views[0]["pixel_proof"]["source_sha256"]
    frames[5]["timestamp_s"] = frames[0]["timestamp_s"]
    assert _conflict_review_frames({"frames": frames}, views, previous_frame_ids={"0", "1", "2", "3"}) == []


@pytest.mark.parametrize("failure_site", ["checkpoint", "trace"])
@pytest.mark.parametrize("error", [BatchBudgetExceeded, FatalProviderError])
def test_failure_to_publish_a_fatal_review_does_not_resume_inference(tmp_path, monkeypatch, failure_site, error):
    import sever.native_pipeline as pipeline
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=5)
    original_save, original_trace = pipeline.save_json, pipeline.trace

    def save(path, data):
        if failure_site == "checkpoint" and isinstance(data, dict) and data.get("status") == "failed":
            raise OSError("disk unavailable during fatal checkpoint")
        return original_save(path, data)

    def trace(path, event, **kwargs):
        if failure_site == "trace" and kwargs.get("status") == "stopped":
            raise OSError("trace unavailable during fatal stop")
        return original_trace(path, event, **kwargs)

    monkeypatch.setattr(pipeline, "save_json", save)
    monkeypatch.setattr(pipeline, "trace", trace)

    async def reviewer(*args, **kwargs):
        raise error("the original fatal reason")

    with pytest.raises(error, match="original fatal reason"):
        run(source, tmp_path / "run", reader, reviewer)
    assert len(seen) == 2


def test_unreadable_field_receives_targeted_fresh_review_and_stops_full_rereads(tmp_path, monkeypatch):
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=6)
    async def reader(frame, **kwargs):
        result = await base_reader(frame, **kwargs)
        result["document"].fields["consignee"] = FieldValue(status="unreadable")
        return result
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        return reviewed(chosen, kwargs["unresolved_fields"])
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == 2 and calls == [["consignee"]]
    assert page["result"]["fields"]["consignee"]["status"] == "recognized"
    assert all(view["document"]["fields"]["consignee"]["status"] == "unreadable" for view in page["image_recognition"]["views"])


@pytest.mark.parametrize("contradiction", [False, True])
def test_partial_joint_recovery_survives_unreadable_views_but_not_new_readable_conflict(tmp_path, monkeypatch, contradiction):
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=4)
    async def reader(frame, **kwargs):
        result = await base_reader(frame, **kwargs)
        result["document"].fields["consignee_address"] = FieldValue(status="unreadable")
        if len(seen) > 2:
            result["document"].fields["consignee"] = (FieldValue(status="recognized", raw_value="Contradictory new pixels",
                evidence=[Evidence(frame_id=frame["frame_id"], text="New literal line")]) if contradiction else FieldValue(status="unreadable"))
        return result
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        if len(calls) > 1:
            return reviewed(chosen, kwargs["unresolved_fields"], resolved=False)
        value = reviewed(chosen, kwargs["unresolved_fields"])
        value["fields"]["consignee_address"] = FieldValue(status="unreadable")
        return value
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == 4
    assert calls[0] == ["consignee", "consignee_address"]
    assert calls[-1] == (["consignee", "consignee_address"] if contradiction else ["consignee_address"])
    assert page["result"]["fields"]["consignee"]["status"] == ("conflict" if contradiction else "recognized")
    first = page["image_recognition"]["conflict_reviews"][0]
    assert ("consignee" in first.get("invalidated_fields", {})) == contradiction


def test_lost_joint_review_evidence_stays_active_after_a_later_successful_review(tmp_path, monkeypatch):
    from sever.image_agent import EvidencePersistenceError
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=4)
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(chosen)
        if len(calls) == 1:
            raise EvidencePersistenceError("Unable to preserve review response")
        return reviewed(chosen, kwargs["unresolved_fields"])
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    error = "native_image: conflict_review_failed:EvidencePersistenceError"
    assert len(calls) == 2
    assert all(field["status"] == "recognized" for field in page["result"]["fields"].values())
    assert error in page["result"]["errors"]
    assert page["image_recognition"]["conflict_reviews"][0]["evidence_persistence_failed"] is True
    assert {"error": error, "recovered": False} in page["image_recognition"]["error_history"]


@pytest.mark.parametrize("final_resolved", [False, True])
def test_readable_joint_value_cannot_be_outvoted_by_later_raw_reads_without_fresh_review(tmp_path, monkeypatch, final_resolved):
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=4)
    async def reader(frame, **kwargs):
        value = await base_reader(frame, **kwargs)
        for name, raw in (("consignee", "New B"), ("consignee_address", "New readable address")):
            value["document"].fields[name] = (FieldValue(status="unreadable") if len(seen) <= 2 else
                FieldValue(status="recognized", raw_value=raw, evidence=[Evidence(frame_id=frame["frame_id"], text=raw)]))
        return value
    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        if len(calls) == 1:
            value = reviewed(chosen, kwargs["unresolved_fields"])
            value["fields"]["consignee_address"] = FieldValue(status="unreadable")
            return value
        checkpoint = json.loads((tmp_path / "run/partial-results.json").read_text())
        assert checkpoint["pages"][0]["result"]["fields"]["consignee"]["status"] == "conflict"
        value = reviewed(chosen, kwargs["unresolved_fields"], resolved=final_resolved)
        if final_resolved:
            value["fields"]["consignee"].raw_value = "New B"
            value["fields"]["consignee"].normalized_value = "New B"
        return value
    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert calls == [["consignee", "consignee_address"], ["consignee"]]
    assert len(seen) == 4
    field = page["result"]["fields"]["consignee"]
    assert field["status"] == ("recognized" if final_resolved else "conflict")
    assert field["raw_value"] == ("New B" if final_resolved else None)
    original = page["image_recognition"]["conflict_reviews"][0]
    assert original["answer"]["fields"]["consignee"]["raw_value"] == "Organization A"
    assert "consignee" in original["invalidated_fields"]


@pytest.mark.parametrize("initial_joint_resolved", [False, True])
def test_final_field_review_includes_low_quality_contradictory_pixels(tmp_path, monkeypatch, initial_joint_resolved):
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=6)

    async def reader(frame, **kwargs):
        value = await base_reader(frame, **kwargs)
        value["document"].fields["consignee_address"] = FieldValue(status="unreadable")
        if frame["frame_id"] == "frame-5":
            value["document"].fields["consignee"] = FieldValue(status="recognized", raw_value="Later contradictory source value",
                evidence=[Evidence(frame_id=frame["frame_id"], text="Later contradictory source value")])
        elif initial_joint_resolved or len(seen) > 2:
            value["document"].fields["consignee"] = FieldValue(status="unreadable")
        return value

    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append({frame["frame_id"] for frame in chosen})
        assert "Later contradictory source value" not in json.dumps(chosen)
        if len(calls) > 1:
            assert "frame-5" in calls[-1]
            assert calls[-1] & calls[0]
        result = reviewed(chosen, kwargs["unresolved_fields"], resolved=initial_joint_resolved or len(calls) > 1)
        if len(calls) == 1:
            result["fields"]["consignee_address"] = FieldValue(status="unreadable")
        return result

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == 6 and len(calls) == 2
    assert page["result"]["fields"]["consignee"]["status"] == "recognized"
    assert page["image_recognition"]["stop_reason"] == "all_required_fields_confirmed_after_joint_review"


def test_field_stays_conflicting_when_all_disputed_evidence_cannot_fit_joint_context(tmp_path, monkeypatch):
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=6)

    async def reader(frame, **kwargs):
        value = await base_reader(frame, **kwargs)
        value["document"].fields["consignee_address"] = FieldValue(status="unreadable")
        value["document"].fields["consignee"] = (FieldValue(status="unreadable") if len(seen) <= 2 else
            FieldValue(status="recognized", raw_value=f"Different source value {len(seen)}",
                evidence=[Evidence(frame_id=frame["frame_id"], text=f"Different source value {len(seen)}")]))
        return value

    calls = []
    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        result = reviewed(chosen, kwargs["unresolved_fields"])
        if len(calls) == 1:
            result["fields"]["consignee_address"] = FieldValue(status="unreadable")
        return result

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == 6
    assert calls == [["consignee", "consignee_address"], ["consignee_address"]]
    assert page["result"]["fields"]["consignee"]["status"] == "conflict"
    assert page["result"]["fields"]["consignee_address"]["status"] == "recognized"
    final_review = page["image_recognition"]["conflict_reviews"][-1]
    assert final_review["status"] == "unresolved"
    assert final_review["blocked_fields"]["consignee"]["uncovered_frame_groups"]
    assert page["image_recognition"]["stop_reason"] == "source_views_exhausted"


def test_no_progress_stop_leaves_frames_unread_but_still_runs_final_review(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=7)
    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append([frame["frame_id"] for frame in chosen])
        return reviewed(chosen, kwargs["unresolved_fields"], resolved=len(calls) > 1)

    out = tmp_path / "run"
    page = run(source, out, reader, reviewer, no_progress_views=3)["pages"][0]
    # Read 1 proposes, read 2 creates the conflict, reads 3-5 add nothing new.
    assert seen == [frame["frame_id"] for frame in frames[:5]]
    assert len(calls) == 2 and set(calls[1]) - set(calls[0])
    assert page["result"]["fields"]["consignee"]["status"] == "recognized"
    assert page["image_recognition"]["stop_reason"] == "all_required_fields_confirmed_after_joint_review"
    assert page["image_recognition"]["no_progress_stop"]["triggered"] is True
    events = [json.loads(line)["event"] for line in (out / "trace.jsonl").read_text().splitlines()]
    assert events.count("leaf_search_stopped_without_progress") == 1


def test_joint_review_value_no_frame_ever_read_is_recorded_but_not_adopted(tmp_path, monkeypatch):
    # The first frame swallows a caption into the address, so the second read
    # creates a conflict and the initial joint review answers with a truncated
    # value no frame read (adopted: no two frames agree yet). Later frames agree
    # with the second reading, which invalidates that answer; the final review
    # repeats the stray value, now rejected, and strong agreement settles it.
    # consignee keeps alternating (3 vs 2) so the leaf reads every frame.
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=5)
    address = 'ООО "А"; 628422, г. Сургут, ул. Мира, д. 21, соор. 16'
    sequence = ["Адрес: " + address] + [address] * 4

    async def reader(frame, **kwargs):
        value = await base_reader(frame, **kwargs)
        value["document"].fields["consignee_address"].raw_value = sequence[len(seen) - 1]
        return value

    calls = []

    async def reviewer(chosen, **kwargs):
        calls.append(kwargs["unresolved_fields"])
        answer = reviewed(chosen, kwargs["unresolved_fields"])
        answer["fields"]["consignee"] = FieldValue(status="unreadable", reason="still disputed")
        answer["fields"]["consignee_address"].raw_value = "ул., д. 21, соор. 16"
        answer["fields"]["consignee_address"].normalized_value = "ул., д. 21, соор. 16"
        return answer

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    # Once four frames agree, the address is settled by strong agreement and the
    # final review is asked only about the still-disputed consignee.
    assert len(seen) == 5 and calls == [["consignee", "consignee_address"], ["consignee"]]
    field = page["result"]["fields"]["consignee_address"]
    assert field["status"] == "recognized" and field["raw_value"] == address
    assert page["result"]["fields"]["consignee"]["status"] == "conflict"
    confirmation = page["image_recognition"]["fusion"]["field_confirmations"]["consignee_address"]
    assert confirmation["strong_agreement"]["agreeing_frame_ids"] == ["frame-1", "frame-2", "frame-3", "frame-4"]
    assert confirmation["superseded_joint_review"]["raw_value"] == "ул., д. 21, соор. 16"
    initial, final = page["image_recognition"]["conflict_reviews"]
    assert initial["phase"] == "initial" and initial["resolved_fields"] == ["consignee_address"]
    assert initial["invalidated_fields"]["consignee_address"]["frame_ids"] == ["frame-2"]
    assert final["phase"] == "final" and not final.get("resolved_fields") and "rejected_novel_values" not in final
    assert [v["document"]["fields"]["consignee_address"]["raw_value"] for v in page["image_recognition"]["views"]] == sequence


def test_joint_review_novel_value_is_rejected_once_two_frames_agree(tmp_path, monkeypatch):
    # Three frames: a slip, then two identical readings. Too few for strong
    # agreement, but enough that a joint value matching no reading is refused.
    source, frames, base_reader, seen = prepared(tmp_path, monkeypatch, count=3)
    address = 'ООО "А"; 628422, г. Сургут, ул. Мира, д. 21, соор. 16'
    sequence = ["Адрес: " + address, address, address]

    async def reader(frame, **kwargs):
        value = await base_reader(frame, **kwargs)
        value["document"].fields["consignee_address"].raw_value = sequence[len(seen) - 1]
        return value

    async def reviewer(chosen, **kwargs):
        answer = reviewed(chosen, kwargs["unresolved_fields"])
        answer["fields"]["consignee"] = FieldValue(status="unreadable", reason="still disputed")
        answer["fields"]["consignee_address"].raw_value = "ул., д. 21, соор. 16"
        answer["fields"]["consignee_address"].normalized_value = "ул., д. 21, соор. 16"
        return answer

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert len(seen) == 3
    assert page["result"]["fields"]["consignee_address"]["status"] == "conflict"
    initial, final = page["image_recognition"]["conflict_reviews"]
    assert initial["resolved_fields"] == ["consignee_address"]          # 1 vs 1: nothing dominant yet
    assert final["resolved_fields"] == []
    assert final["rejected_novel_values"]["consignee_address"]["single_frame_values"] == \
        sorted([_field_comparison("consignee_address", address), _field_comparison("consignee_address", "Адрес: " + address)])


def test_joint_review_may_settle_a_field_single_frames_scatter_on(tmp_path, monkeypatch):
    source, frames, reader, seen = prepared(tmp_path, monkeypatch, count=2)

    async def reviewer(chosen, **kwargs):
        answer = reviewed(chosen, kwargs["unresolved_fields"])
        answer["fields"]["consignee"].raw_value = 'ООО "Organization A"'
        answer["fields"]["consignee"].normalized_value = 'ООО "Organization A"'
        return answer

    page = run(source, tmp_path / "run", reader, reviewer)["pages"][0]
    assert page["result"]["fields"]["consignee"]["raw_value"] == 'ООО "Organization A"'
    assert "rejected_novel_values" not in page["image_recognition"]["conflict_reviews"][0]
