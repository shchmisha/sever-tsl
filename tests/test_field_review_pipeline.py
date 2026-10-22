import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded
from sever.documents import assemble_documents
from sever.field_review import checked_detail_field, verify_document_fields
from sever.native_pipeline import process_video_images
from sever.schema import Evidence, FieldValue
from test_documents import UPD, sheet
from test_native_conflict_pipeline import prepared
from test_native_pipeline import answer


def detail_answer(frames, name, value="Москва, дом 15, стр. 8", status="recognized"):
    field = FieldValue(status=status, raw_value=value if status == "recognized" else None,
        normalized_value=value if status == "recognized" else None,
        evidence=[Evidence(frame_id=frame["frame_id"], text="Грузоотправитель и его адрес", bbox=[.2, .2, .5, .1]) for frame in frames],
        reason="Literal fresh detail readings agree" if status == "recognized" else "Fine print remains different")
    return {"fields": {name: field.model_dump(mode="json")}, "recognition": {"response_reuse": False}}


@pytest.mark.parametrize("value", ["Москва, дом 15, стр. 8", "Москва, дом 15, стр. В", None])
def test_joint_character_change_requires_fresh_detail_confirmation(tmp_path, monkeypatch, value):
    source, frames, _, _ = prepared(tmp_path, monkeypatch, count=2)

    async def reader(frame, *, source_id, page_id, **kwargs):
        result = answer(frame, source_id, page_id, "upd")
        raw = "Москва, дом 16, стр. 8" if frame["frame_id"] == frames[0]["frame_id"] else "Москва, дом 15, стр. 8"
        result["document"].fields["consignor_address"].raw_value = raw
        result["document"].fields["consignor_address"].normalized_value = raw
        result["recognition"]["pixel_proof"] = {"source_sha256": hashlib.sha256(open(frame["path"], "rb").read()).hexdigest(),
            "frame_id": frame["frame_id"], "source_path": frame["path"], "timestamp_s": frame["timestamp_s"],
            "rotation_clockwise_degrees": 0}
        return result

    async def joint(chosen, **kwargs):
        return detail_answer(chosen, "consignor_address", "Москва, дом 15, стр. В")

    calls = []

    async def verifier(chosen, **kwargs):
        calls.append((deepcopy(chosen), kwargs["field_name"]))
        assert "Москва" not in json.dumps(chosen, ensure_ascii=False)
        assert not {"prior_answers", "candidates", "expected_value"} & set(kwargs)
        if value is None:
            raise TimeoutError("Deliberate detail-call timeout")
        return detail_answer(chosen, kwargs["field_name"], value)

    result = asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(), reader=reader,
        conflict_reader=joint, field_verifier=verifier, additional_frames=lambda *args: iter(())))
    page = result["pages"][0]
    assert len(calls) == 1
    assert page["result"]["fields"]["consignor_address"]["raw_value"] == value
    assert page["image_recognition"]["detail_verifications"][0]["status"] == ("failed" if value is None else "resolved")
    if value is None:
        assert page["result"]["fields"]["consignor_address"]["status"] == "conflict"
        assert "consignor_address: detail_verification_failed:TimeoutError" in page["result"]["errors"]
    assert page["image_recognition"]["conflict_reviews"][0]["answer"]["fields"]["consignor_address"]["raw_value"].endswith("В")


def test_detail_verification_rejects_foreign_and_duplicate_source_evidence():
    frames = [{"frame_id": "a", "source_sha256": "1" * 64}, {"frame_id": "b", "source_sha256": "2" * 64}]
    result = detail_answer(frames, "consignor_address")
    result["fields"]["consignor_address"]["evidence"][1]["frame_id"] = "other"
    with pytest.raises(ValueError, match="another source"):
        checked_detail_field(result, "consignor_address", frames)
    result = detail_answer(frames, "consignor_address")
    frames[1]["source_sha256"] = frames[0]["source_sha256"]
    with pytest.raises(ValueError, match="two distinct"):
        checked_detail_field(result, "consignor_address", frames)


def pages_with_dispute(tmp_path):
    pages = [sheet("page-a", 0, "upd", **UPD),
             sheet("page-b", 4, "upd", **{**UPD, "consignor_address": "Москва, дом 15, стр. В"})]
    for index, page in enumerate(pages):
        frame = {"frame_id": "frame-" + str(index), "path": str(tmp_path / f"{index}.png"),
                 "timestamp_s": index + .5, "quality": {"relative_selection_score": 1}}
        page["frames"] = [frame]
        page["result"]["source_id"] = "source"
        page["image_recognition"] = {"views": [{"frame_id": frame["frame_id"], "document": deepcopy(page["result"]),
            "pixel_proof": {"source_sha256": str(index + 1) * 64, "source_path": frame["path"],
                            "frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"],
                            "rotation_clockwise_degrees": 0}}]}
    return pages


@pytest.mark.parametrize("status", ["recognized", "conflict", "unreadable"])
def test_cross_sheet_dispute_is_verified_and_export_rebuild_preserves_decision(tmp_path, status):
    pages = pages_with_dispute(tmp_path)
    originals = deepcopy(pages)
    calls = []

    async def verifier(frames, **kwargs):
        calls.append(frames)
        assert len({frame["source_page_id"] for frame in frames}) == 2
        return detail_answer(frames, kwargs["field_name"], status=status)

    entries = asyncio.run(verify_document_fields(pages, source_id="source", config=AgentConfig(), out=tmp_path,
        trace_path=tmp_path / "trace.jsonl", verifier=verifier))
    result = {"source": {"sha256": "source"}, "pages": pages, "document_field_verifications": entries}
    field = assemble_documents(result)["documents"][0]["fields"]["consignor_address"]
    assert len(calls) == 1 and pages == originals
    assert field["status"] == status
    assert field["raw_value"] == ("Москва, дом 15, стр. 8" if status == "recognized" else None)
    assert len(field["candidates"]) == 2
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    changed = {**result, "source": {"sha256": "another-video"}}
    assert assemble_documents(changed)["documents"][0]["fields"]["consignor_address"]["status"] == "conflict"


def test_blank_or_absent_fields_do_not_start_document_detail_calls(tmp_path):
    pages = [sheet("page-a", 0, "upd", **{**UPD, "consignor_address": None}),
             sheet("page-b", 4, "upd", **{**UPD, "consignor_address": "BLANK"})]

    async def forbidden(*args, **kwargs):
        pytest.fail("A blank/absent field does not require a character reread")

    assert asyncio.run(verify_document_fields(pages, source_id="source", config=AgentConfig(), out=tmp_path,
        trace_path=tmp_path / "trace.jsonl", verifier=forbidden)) == []


def test_document_detail_budget_failure_is_not_turned_into_a_success(tmp_path):
    async def stopped(*args, **kwargs):
        raise BatchBudgetExceeded("bounded failure")

    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(verify_document_fields(pages_with_dispute(tmp_path), source_id="source", config=AgentConfig(),
            out=tmp_path, trace_path=tmp_path / "trace.jsonl", verifier=stopped))


def test_uncovered_cross_sheet_conflict_stays_unresolved(tmp_path):
    pages = pages_with_dispute(tmp_path)
    for index, value in enumerate(["Москва, дом 15, стр. 5", "Москва, дом 15, стр. 6"], 2):
        page = deepcopy(pages[0])
        page["page_id"] = f"page-{index}"
        page["result"]["page_id"] = page["page_id"]
        page["result"]["fields"]["consignor_address"]["raw_value"] = value
        pages.append(page)

    async def forbidden(*args, **kwargs):
        pytest.fail("Four disputed values cannot be covered by a three-frame pass")

    entries = asyncio.run(verify_document_fields(pages, source_id="source", config=AgentConfig(), out=tmp_path,
        trace_path=tmp_path / "trace.jsonl", verifier=forbidden))
    assembly = assemble_documents({"source": {"sha256": "source"}, "pages": pages,
                                   "document_field_verifications": entries})
    field = assembly["documents"][0]["fields"]["consignor_address"]
    assert entries[0]["status"] == "insufficient_source_views"
    assert field["status"] == "conflict" and field["raw_value"] is None


def test_supported_cross_sheet_majority_keeps_existing_behavior(tmp_path):
    pages = pages_with_dispute(tmp_path)
    third = deepcopy(pages[0])
    third["page_id"] = "page-c"
    third["result"]["page_id"] = "page-c"
    pages.append(third)

    async def forbidden(*args, **kwargs):
        pytest.fail("A supported majority is already resolved by the existing policy")

    assert asyncio.run(verify_document_fields(pages, source_id="source", config=AgentConfig(), out=tmp_path,
        trace_path=tmp_path / "trace.jsonl", verifier=forbidden)) == []
    field = assemble_documents({"pages": pages})["documents"][0]["fields"]["consignor_address"]
    assert field["status"] == "recognized" and field["raw_value"] == UPD["consignor_address"]


def test_document_detail_timeout_is_exposed_in_api_job_status(tmp_path, monkeypatch):
    import uuid
    import sever.api as api

    monkeypatch.setattr(api, "ROOT", tmp_path)

    async def processed(*args, **kwargs):
        return {"pages": [{"result": {"errors": []}}], "usage": {},
                "document_field_verifications": [{"status": "failed", "error_type": "TimeoutError"}]}

    monkeypatch.setattr(api, "process_video", processed)
    identifier = str(uuid.uuid4())
    api.execute(identifier, api.JobRequest(source_path=str(tmp_path / "unused.mp4")), AgentConfig())
    job = json.loads((tmp_path / identifier / "job.json").read_text())
    assert job["status"] == "completed_with_errors"
    assert job["pages_with_errors"] == 0 and job["document_fields_with_errors"] == 1
