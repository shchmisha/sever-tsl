import asyncio
from copy import deepcopy
import hashlib
import json

from PIL import Image, ImageDraw
import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded
from sever.native_pipeline import process_video_images, _redundant_pixels, _stop_reason
from sever.schema import DocumentResult, Evidence, FieldValue, FIELDS_BY_TYPE


def setup_pages(tmp_path, monkeypatch, count=3):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"independent-source-identity")
    frames = []
    for n in range(count):
        path = tmp_path / f"frame-{n}.png"
        image = Image.new("RGB", (400, 600), "white")
        draw = ImageDraw.Draw(image)
        for row in range(20):
            draw.line((20, 20+row*20, 370, 20+row*20), fill="black", width=2)
        image.save(path)
        frames.append({"frame_id": f"frame-{n}", "path": str(path), "timestamp_s": .5+n*.5,
                       "quality": {"relative_selection_score": 1-n*.01}})
    pages = [{"page_id": "page-1", "start_s": 0, "end_s": 2,
              "selected_frame_id": frames[0]["frame_id"], "frames": frames}]
    monkeypatch.setattr("sever.video.analyze_video", lambda *a, **kw: deepcopy(pages))
    # If the new path accidentally enters either old gate, the test fails.
    def forbidden(*a, **kw):
        raise AssertionError("OCR and separate type classification must not run")
    monkeypatch.setattr("sever.ocr.read_image", forbidden)
    monkeypatch.setattr("sever.classifier.classify_pages", forbidden)
    return source, frames


def answer(frame, source_id, page_id, kind, label="УПД"):
    evidence = Evidence(frame_id=frame["frame_id"], text=label, bbox=[0, 0, 1, 1])
    values = {name: "15.02.2020" if name.endswith("date") else "0042" if name.endswith("number")
              else "example-"+name for name in FIELDS_BY_TYPE[kind]}
    fields = {name: FieldValue(raw_value=values[name], normalized_value=values[name],
                  status="recognized", evidence=[Evidence(frame_id=frame["frame_id"],
                  text=name+" "+values[name], bbox=[0, 0, 1, 1])]) for name in FIELDS_BY_TYPE[kind]}
    doc = DocumentResult(source_id=source_id, page_id=page_id, document_type=kind,
                         classification_evidence=[evidence], fields=fields)
    return {"document": doc, "recognition": {"observed_form": label,
             "model_document_type": kind,
             "visible_text": label+" — visible source transcription", "origin": "model_source_pixels"},
            "panels": [{"panel_id": "context", "bbox": [0, 0, 1, 1]}]}


def test_unknown_first_image_does_not_block_direct_reading_of_later_frames(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        return answer(frame, source_id, page_id, "unknown" if len(seen)==1 else "upd")
    config = AgentConfig(document_reader="native_image", verify_inn=False)
    out = tmp_path / "run"
    result = asyncio.run(process_video_images(source, out, config=config, reader=reader,
                                              additional_frames=lambda *a: iter(())))
    assert seen == [f["frame_id"] for f in frames]
    page = result["pages"][0]
    assert page["result"]["document_type"] == "upd"
    assert all(f["status"] == "recognized" for f in page["result"]["fields"].values())
    assert page["image_recognition"]["separate_classifier_used"] is False
    assert len(page["image_recognition"]["views"]) == 3
    assert result["configuration"]["local_ocr_used"] is False
    assert (out / "fields.csv").is_file() and (out / "report.html").is_file()
    assert json.loads((out / "results.json").read_text())["pages"] == result["pages"]
    events = [json.loads(l)["event"] for l in (out/"trace.jsonl").read_text().splitlines()]
    assert events.count("image_view_attempt") == 3
    assert not any(e in {"ocr_read", "classifier_validation"} for e in events)


def test_non_target_form_is_transcribed_and_exported_without_reading_all_frames(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, "unknown", label="CMR")
        result["recognition"]["form_scope"] = "out_of_scope"
        return result
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    assert len(seen) == 2
    assert page["result"]["document_type"] == "unknown"
    assert page["image_recognition"]["stop_reason"] == "out_of_scope_form_confirmed"
    assert "CMR" in (tmp_path/"run/report.html").read_text()
    assert "visible source transcription" in (tmp_path/"run/report.html").read_text()


def test_additional_source_views_are_read_without_a_per_leaf_count_limit(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=1)
    extra = []
    for n in range(5):
        path = tmp_path / f"extra-{n}.png"
        with Image.open(frames[0]["path"]) as image:
            # A later source view exposes changed local pixels; matching model
            # replies on earlier views must not impose a numerical frame cap.
            ImageDraw.Draw(image).point((30+n*3, 35), fill="black")
            image.save(path)
        extra.append({**frames[0], "path": str(path), "frame_id": f"extra-{n}",
                      "timestamp_s": .75+n*.1})
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, "upd")
        if len(seen) < 5:
            doc = result["document"]
            doc.fields["consignor_address"] = FieldValue(status="unreadable", reason="Illegible source")
        return result
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra)))
    assert len(seen) == 6
    assert result["pages"][0]["result"]["fields"]["consignor_address"]["status"] == "recognized"
    assert result["pages"][0]["image_recognition"]["frame_count_limit"] is None


def test_model_budget_failure_preserves_current_page_checkpoint(tmp_path, monkeypatch):
    source, _ = setup_pages(tmp_path, monkeypatch)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        if len(seen)==2:
            raise BatchBudgetExceeded("Shared budget exhausted")
        return answer(frame, source_id, page_id, "upd")
    out = tmp_path/"run"
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
                                         additional_frames=lambda *a: iter(())))
    saved = json.loads((out/"partial-results.json").read_text())
    assert len(saved["pages"][0]["image_recognition"]["views"]) == 1
    assert saved["pages"][0]["image_recognition"]["complete"] is False
    assert not (out/"results.json").exists()
    assert saved["status"] == "failed"
    assert "Обработка остановлена" in (out/"report.html").read_text()
    assert "visible source transcription" in (out/"report.html").read_text()


def test_three_failed_pages_publish_terminal_checkpoint_and_report_with_all_errors(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=4)
    pages = [{"page_id": f"page-{n+1}", "start_s": n, "end_s": n+1,
              "selected_frame_id": frame["frame_id"], "frames": [frame]}
             for n, frame in enumerate(frames)]
    monkeypatch.setattr("sever.video.analyze_video", lambda *a, **kw: deepcopy(pages))
    seen = []
    async def failed_reader(frame, **kwargs):
        seen.append(frame["frame_id"])
        raise TimeoutError("Offline model fixture failed")
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Three consecutive image readings failed"):
        asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=failed_reader,
                                         additional_frames=lambda *a: iter(())))
    checkpoint = json.loads((out / "partial-results.json").read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["source"] == {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert checkpoint["source_id"] == checkpoint["source"]["sha256"]
    assert checkpoint["elapsed_seconds"] >= 0
    assert [page["page_id"] for page in checkpoint["pages"]] == ["page-1", "page-2", "page-3"]
    assert seen == [frame["frame_id"] for frame in frames[:3]]
    report = (out / "report.html").read_text()
    assert "Обработка остановлена" in report and "Итоговый результат не сформирован" in report
    assert 'http-equiv="refresh"' not in report
    assert 'href="results.json"' not in report and 'href="fields.csv"' not in report
    assert 'href="partial-results.json"' in report and report.count("<article>") == 3
    for n, page in enumerate(checkpoint["pages"]):
        error = f"native_image: image_reading_failed:frame-{n}:TimeoutError"
        assert error in page["result"]["errors"] and error in report
        assert page["image_recognition"]["views"] == []
        assert page["image_recognition"]["attempts"][0]["status"] == "failed"
        assert (out / "image-readings" / f"page-{n+1}.json").is_file()
    assert not (out / "results.json").exists() and not (out / "fields.csv").exists()
    events = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "batch_stopped"
    assert events[-1]["reason"] == "three_consecutive_image_reading_failures"


def test_completed_page_processing_checkpoint_preserves_source_status_and_elapsed(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    pages = [{"page_id": "page-1", "start_s": 0, "end_s": 1,
              "selected_frame_id": frames[0]["frame_id"], "frames": frames[:2]},
             {"page_id": "page-2", "start_s": 1, "end_s": 2,
              "selected_frame_id": frames[2]["frame_id"], "frames": frames[2:]}]
    monkeypatch.setattr("sever.video.analyze_video", lambda *a, **kw: deepcopy(pages))
    out, checkpoints = tmp_path / "run", []
    async def reader(frame, *, source_id, page_id, **kwargs):
        if page_id == "page-2":
            checkpoint = json.loads((out / "partial-results.json").read_text())
            checkpoints.append(checkpoint)
            assert checkpoint["status"] == "processing"
            assert checkpoint["source"] == {"path": str(source), "sha256": source_id}
            assert checkpoint["source_id"] == source_id and checkpoint["elapsed_seconds"] >= 0
            assert len(checkpoint["pages"]) == 1
            assert checkpoint["pages"][0]["image_recognition"]["stop_reason"] == "all_required_fields_confirmed"
            report = (out / "report.html").read_text()
            assert 'http-equiv="refresh"' in report and "обработка продолжается" in report
            assert str(source) in report and source_id in report
        return answer(frame, source_id, page_id, "upd")
    result = asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    assert len(checkpoints) == 1 and len(result["pages"]) == 2
    assert 'http-equiv="refresh"' not in (out / "report.html").read_text()


def test_failed_page_report_storage_error_is_traced_without_hiding_terminal_checkpoint(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    pages = [{"page_id": f"page-{n+1}", "start_s": n, "end_s": n+1,
              "selected_frame_id": frame["frame_id"], "frames": [frame]}
             for n, frame in enumerate(frames)]
    monkeypatch.setattr("sever.video.analyze_video", lambda *a, **kw: deepcopy(pages))
    def report_unavailable(*args, **kwargs):
        raise OSError("Offline report storage fixture failed")
    monkeypatch.setattr("sever.report.write_report", report_unavailable)
    async def failed_reader(*args, **kwargs):
        raise TimeoutError("Offline model fixture failed")
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Three consecutive image readings failed"):
        asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=failed_reader,
                                         additional_frames=lambda *a: iter(())))
    checkpoint = json.loads((out / "partial-results.json").read_text())
    assert checkpoint["status"] == "failed" and len(checkpoint["pages"]) == 3
    events = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["event"] == "progress_report_failed"]
    assert [event["page_id"] for event in failures] == ["page-1", "page-2", "page-3"]
    assert all(event["error_type"] == "OSError" for event in failures)
    assert events[-1]["event"] == "batch_stopped"


def test_foreign_frame_evidence_cannot_promote_fields(tmp_path, monkeypatch):
    source, _ = setup_pages(tmp_path, monkeypatch, count=1)
    async def reader(frame, *, source_id, page_id, **kw):
        result = answer(frame, source_id, page_id, "upd")
        result["document"].fields["invoice_number"].evidence[0].frame_id = "foreign-frame"
        return result
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    assert page["result"]["document_type"] == "unknown"
    assert page["image_recognition"]["views"] == []
    assert page["image_recognition"]["attempts"][0]["error_type"] == "ValueError"


def test_successful_read_does_not_hide_failed_corroborating_view(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=2)
    async def reader(frame, *, source_id, page_id, **kw):
        if frame["frame_id"] == frames[1]["frame_id"]:
            raise TimeoutError("Unavailable model response")
        return answer(frame, source_id, page_id, "upd")
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    assert len(page["image_recognition"]["views"]) == 1
    assert page["result"]["document_type"] == "unknown"
    assert page["result"]["errors"] == [
        "native_image: image_reading_failed:frame-1:TimeoutError"
    ]


def test_invalid_supported_readings_do_not_stop_later_image_reading(tmp_path, monkeypatch):
    source, _ = setup_pages(tmp_path, monkeypatch, count=4)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, "unknown" if len(seen) <= 2 else "upd")
        if len(seen) <= 2:
            result["recognition"]["model_document_type"] = "upd"
            result["document"].errors.append("Own heading was not grounded")
        return result
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    assert len(seen) == 4
    assert result["pages"][0]["result"]["document_type"] == "upd"


def test_ungrounded_competing_type_does_not_stop_later_grounded_views(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=7)
    # These are distinct actual source pixels, so redundancy cannot hide reads.
    for index, frame in enumerate(frames):
        with Image.open(frame["path"]) as image:
            image.putpixel((30, 35), (index, 0, 0))
            image.save(frame["path"])
    seen = []
    missing_heading = "Model document type has no supplied-panel heading evidence"

    async def reader(frame, *, source_id, page_id, **kw):
        index = len(seen)
        seen.append(frame["frame_id"])
        result = answer(frame, source_id, page_id, "ttn" if index == 4 else "upd")
        if index < 4:
            result["document"].fields["consignor_address"] = FieldValue(status="unreadable")
        if index == 4:
            result["document"].classification_evidence = []
            result["document"].errors = [missing_heading]
        return result

    out = tmp_path / "run"
    result = asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    recognition = page["image_recognition"]
    assert seen == [frame["frame_id"] for frame in frames]
    assert recognition["stop_reason"] == "all_required_fields_confirmed"
    assert page["result"]["document_type"] == "upd"
    assert all(field["status"] == "recognized" for field in page["result"]["fields"].values())
    assert page["result"]["errors"] == []
    assert {"error": missing_heading, "recovered": True} in recognition["error_history"]
    assert [view["document_type"] for view in recognition["views"]] == ["upd"] * 4 + ["ttn", "upd", "upd"]
    assert recognition["views"][4]["document"]["errors"] == [missing_heading]
    assert recognition["views"][4]["document"]["fields"]["consignor"]["raw_value"] == "example-consignor"
    assert recognition["fusion"]["readings"][4] == recognition["views"][4]["document"]
    assert missing_heading in (out / "report.html").read_text()


def test_grounded_type_disagreement_still_stops_conservatively(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    seen = []

    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        return answer(frame, source_id, page_id, "upd" if len(seen) == 1 else "ttn")

    result = asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(),
        reader=reader, additional_frames=lambda *a: iter(())))
    page = result["pages"][0]
    assert seen == [frame["frame_id"] for frame in frames[:2]]
    assert page["result"]["document_type"] == "unknown" and page["result"]["fields"] == {}
    assert page["image_recognition"]["stop_reason"] == "disagreeing_document_readings"
    assert [view["document_type"] for view in page["image_recognition"]["views"]] == ["upd", "ttn"]
    assert page["result"]["errors"] == ["image_fusion:conflicting_document_types"]


def unresolved_answer(frame, source_id, page_id, proof=None):
    result = answer(frame, source_id, page_id, "upd")
    result["document"].fields["consignor_address"] = FieldValue(status="unreadable",
        reason="The model cannot read this fragment", evidence=[] if proof is None else
        [Evidence(frame_id=frame["frame_id"], text=proof, bbox=[0, 0, 1, 1])])
    return result


def test_redundant_candidates_are_enumerated_without_becoming_new_model_evidence(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    discovered, seen = [], []
    def extra(*args):
        for n in range(17):
            frame = {**frames[0], "frame_id": f"extra-{n}", "timestamp_s": 2+n*.1}
            discovered.append(frame["frame_id"])
            yield frame
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        return unresolved_answer(frame, source_id, page_id)
    out = tmp_path / "run"
    result = asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
                                              additional_frames=extra))
    page = result["pages"][0]
    assert len(discovered) == 17 and len(seen) == 3
    assert page["result"]["fields"]["consignor_address"]["status"] == "unreadable"
    recognition = page["image_recognition"]
    assert recognition["stop_reason"] == "source_views_exhausted_after_redundant_candidates"
    assert recognition["frame_count_limit"] is None and len(recognition["views"]) == 3
    skipped = recognition["redundancy_saturation"]["skipped_candidates"]
    assert len(skipped) == 17
    assert skipped[0]["supporting_read_frame_ids"] == seen
    assert skipped[0]["unresolved_fields"] == ["consignor_address"]
    assert skipped[0]["new_model_evidence"] is False
    assert all(comparison["maximum_channel_difference"] == 0
               and comparison["candidate_file_sha256"]
               for comparison in skipped[0]["pixel_comparisons"])
    field_evidence_ids = {e["frame_id"] for field in page["result"]["fields"].values()
                          for e in field["evidence"]}
    assert not field_evidence_ids.intersection(discovered)
    trace = [json.loads(line) for line in (out/"trace.jsonl").read_text().splitlines()]
    assert len([event for event in trace if event["event"] == "image_view_attempt"
                and event.get("status") == "skipped"]) == 17


def test_changed_single_native_pixel_is_read_and_later_field_can_be_confirmed(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    path = tmp_path / "tiny-new-detail.png"
    with Image.open(frames[0]["path"]) as image:
        ImageDraw.Draw(image).point((32, 37), fill="black")
        image.save(path)
    extra = [{**frames[0], "frame_id": f"new-{n}", "path": str(path),
              "timestamp_s": 2+n*.1} for n in range(2)]
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        if frame["frame_id"].startswith("new-"):
            return answer(frame, source_id, page_id, "upd")
        return unresolved_answer(frame, source_id, page_id)
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra)))
    assert len(seen) == 5
    assert result["pages"][0]["result"]["fields"]["consignor_address"]["status"] == "recognized"
    assert result["pages"][0]["image_recognition"]["redundancy_saturation"]["skipped_candidates"] == []


def novel_views(tmp_path, frames, count=18):
    extra = []
    for n in range(count):
        path = tmp_path / f"novel-{n}.png"
        with Image.open(frames[0]["path"]) as image:
            ImageDraw.Draw(image).point((30+n*3, 37), fill="black")
            image.save(path)
        extra.append({**frames[0], "frame_id": f"novel-{n}", "path": str(path),
                      "timestamp_s": 2+n*.1})
    return extra


def test_unreadable_leaf_stops_after_consecutive_views_without_progress(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    extra = novel_views(tmp_path, frames)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        return unresolved_answer(frame, source_id, page_id)
    out = tmp_path / "run"
    result = asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra)))
    # Read 1 proposes values, read 2 confirms the type and fields; then three
    # reads add nothing, so the remaining 16 changed views stay unread.
    assert seen == ["frame-0", "frame-1", "frame-2", "novel-0", "novel-1"]
    page = result["pages"][0]
    assert page["result"]["fields"]["consignor_address"]["status"] == "unreadable"
    recognition = page["image_recognition"]
    assert recognition["stop_reason"] == "source_views_unread_after_no_progress"
    assert recognition["no_progress_stop"] == {"policy": result["configuration"]["no_progress_stop"],
                                               "triggered": True, "views_without_progress": 3}
    assert recognition["frame_count_limit"] is None


def test_disabled_no_progress_stop_keeps_reading_every_changed_view(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    extra = novel_views(tmp_path, frames)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        return unresolved_answer(frame, source_id, page_id)
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra), no_progress_views=None))
    assert len(seen) == 21
    assert result["pages"][0]["result"]["fields"]["consignor_address"]["status"] == "unreadable"
    assert result["pages"][0]["image_recognition"]["no_progress_stop"]["triggered"] is False


def test_failed_reads_count_as_views_without_progress(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    extra = novel_views(tmp_path, frames, count=6)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        if frame["frame_id"].startswith("novel-"):
            raise TimeoutError("No response")
        return unresolved_answer(frame, source_id, page_id)
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra)))
    page = result["pages"][0]
    # frame-2 adds nothing, then two failed reads reach the limit of three.
    assert seen == ["frame-0", "frame-1", "frame-2", "novel-0", "novel-1"]
    assert page["image_recognition"]["stop_reason"] == "source_views_unread_after_no_progress"
    assert [a["status"] for a in page["image_recognition"]["attempts"]] == ["read"] * 3 + ["failed"] * 2
    assert any(error.startswith("native_image: image_reading_failed:novel-0") for error in page["result"]["errors"])


def test_new_value_proposal_resets_the_no_progress_count(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    extra = novel_views(tmp_path, frames, count=8)
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        result = unresolved_answer(frame, source_id, page_id)
        if frame["frame_id"] == "novel-1":
            result["document"].fields["consignor_address"] = FieldValue(
                status="recognized", raw_value="New readable address", normalized_value="New readable address",
                evidence=[Evidence(frame_id=frame["frame_id"], text="Адрес: New readable address", bbox=[0, 0, 1, 1])])
        return result
    asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                     additional_frames=lambda *a: iter(extra)))
    # novel-1 is the first view with a new value, so three more reads follow it.
    assert seen == ["frame-0", "frame-1", "frame-2", "novel-0", "novel-1", "novel-2", "novel-3", "novel-4"]


def test_reworded_quote_of_unchanged_unreadable_field_keeps_the_same_signature():
    from sever.native_pipeline import _reading_signature

    def reading(quote, value="0042"):
        fields = {name: FieldValue(raw_value=value if name == "invoice_number" else "example-"+name,
                                   status="recognized",
                                   evidence=[Evidence(frame_id="f", text=name, bbox=[0, 0, 1, 1])])
                  for name in FIELDS_BY_TYPE["upd"]}
        fields["consignor_address"] = FieldValue(status="unreadable", reason="blurred",
            evidence=[Evidence(frame_id="f", text=quote, bbox=[0, 0, 1, 1])])
        return DocumentResult(source_id="s", page_id="p", document_type="upd", fields=fields)
    document = DocumentResult(source_id="s", page_id="p", document_type="unknown")
    assert _reading_signature(reading("Адрес: …"), document) == _reading_signature(reading("Адрес грузо…"), document)
    assert _reading_signature(reading("Адрес: …"), document) != _reading_signature(reading("Адрес: …", "0099"), document)


@pytest.mark.parametrize("change", ["new_quote", "new_raw", "model_error", "provider_failure"])
def test_new_proposal_conflict_or_failed_read_resets_saturation(tmp_path, monkeypatch, change):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    path = tmp_path / "changed.png"
    with Image.open(frames[0]["path"]) as image:
        ImageDraw.Draw(image).point((32, 37), fill="black")
        image.save(path)
    extra = [{**frames[0], "frame_id": f"changed-{n}", "path": str(path),
              "timestamp_s": 2+n*.1} for n in range(3)]
    seen = []
    async def reader(frame, *, source_id, page_id, **kw):
        seen.append(frame["frame_id"])
        if frame["frame_id"] == "changed-0" and change == "provider_failure":
            raise TimeoutError("No response")
        result = unresolved_answer(frame, source_id, page_id)
        if frame["frame_id"].startswith("changed-"):
            if change == "new_quote":
                result = unresolved_answer(frame, source_id, page_id, proof="Fresh partial transcription")
            elif change == "new_raw":
                result["document"].fields["invoice_number"].raw_value = "0099"
                result["document"].fields["invoice_number"].normalized_value = "0099"
            elif change == "model_error" and frame["frame_id"] == "changed-0":
                result["document"].errors.append("Malformed model proposal")
        return result
    result = asyncio.run(process_video_images(source, tmp_path/"run", config=AgentConfig(), reader=reader,
                                              additional_frames=lambda *a: iter(extra), no_progress_views=None))
    assert len(seen) == 6
    assert result["pages"][0]["image_recognition"]["redundancy_saturation"]["skipped_candidates"] == []


def test_uncertain_and_higher_detail_source_images_are_not_redundant(tmp_path):
    def frame(path, label):
        return {"path": str(path), "frame_id": label, "timestamp_s": 1.0}
    old = tmp_path / "old.png"
    image = Image.new("RGB", (80, 100), (200, 200, 200))
    ImageDraw.Draw(image).line((10, 10, 70, 10), fill=(100, 100, 100), width=2)
    image.save(old)
    changed = tmp_path / "sharper.png"
    ImageDraw.Draw(image).line((10, 10, 70, 10), fill=(99, 99, 99), width=2)
    image.save(changed)
    redundant, proof = _redundant_pixels(frame(changed, "new"), frame(old, "old"))
    assert not redundant and proof["decision"] == "changed_local_source_pixels"
    larger = tmp_path / "larger.png"
    image.resize((160, 200)).save(larger)
    redundant, proof = _redundant_pixels(frame(larger, "new"), frame(old, "old"))
    assert not redundant and proof["decision"] == "different_source_geometry"
    redundant, proof = _redundant_pixels(frame(tmp_path/"missing.png", "new"), frame(old, "old"))
    assert not redundant and proof["decision"] == "uncertain_source_pixels"


def test_tiny_field_contrast_improvement_cannot_be_hidden_by_large_flatter_area(tmp_path):
    import numpy as np
    old_pixels = np.full((100, 100, 3), 200, dtype=np.uint8)
    old_pixels[:50:2, :, :] = 100
    old_pixels[75:80, 75:80, :] = 100
    new_pixels = old_pixels.copy()
    new_pixels[:50:2, :, :] = 101
    new_pixels[75:80, 75:80, :] = 99
    old, new = tmp_path/"old.png", tmp_path/"new.png"
    Image.fromarray(old_pixels).save(old)
    Image.fromarray(new_pixels).save(new)
    redundant, proof = _redundant_pixels({"path": str(new), "frame_id": "new"},
                                        {"path": str(old), "frame_id": "old"})
    assert proof["maximum_channel_difference"] == 1
    assert not redundant and proof["decision"] == "changed_local_source_pixels"


@pytest.mark.parametrize("scopes,labels,expected", [
    (["out_of_scope", "out_of_scope"], ["CMR", "СМР"], "out_of_scope_form_confirmed"),
    (["insufficient_evidence", "insufficient_evidence"], ["CMR", "CMR"], None),
    (["out_of_scope", "insufficient_evidence"], ["CMR", "CMR"], None),
    ([None, None], ["CMR", "CMR"], None),
])
def test_unknown_stopping_uses_model_scope_not_literal_heading_match(scopes, labels, expected):
    readings, views = [], []
    for n, (scope, label) in enumerate(zip(scopes, labels)):
        result = answer({"frame_id": f"frame-{n}"}, "source", "page", "unknown", label=label)
        result["recognition"]["form_scope"] = scope
        readings.append(result["document"])
        views.append(result["recognition"])
    assert _stop_reason(readings[-1], readings, views) == expected


@pytest.mark.parametrize("invalid", ["errors", "duplicate", "empty_heading", "empty_transcription"])
def test_out_of_scope_requires_two_successful_distinct_transcribed_source_views(invalid):
    readings, views = [], []
    for n in range(2):
        result = answer({"frame_id": "same" if invalid == "duplicate" else f"frame-{n}"},
                        "source", "page", "unknown", label="A visible form")
        result["recognition"]["form_scope"] = "out_of_scope"
        if n == 1:
            if invalid == "errors":
                result["document"].errors.append("Rejected proposal")
            elif invalid == "empty_heading":
                result["recognition"]["observed_form"] = ""
            elif invalid == "empty_transcription":
                result["recognition"]["visible_text"] = ""
        readings.append(result["document"])
        views.append(result["recognition"])
    assert _stop_reason(readings[-1], readings, views) is None


def test_default_native_path_uses_source_zoom_reader_on_independent_frames(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch)
    seen = []
    async def source_zoom_reader(frame, *, source_id, page_id, **kwargs):
        seen.append(frame["frame_id"])
        return answer(frame, source_id, page_id, "upd")
    async def forbidden_legacy(*args, **kwargs):
        raise AssertionError("The native pipeline bypassed the source-zoom wrapper")
    monkeypatch.setattr("sever.zoom_image_reader.read_frame_zoom", source_zoom_reader)
    monkeypatch.setattr("sever.image_reader.read_frame", forbidden_legacy)
    result = asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(),
                                              additional_frames=lambda *args: iter(())))
    assert seen == [frame["frame_id"] for frame in frames[:2]]
    assert all(field["status"] == "recognized" for field in result["pages"][0]["result"]["fields"].values())


def test_per_view_budget_exhaustion_continues_to_fresh_corroborrating_views(tmp_path, monkeypatch):
    from sever.image_agent import ViewBudgetExceeded
    source, frames = setup_pages(tmp_path, monkeypatch, count=4)
    seen = []
    async def reader(frame, *, source_id, page_id, **kwargs):
        seen.append(frame["frame_id"])
        if len(seen) == 1:
            raise ViewBudgetExceeded("Local single-view tool budget exhausted")
        return answer(frame, source_id, page_id, "upd")
    result = asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(), reader=reader,
        additional_frames=lambda *args: iter(())))
    page = result["pages"][0]
    assert len(seen) == 3
    assert page["result"]["errors"] == []
    assert page["image_recognition"]["stop_reason"] == "all_required_fields_confirmed"
    assert page["image_recognition"]["attempts"][0]["status"] == "failed"
    assert page["image_recognition"]["error_history"] == [{
        "error": "native_image: image_reading_failed:frame-0:ViewBudgetExceeded", "recovered": True}]


def test_lost_view_evidence_stays_an_active_error_after_field_recovery(tmp_path, monkeypatch):
    from sever.image_agent import EvidencePersistenceError
    source, frames = setup_pages(tmp_path, monkeypatch, count=4)
    seen = []
    async def reader(frame, *, source_id, page_id, **kwargs):
        seen.append(frame["frame_id"])
        if len(seen) == 1:
            raise EvidencePersistenceError("Unable to save a received answer")
        return answer(frame, source_id, page_id, "upd")
    result = asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(), reader=reader,
        additional_frames=lambda *args: iter(())))
    page = result["pages"][0]
    error = "native_image: image_reading_failed:frame-0:EvidencePersistenceError"
    assert len(seen) == 3
    assert all(field["status"] == "recognized" for field in page["result"]["fields"].values())
    assert page["result"]["errors"] == [error]
    assert page["image_recognition"]["attempts"][0]["evidence_persistence_failed"] is True
    assert page["image_recognition"]["error_history"] == [{"error": error, "recovered": False}]


def test_uncertain_video_coverage_is_preserved_in_progress_and_final_exports(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=3)
    out = tmp_path / "run"
    (out / "video").mkdir(parents=True)
    (out / "video/video-analysis.json").write_text(json.dumps({"sampling": {
        "discarded_transition_evidence": [{"timestamp_s": 1.25, "reason": "no_stable_brief_view_confirmed"}]}}))
    progress_coverage = []
    async def reader(frame, *, source_id, page_id, **kwargs):
        checkpoint = out / "partial-results.json"
        if checkpoint.exists():
            progress_coverage.append(json.loads(checkpoint.read_text())["video_coverage"])
        return answer(frame, source_id, page_id, "upd")
    result = asyncio.run(process_video_images(source, out, config=AgentConfig(), reader=reader,
        additional_frames=lambda *args: iter(())))
    expected = result["video_coverage"]
    assert expected["requires_review"] is True and expected["uncertain_view_count"] == 1
    assert expected["uncertain_views"] == [{"timestamp_s": 1.25, "reason": "no_stable_brief_view_confirmed"}]
    assert progress_coverage and all(item == expected for item in progress_coverage)
    assert json.loads((out / "partial-results.json").read_text())["video_coverage"] == expected
    assert json.loads((out / "results.json").read_text())["video_coverage"] == expected


def test_default_reader_gets_one_orientation_store_per_leaf(tmp_path, monkeypatch):
    source, frames = setup_pages(tmp_path, monkeypatch, count=4)
    pages = [{"page_id": f"page-{n}", "start_s": n, "end_s": n + 1,
              "selected_frame_id": frames[2*n]["frame_id"], "frames": deepcopy(frames[2*n:2*n+2])}
             for n in range(2)]
    monkeypatch.setattr("sever.video.analyze_video", lambda *a, **kw: deepcopy(pages))
    stores = []
    async def fake_zoom(frame, *, source_id, page_id, leaf_orientation, **kw):
        stores.append((page_id, id(leaf_orientation)))
        leaf_orientation.setdefault("rotation_clockwise", 90)
        return answer(frame, source_id, page_id, "upd")
    async def no_review(*args, **kwargs):
        raise AssertionError("Agreeing readings need no joint review")
    monkeypatch.setattr("sever.zoom_image_reader.read_frame_zoom", fake_zoom)
    monkeypatch.setattr("sever.image_conflicts.read_leaf_conflicts", no_review)
    asyncio.run(process_video_images(source, tmp_path / "run", config=AgentConfig(),
                                     additional_frames=lambda *a: iter(())))
    assert [page for page, _ in stores] == ["page-0", "page-0", "page-1", "page-1"]
    assert stores[0][1] == stores[1][1] and stores[2][1] == stores[3][1]
    assert stores[0][1] != stores[2][1]
