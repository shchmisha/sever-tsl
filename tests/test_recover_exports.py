"""Audited machine-only ENOSPC export recovery, entirely local fixtures."""
from copy import deepcopy
import errno
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image

import pytest

from sever.schema import FIELDS_BY_TYPE
from sever.report import write_report
from tools import recover_exports as recovery


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return ref(path)


def ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def machine(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    source = tmp_path / "actual-input.mp4"
    source.write_bytes(b"Local fixture; no decoding or network is required")
    source_id = ref(source)["sha256"]
    pages, classifications, readings = [], {}, []
    for index in range(2):
        page_id, frame_id = f"page-{index + 1}", f"frame-{index + 1}"
        frame = out / "video" / f"{frame_id}.jpg"
        frame.parent.mkdir(exist_ok=True)
        Image.new("RGB", (96, 64), (30 + index * 70, 80, 130)).save(frame)
        oriented = out / "ocr-cache" / f"{frame_id}.png"
        oriented.parent.mkdir(exist_ok=True)
        oriented.write_bytes(f"Unique oriented pixels {index}".encode())
        box = [.1, .1, .4, .03]
        region = {"region_index": 0, "text": "000017", "bbox": box}
        evidence = {"frame_id": frame_id, "region_index": 0, "text": region["text"], "bbox": box}
        fields = {name: {"status": "unreadable", "raw_value": None, "normalized_value": None,
                         "confidence": None, "reason": "Unresolved saved machine reading", "evidence": []}
                  for name in FIELDS_BY_TYPE["upd"]}
        classification = {"source_id": source_id, "page_id": page_id, "document_type": "upd",
                          "type_confidence": None, "classification_evidence": [], "fields": deepcopy(fields),
                          "errors": [], "notes": []}
        result = deepcopy(classification)
        result["fields"]["invoice_number"] = {"status": "recognized", "raw_value": "000017",
            "normalized_value": "000017", "confidence": None, "evidence": [evidence], "reason": "Saved source reading"}
        ocr_path = out / "ocr" / f"{page_id}.json"
        observation = {"frame_id": frame_id, "timestamp_s": index + .5, "regions": [region],
            "processing": {"source_path": str(frame), "source_sha256": ref(frame)["sha256"],
                           "ocr_image_path": str(oriented)}}
        write_json(ocr_path, [observation])
        page = {"page_id": page_id, "start_s": index, "end_s": index + 1, "selected_frame_id": frame_id,
                "frames": [{"frame_id": frame_id, "path": str(frame), "timestamp_s": index + .5}],
                "classification": classification, "result": result, "ocr_artifact": str(ocr_path)}
        pages.append(page)
        classifications[page_id] = classification
        decision = {"document_type": "upd", "saved_field_value": "000017"}
        cache_path = out / "vision-cache" / f"{page_id}.json"
        cache = {"status": "success", "source_id": source_id, "page_id": page_id, "model": "offline-model",
                 "base_url": "https://unused.invalid", "input_sha256": f"input-fingerprint-{index}",
                 "decision": decision, "crop": {"frame_id": frame_id, "upload_sha256": f"upload-fingerprint-{index}",
                 "oriented_image_sha256": ref(oriented)["sha256"]}}
        write_json(cache_path, cache)
        readings.append({"source_id": source_id, "page_id": page_id, "frame_id": frame_id,
            "cache_path": str(cache_path), "cache_file_sha256": ref(cache_path)["sha256"],
            "model": cache["model"], "base_url": cache["base_url"], "input_sha256": cache["input_sha256"],
            "decision_sha256": recovery.canonical_sha256(decision), "context_upload_sha256": cache["crop"]["upload_sha256"],
            "ocr_source_image_sha256": ref(frame)["sha256"], "ocr_image_path": str(oriented),
            "current_input_fingerprint_verified_by": "Controlled local cache audit"})
    config = {"sample_interval": .5, "model": "offline-model",
              "field_engine": "selected role/header pixels via offline-model"}
    parent = {"schema_version": 1, "source": ref(source), "configuration": config, "pages": pages,
              "usage": {"requests": 3, "input_tokens": 100, "output_tokens": 20, "actual_cost_rub": .5},
              "elapsed_seconds": 123.4}
    parent_path = out / "history" / "previous" / "results.json"
    parent_ref = write_json(parent_path, parent)
    checkpoint_ref = write_json(out / "partial-results.json", {"source_id": source_id, "pages": pages[:1]})
    inventory = [{key: value for key, value in page.items() if key not in {"classification", "result", "ocr_artifact"}} for page in pages]
    inventory_ref = write_json(out / "selected-pages.json", inventory)
    classifications_ref = write_json(out / "classifications.json", classifications)
    trace_path = out / "trace.jsonl"
    old = {"event": "model_request_started", "time": 1, "source_id": source_id}
    started = {"event": "run_started", "time": 1000, "source_id": source_id, "sample_interval": .5}
    hit = {"event": "vision_cache_hit", "time": 1003, "usage_delta": {"requests": 0}}
    trace_path.write_text("\n".join(json.dumps(row) for row in (old, started, hit)) + "\n")
    sources = []
    for scope, selected, parent_input in (("latest_machine_partial_prefix", pages[:1], checkpoint_ref),
                                        ("preserved_complete_suffix", pages[1:], parent_ref)):
        sources.append({"scope": scope, "source_id": source_id, "parent_results_path": parent_input["path"],
            "parent_results_sha256": parent_input["sha256"], "page_ids": [page["page_id"] for page in selected],
            "pages": [{"page_id": page["page_id"], "selected_frame_id": page["selected_frame_id"], "unchanged": True,
                       "parent_page_sha256": recovery.canonical_sha256(page), "classification_object": page["classification"],
                       "classification_sha256": recovery.canonical_sha256(page["classification"]),
                       "read_frame_ids": [page["selected_frame_id"]], "ocr_artifact": ref(Path(page["ocr_artifact"]))}
                      for page in selected]})
    audit = {"passed": True, "parents_unchanged": True, "truth_used": False, "network_allowed": False,
        "decision_sha256_convention": "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True, separators=(comma,colon))",
        "failures": [], "changes": [], "usage": {"requests": 0}, "sources": sources, "cache_readings": readings,
        "recovery_inputs": {"source": ref(source), "parent_results": parent_ref, "checkpoint": checkpoint_ref,
            "inventory": inventory_ref, "classifications": classifications_ref, "current_trace": ref(trace_path),
            "configuration": config, "configuration_sha256": recovery.canonical_sha256(config),
            "inventory_identity_unchanged": True, "classifications_unchanged": True,
            "checkpoint_identity_matches_parent_prefix": True, "parent_page_count": 2, "checkpoint_page_count": 1,
            "expected_suffix_page_ids": [pages[1]["page_id"]], "trace_run_started_line": 2,
            "paid_requests_since_boundary": 0, "paid_responses_since_boundary": 0,
            "cache_reader_model": "offline-model", "cache_reader_base_url": "https://unused.invalid"}}
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, audit)
    return {"out": out, "audit": audit, "audit_path": audit_path, "parent": parent,
            "parent_path": parent_path, "checkpoint": out / "partial-results.json", "trace": trace_path}


def run(machine, *, check_only=False):
    write_json(machine["audit_path"], machine["audit"])
    return recovery.recover_exports(machine["audit_path"], ref(machine["audit_path"])["sha256"],
                                    machine["out"], check_only=check_only)


def test_machine_prefix_and_audited_suffix_recover_exports_with_explicit_runtime_and_usage(machine):
    preserved = {path: path.read_bytes() for path in (machine["parent_path"], machine["checkpoint"], machine["audit_path"])}
    trace_prefix = machine["trace"].read_bytes()
    original_frames = {Path(frame["path"]): Path(frame["path"]).read_bytes()
                       for page in machine["parent"]["pages"] for frame in page["frames"]}
    result = run(machine)
    assert result["pages"] == machine["parent"]["pages"]
    assert result["usage"] == machine["parent"]["usage"]
    assert result["elapsed_seconds"] == result["recovery"]["parent_elapsed_seconds"] == 123.4
    assert result["recovery"]["replay_duration_seconds"] == 3
    assert result["recovery"]["replay_provider_usage"]["requests"] == 0
    assert result["recovery"]["recovery_elapsed_seconds"] >= 0
    assert "not a fresh cold run" in result["elapsed_semantics"]
    assert all(path.read_bytes() == original for path, original in preserved.items())
    rows = [json.loads(row) for row in machine["trace"].read_text().splitlines()]
    event = rows[-1]
    assert event["event"] == "exports_recovered" and event["source_complete_proven"] is True
    assert event["results_sha256"] == ref(machine["out"] / "results.json")["sha256"]
    assert event["trace_input_byte_length"] == len(trace_prefix)
    assert hashlib.sha256(machine["trace"].read_bytes()[:event["trace_input_byte_length"]]).hexdigest() == event["trace_input_sha256"]
    assert rows[-1]["replay_provider_calls"] == 0 and not any(row["event"] == "run_completed" for row in rows)
    report = (machine["out"] / "report.html").read_text()
    assert "Восстановленный экспорт" in report and "123.4 с исходного полного прогона" in report
    assert '.recover-exports-' not in report and 'href="video/frame-1.jpg"' in report
    # Persistent previews must survive temporary export-stage cleanup; opening
    # the report must never fetch the original full-resolution source inline.
    # Document cards reuse the same previews as thumbnails; every source frame still has exactly one preview file.
    previews = sorted({src for src in re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', report) if not src.endswith("-thumb.jpg")})
    assert len(previews) == len(original_frames)
    for relative in previews:
        assert relative.startswith("report-assets/")
        with Image.open(machine["out"] / relative) as preview:
            assert preview.format == "JPEG" and preview.size == (96, 64)
    assert all(path.read_bytes() == original for path, original in original_frames.items())
    assert not list(machine["out"].glob(".recover-exports-*"))
    assert Path(result["recovery"]["inventory"]["path"]).is_file()
    assert ref(Path(result["recovery"]["inventory"]["path"]))["sha256"] == result["recovery"]["inventory"]["sha256"]


@pytest.mark.parametrize("document_type", ["unknown", "upd"])
def test_native_image_reading_is_visible_for_all_forms_with_original_source_and_literal_text(machine, document_type):
    result = deepcopy(machine["parent"])
    result["pages"] = result["pages"][:1]
    page = result["pages"][0]
    page["result"]["document_type"] = document_type
    if document_type == "unknown":
        page["result"]["fields"] = {}
    else:
        page["result"]["fields"]["invoice_number"]["evidence"] = [
            {"frame_id": page["selected_frame_id"], "region_index": None,
             "text": "000017", "bbox": [0, 0, 1, 1]}]
    page["result"]["errors"] = ["Actual saved model warning"]
    literal = 'CMR\nОтправитель: <script>printed text & quotation</script>'
    extra_frame = machine["out"] / "video" / "ещё один кадр.jpg"
    Image.new("RGB", (96, 64), (80, 30, 130)).save(extra_frame)
    page["image_recognition"] = {"views": [
        {"frame_id": page["selected_frame_id"], "observed_form": "CMR <лицевая сторона>",
         "visible_text": literal, "document_type": document_type,
         "panelproof": {"bbox": [0, 0, 1, 1], "region_index": None}},
        {"frame_id": "additional-source-view", "frame_path": str(extra_frame),
         "observed_form": "Оборотная сторона", "visible_text": "Подпись второй стороны",
         "document_type": "unknown"}]}
    before = deepcopy(result)
    write_report(machine["out"], result)
    report = (machine["out"] / "report.html").read_text()
    visible = report.split("<details>", 1)[0]
    assert result == before
    assert "Чтение изображения моделью" in visible
    assert html.escape(literal) in visible and "<script>" not in report
    assert html.escape("CMR <лицевая сторона>") in visible
    assert "Подпись второй стороны" in visible and "Actual saved model warning" in visible
    assert 'href="video/frame-1.jpg">frame-1</a>' in visible
    assert f'href="{quote("video/ещё один кадр.jpg")}">additional-source-view</a>' in visible
    assert 'src="video/frame-1.jpg"' not in visible and 'src="report-assets/' in visible
    assert "Прочитанные фрагменты" not in visible
    assert "Координаты доказательств относятся к изображению или фрагменту" in visible
    if document_type == "unknown":
        assert "Форма не подтверждена" in visible and "Подтверждённых полей нет" in visible


def test_report_shows_each_model_proposal_even_when_final_field_conflicts(machine):
    result = deepcopy(machine["parent"])
    result["pages"] = result["pages"][:1]
    page = result["pages"][0]
    page["result"]["fields"]["invoice_number"].update(raw_value=None, normalized_value=None, status="conflict")
    page["image_recognition"] = {"views": [
        {"frame_id": page["selected_frame_id"], "observed_form": "УПД", "visible_text": "",
         "document": {"fields": {"invoice_number": {"raw_value": value, "status": "recognized"}}}}
        for value in ["номер <А>", "номер Б & В"]]}
    before = deepcopy(result)
    write_report(machine["out"], result)
    visible = (machine["out"] / "report.html").read_text().split("<details>", 1)[0]
    assert "Ответ модели на этом кадре" in visible
    assert "номер &lt;А&gt;" in visible and "номер Б &amp; В" in visible
    assert "кадры расходятся" in visible and "прочитано" in visible
    assert result == before


def test_failed_native_image_reading_preserves_unknown_and_error_without_ocr_fallback(machine):
    result = deepcopy(machine["parent"])
    result["pages"] = result["pages"][:1]
    page = result["pages"][0]
    page["result"].update(document_type="unknown", fields={}, errors=["image_reading_failed"],
        classification_evidence=[{"frame_id": "frame-1", "text": "Legacy OCR quote"}])
    page["image_recognition"] = {"views": []}
    write_report(machine["out"], result)
    visible = (machine["out"] / "report.html").read_text().split("<details>", 1)[0]
    assert "Форма не подтверждена" in visible and "image_reading_failed" in visible
    assert "В результате нет сохранённого чтения изображения" in visible
    assert "Legacy OCR quote" not in visible
    del page["image_recognition"]
    write_report(machine["out"], result)
    visible = (machine["out"] / "report.html").read_text().split("<details>", 1)[0]
    assert "Прочитанные фрагменты" in visible and "Legacy OCR quote" in visible
    assert "Чтение изображения моделью" not in visible


def test_processing_report_shows_saved_readings_and_partial_links_then_restores_final_report(machine):
    result = deepcopy(machine["parent"])
    result["pages"] = result["pages"][:1]
    page = result["pages"][0]
    saved_text = page["result"]["fields"]["invoice_number"]["evidence"][0]["text"]
    page["image_recognition"] = {"views": [{"frame_id": page["selected_frame_id"],
        "observed_form": page["result"]["document_type"], "visible_text": saved_text}]}
    before = deepcopy(result)
    original_frame = Path(page["frames"][0]["path"])
    original_bytes = original_frame.read_bytes()
    write_report(machine["out"], result)
    final_html = (machine["out"] / "report.html").read_text()
    assert 'href="results.json"' in final_html and 'href="fields.csv"' in final_html
    assert 'http-equiv="refresh"' not in final_html and 'href="partial-results.json"' not in final_html

    result["status"] = "processing"
    write_report(machine["out"], result)
    processing_html = (machine["out"] / "report.html").read_text()
    visible = processing_html.split("<details>", 1)[0]
    assert "Север ТСЛ — обработка продолжается" in visible
    assert "Это промежуточный результат" in visible
    assert "значения ещё могут уточняться" in visible
    assert 'href="partial-results.json"' in visible and 'href="trace.jsonl"' in visible
    assert 'href="results.json"' not in processing_html and 'href="fields.csv"' not in processing_html
    assert '>CSV<' not in processing_html
    assert '<meta http-equiv="refresh" content="10">' in processing_html
    assert "Чтение изображения моделью" in visible and html.escape(saved_text) in visible
    assert 'href="video/frame-1.jpg">frame-1</a>' in visible
    assert original_frame.read_bytes() == original_bytes
    assert result == {**before, "status": "processing"}

    del result["status"]
    write_report(machine["out"], result)
    assert (machine["out"] / "report.html").read_text() == final_html
    assert result == before and original_frame.read_bytes() == original_bytes


def test_failed_report_keeps_partial_results_without_csv_or_automatic_refresh(machine):
    result = deepcopy(machine["parent"])
    result["status"] = "failed"
    result["pages"][0]["result"]["errors"] = ["Synthetic stopped-job warning"]
    before = deepcopy(result)
    write_report(machine["out"], result)
    report = (machine["out"] / "report.html").read_text()
    visible = report.split("<details>", 1)[0]
    assert "Обработка остановлена" in visible
    assert "Это промежуточный результат" in visible and "Итоговый результат не сформирован" in visible
    assert "Synthetic stopped-job warning" in visible
    assert 'href="partial-results.json"' in report and 'href="trace.jsonl"' in report
    assert 'href="results.json"' not in report and 'href="fields.csv"' not in report
    assert '>CSV<' not in report and 'http-equiv="refresh"' not in report
    assert "Обработка продолжается" not in report
    assert result == before


def test_check_only_does_not_modify_checkpoint_trace_or_export_inventory(machine):
    before = {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()}
    result = run(machine, check_only=True)
    assert result == {"checks_passed": True, "page_count": 2, "exports_written": False,
                      "prefix_page_count": 1, "suffix_page_count": 1}
    assert {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()} == before


def test_declared_default_separator_decision_hash_is_checked_without_alternative_guess(machine):
    machine["audit"]["decision_sha256_convention"] = "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True), default separators"
    for reading in machine["audit"]["cache_readings"]:
        cache = json.loads(Path(reading["cache_path"]).read_text())
        reading["decision_sha256"] = hashlib.sha256(json.dumps(
            cache["decision"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert run(machine, check_only=True)["checks_passed"] is True
    reading["decision_sha256"] = recovery.canonical_sha256(cache["decision"])
    with pytest.raises(recovery.RecoveryRefused, match="Cache input/decision proof differs"):
        run(machine, check_only=True)


def test_source_artifact_link_relocation_preserves_all_machine_fields_and_input_page_digest(machine, tmp_path):
    suffix = machine["audit"]["sources"][1]["pages"][0]
    original = Path(suffix["ocr_artifact"]["path"])
    complete = tmp_path / "audit-complete-ocr.json"
    complete.write_bytes(original.read_bytes())
    suffix["ocr_artifact"] = ref(complete)
    result = run(machine)
    page = result["pages"][1]
    assert page["ocr_artifact"] == str(complete)
    assert page["result"] == machine["parent"]["pages"][1]["result"]
    assert result["pages"][0] == machine["parent"]["pages"][0]
    relocation = result["recovery"]["ocr_artifact_relocations"][0]
    assert relocation["original_page_sha256"] == recovery.canonical_sha256(machine["parent"]["pages"][1])
    assert original.read_bytes() == complete.read_bytes()


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(passed=False), "audit did not pass"),
    (lambda d: d.update(truth_used=True), "independent local"),
    (lambda d: d.pop("decision_sha256_convention"), "Missing or unsupported raw decision"),
    (lambda d: d["sources"][1]["pages"][0].update(unchanged=False), "unchanged proof"),
    (lambda d: d["sources"][1]["pages"][0].update(parent_page_sha256="wrong"), "unchanged proof"),
    (lambda d: d["recovery_inputs"].update(configuration_sha256="wrong"), "Configuration/model differs"),
    (lambda d: d["recovery_inputs"].update(expected_suffix_page_ids=["invented-page"]), "Suffix page order"),
    (lambda d: d["cache_readings"][1].update(input_sha256="wrong"), "Cache input/decision proof"),
    (lambda d: d["cache_readings"][1].update(model="other-model"), "Cache model/endpoint"),
])
def test_failed_proof_refuses_before_any_current_export_mutation(machine, mutation, match):
    before = {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()}
    mutation(machine["audit"])
    with pytest.raises(recovery.RecoveryRefused, match=match):
        run(machine)
    assert {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()} == before


def test_changed_checkpoint_bytes_cannot_be_silently_replaced_by_archived_prefix(machine):
    machine["checkpoint"].write_text("{}")
    with pytest.raises(recovery.RecoveryRefused, match="File SHA-256 mismatch"):
        run(machine)
    assert machine["checkpoint"].read_text() == "{}"
    assert not (machine["out"] / "results.json").exists()


def test_provider_event_in_latest_replay_is_not_confused_with_old_run_events(machine):
    with machine["trace"].open("a") as stream:
        stream.write(json.dumps({"event": "model_request_started", "time": 1004}) + "\n")
    machine["audit"]["recovery_inputs"]["current_trace"] = ref(machine["trace"])
    with pytest.raises(recovery.RecoveryRefused, match="Replay trace contains provider"):
        run(machine)
    assert not (machine["out"] / "results.json").exists()


def test_evidence_from_missing_source_frame_is_rejected_even_if_page_digest_is_updated(machine):
    parent = machine["parent"]
    parent["pages"][1]["result"]["fields"]["invoice_number"]["evidence"][0]["frame_id"] = "wrong-frame"
    parent_ref = write_json(machine["parent_path"], parent)
    machine["audit"]["recovery_inputs"]["parent_results"] = parent_ref
    suffix = machine["audit"]["sources"][1]
    suffix["parent_results_sha256"] = parent_ref["sha256"]
    suffix["pages"][0]["parent_page_sha256"] = recovery.canonical_sha256(parent["pages"][1])
    with pytest.raises(recovery.RecoveryRefused, match="Evidence references a frame"):
        run(machine)


def test_enospc_during_staging_keeps_all_machine_inputs_and_no_completed_result(machine, monkeypatch):
    before = {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()}

    def failed_export(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(recovery, "export_csv", failed_export)
    with pytest.raises(OSError) as exc:
        run(machine)
    assert exc.value.errno == errno.ENOSPC
    assert {path: path.read_bytes() for path in machine["out"].rglob("*") if path.is_file()} == before
    assert not list(machine["out"].glob(".recover-exports-*"))


def test_enospc_on_final_promotion_archives_partial_exports_and_never_marks_completed(machine, monkeypatch):
    checkpoint = machine["checkpoint"].read_bytes()
    trace = machine["trace"].read_bytes()
    real_replace = Path.replace

    def fail_final(self, target):
        if self.name == "results.json" and self.parent.name.startswith(".recover-exports-"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_final)
    with pytest.raises(OSError):
        run(machine)
    assert not (machine["out"] / "results.json").exists()
    assert machine["checkpoint"].read_bytes() == checkpoint and machine["trace"].read_bytes() == trace
    assert list((machine["out"] / "history").glob("before-export-recovery-*/selected-pages.json"))
    assert list((machine["out"] / "history").glob("failed-export-recovery-*/fields.csv"))
