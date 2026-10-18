import csv
import json

from typer.testing import CliRunner

from sever.cli import app
from sever.documents import BLANK_TEXT, DOCUMENT_CSV_COLUMNS, assemble_documents, document_row, export_documents
from sever.report import write_report
from sever.schema import FIELDS_BY_TYPE


def field(value=None, status=None, frames=("f1", "f2")):
    status = status or ("recognized" if value is not None else "unreadable")
    return {"raw_value": value, "normalized_value": value if status == "recognized" else None, "status": status,
            "confidence": None, "evidence": [{"frame_id": frame, "region_index": None, "text": "x", "bbox": None} for frame in frames]
            if status in {"recognized", "not_present"} else []}


def sheet(page_id, start, kind, **values):
    names = FIELDS_BY_TYPE[kind]
    fields = {}
    for name in names:
        value = values.get(name)
        if value == "BLANK":
            fields[name] = field(status="not_present")
        elif value == "CONFLICT":
            fields[name] = field(status="conflict")
        else:
            fields[name] = field(value)
    return {"page_id": page_id, "start_s": float(start), "end_s": float(start) + 4.0, "frames": [],
            "result": {"source_id": "s", "page_id": page_id, "document_type": kind, "type_confidence": None,
                       "classification_evidence": [], "fields": fields, "errors": [], "notes": []}}


UPD = dict(invoice_number="2591", invoice_date="23 августа 2026 г.", consignor='ООО "ФРЭШ-ЭКСПРЕСС"',
           consignor_address="454004, Челябинск", consignee='ООО "УМНЫЙ РИТЕЙЛ"', consignee_address="628422, Сургут")
TTN_FRONT = dict(consignor='ООО "ФРЭШ-ЭКСПРЕСС"', consignee='ООО "УМНЫЙ РИТЕЙЛ"', document_number="2591", document_date="23.08.2026")


def video(*pages):
    return {"source": {"path": "v.mp4", "sha256": "0" * 64}, "pages": list(pages), "elapsed_seconds": 1.0,
            "video_coverage": {}}


def test_repeats_reverse_sides_and_continuations_become_one_document_per_form_and_number():
    result = video(
        sheet("page-0001", 0, "upd", **UPD),
        sheet("page-0002", 4, "ttn", **TTN_FRONT),
        sheet("page-0003", 8, "ttn", document_number="2591"),          # reverse side: only the number
        sheet("page-0004", 12, "upd", **UPD),                           # second copy
        sheet("page-0005", 16, "ttn", **TTN_FRONT),
        sheet("page-0006", 20, "ttn"),                                  # continuation page: nothing readable
        sheet("page-0007", 24, "upd", invoice_number="№ 2591", invoice_date="23 августа 2026 г."),  # sheet 2 heading
        sheet("page-0008", 28, "ttn"),                                  # continuation shown after a УПД sheet
    )
    assembly = assemble_documents(result)
    assert assembly["summary"]["documents"] == 2 and assembly["summary"]["complete_documents"] == 2
    upd, ttn = assembly["documents"]
    assert (upd["document_type"], upd["number"]) == ("upd", "2591")
    assert [s["page_id"] for s in upd["sheets"]] == ["page-0001", "page-0004", "page-0007"]
    assert upd["fields"]["consignor_address"]["raw_value"] == "454004, Челябинск"
    assert upd["fields"]["consignor"]["source_page_ids"] == ["page-0001", "page-0004"]
    assert [(s["page_id"], s["assignment"]) for s in ttn["sheets"]] == [
        ("page-0002", "own_number"), ("page-0003", "own_number"), ("page-0005", "own_number"),
        ("page-0006", "continues_previous_sheet"), ("page-0008", "only_document_of_this_type")]
    assert ttn["fields"]["document_date"]["raw_value"] == "23.08.2026"
    assert assembly["unassigned_sheets"] == [] and assembly["other_forms"] == []


def test_out_of_scope_forms_are_not_documents():
    other = sheet("page-0002", 4, "upd", **UPD)
    other["result"] = {**other["result"], "document_type": "unknown", "fields": {}}
    assembly = assemble_documents(video(sheet("page-0001", 0, "upd", **UPD), other))
    assert assembly["summary"]["documents"] == 1
    assert [item["page_id"] for item in assembly["other_forms"]] == ["page-0002"]


def test_sheet_without_number_and_values_between_other_documents_is_listed_not_invented():
    result = video(sheet("page-0001", 0, "ttn", **TTN_FRONT),
                   sheet("page-0002", 4, "upd", **UPD),
                   sheet("page-0003", 8, "ttn"),
                   sheet("page-0004", 12, "upd", **UPD),
                   sheet("page-0005", 16, "ttn", **{**TTN_FRONT, "document_number": "777"}))
    assembly = assemble_documents(result)
    assert [d["number"] for d in assembly["documents"]] == ["2591", "2591", "777"]
    assert [item["page_id"] for item in assembly["unassigned_sheets"]] == ["page-0003"]


def test_sheet_with_values_but_no_number_keeps_its_values_as_a_separate_document():
    result = video(sheet("page-0001", 0, "ttn", **TTN_FRONT),
                   sheet("page-0002", 4, "upd", **UPD),
                   sheet("page-0003", 8, "ttn", consignor='ООО "Ромашка"'),
                   sheet("page-0004", 12, "upd", **UPD),
                   sheet("page-0005", 16, "ttn", **{**TTN_FRONT, "document_number": "777"}))
    documents = assemble_documents(result)["documents"]
    separate = next(d for d in documents if d["sheets"][0]["page_id"] == "page-0003")
    assert separate["number"] is None and separate["sheets"][0]["assignment"] == "number_not_read"
    assert separate["fields"]["consignor"]["raw_value"] == 'ООО "Ромашка"'
    assert "document_number" in separate["undetermined_fields"]


def test_disagreeing_sheets_keep_the_majority_and_report_an_even_split_as_conflict():
    majority = assemble_documents(video(sheet("page-0001", 0, "ttn", **TTN_FRONT),
                                        sheet("page-0002", 4, "ttn", **TTN_FRONT),
                                        sheet("page-0003", 8, "ttn", **{**TTN_FRONT, "consignee": 'ООО "УМНЫЙ РИТЕЙ"'})))
    consignee = majority["documents"][0]["fields"]["consignee"]
    assert consignee["status"] == "recognized" and consignee["raw_value"] == 'ООО "УМНЫЙ РИТЕЙЛ"'
    assert [c["raw_value"] for c in consignee["candidates"]] == ['ООО "УМНЫЙ РИТЕЙЛ"', 'ООО "УМНЫЙ РИТЕЙ"']

    split = assemble_documents(video(sheet("page-0001", 0, "ttn", **TTN_FRONT),
                                     sheet("page-0002", 4, "ttn", **{**TTN_FRONT, "document_date": "24.08.2026"})))
    record = split["documents"][0]
    assert record["fields"]["document_date"]["status"] == "conflict"
    assert record["fields"]["document_date"]["raw_value"] is None
    assert record["status"] == "incomplete"
    row = document_row(record)
    assert row["Дата"] == "" and "разные значения на листах (23.08.2026 / 24.08.2026)" in row["Не удалось определить"]


def test_formatting_only_differences_still_agree():
    assembly = assemble_documents(video(sheet("page-0001", 0, "ttn", **TTN_FRONT),
                                        sheet("page-0002", 4, "ttn", **{**TTN_FRONT, "consignee": 'ООО «Умный Ритейл»'})))
    assert assembly["documents"][0]["fields"]["consignee"]["candidates"] == []


def test_csv_has_one_row_per_document_with_contract_columns(tmp_path):
    trn = dict(document_date="01.09.2025", document_number="BLANK", consignor="АО Тандер", customer="ООО Савушкин",
               consignee="ООО Магнит")
    assembly = assemble_documents(video(sheet("page-0001", 0, "upd", **UPD), sheet("page-0002", 4, "upd", **UPD),
                                        sheet("page-0003", 8, "trn", **trn)))
    export_documents(tmp_path, assembly)
    with (tmp_path / "documents.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == DOCUMENT_CSV_COLUMNS
    assert [row["Тип документа"] for row in rows] == ["УПД", "ТрН"]
    assert rows[0]["Номер"] == "2591" and rows[0]["Адрес грузополучателя"] == "628422, Сургут"
    assert rows[0]["Листы в видео, с"] == "0.0–4.0; 4.0–8.0"
    assert rows[1]["Номер"] == BLANK_TEXT and rows[1]["Заказчик услуг по перевозке"] == "ООО Савушкин"
    assert rows[1]["Поставщик"] == "" and rows[1]["Не удалось определить"] == ""
    saved = json.loads((tmp_path / "documents.json").read_text(encoding="utf-8"))
    assert saved["summary"]["documents"] == 2


def test_report_leads_with_documents_and_folds_sheets(tmp_path):
    result = video(sheet("page-0001", 0, "upd", **UPD), sheet("page-0002", 4, "ttn", **TTN_FRONT),
                   sheet("page-0003", 8, "ttn", document_number="2591"))
    write_report(tmp_path, result)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert page.index('id="documents"') < page.index('id="sheet-page-0001"')
    assert "Документы в видео: 2" in page
    assert '<h3>№ 2591 от 23 августа 2026 г.</h3>' in page and "Адрес грузоотправителя" in page
    assert 'href="#sheet-page-0003"' in page
    assert "лист документа ТТН № 2591" in page
    assert 'href="documents.csv"' in page


def test_documents_command_rebuilds_exports_from_a_finished_run(tmp_path):
    result = video(sheet("page-0001", 0, "upd", **UPD), sheet("page-0002", 4, "upd", **UPD))
    (tmp_path / "results.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "report.html").write_text("old", encoding="utf-8")
    outcome = CliRunner().invoke(app, ["documents", str(tmp_path)])
    assert outcome.exit_code == 0, outcome.output
    assert json.loads(outcome.output)["documents"] == 1
    assert (tmp_path / "documents.csv").is_file()
    assert 'id="documents"' in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert [path.read_text() for path in tmp_path.glob("history/*/report.html")] == ["old"]
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == result


from sever.evaluation import evaluate_documents  # noqa: E402


def scored_sheet(page_id, start, kind, **values):
    page = sheet(page_id, start, kind, **values)
    page["frames"] = [{"frame_id": f"{page_id}-f", "timestamp_s": start + 2.0, "path": "unused.png"}]
    page["selected_frame_id"] = f"{page_id}-f"
    return page


def label(value):
    return {"value": value, "status": "recognized", "labeled": True}


NOT_VISIBLE = {"value": None, "labeled": False, "manual_visibility": "not_visible_on_this_leaf"}


def truth_sheet(page_id, start, kind, document_id, **labels):
    return {"page_id": page_id, "start_s": float(start), "end_s": float(start) + 4.0, "document_type": kind,
            "document_id": document_id, "fields": {name: labels.get(name, NOT_VISIBLE) for name in FIELDS_BY_TYPE[kind]}}


def truth_video(*pages):
    return {"source_sha256": "0" * 64, "pages": list(pages)}


UPD_LABELS = {name: label(value) for name, value in UPD.items()}
TTN_LABELS = {name: label(value) for name, value in TTN_FRONT.items()}


def test_document_score_joins_repeats_and_reverse_sides():
    truth = truth_video(truth_sheet("t1", 0, "upd", "upd:2591", **UPD_LABELS),
                        truth_sheet("t2", 4, "ttn", "ttn:2591", **TTN_LABELS),
                        truth_sheet("t3", 8, "ttn", "ttn:2591", document_number=label("2591")),
                        truth_sheet("t4", 12, "upd", "upd:2591", **UPD_LABELS))
    result = video(scored_sheet("p1", 0, "upd", **UPD), scored_sheet("p2", 4, "ttn", **TTN_FRONT),
                   scored_sheet("p3", 8, "ttn", document_number="2591"), scored_sheet("p4", 12, "upd", **UPD))
    report = evaluate_documents(result, truth)
    assert (report["truth_documents"], report["output_documents"]) == (2, 2)
    assert report["documents_found"]["correct"] == 2 and report["document_type_accuracy"]["correct"] == 2
    assert report["labeled_fields"] == {"correct": 10, "total": 10, "rate": 1.0}
    assert report["e2e_on_fully_labeled"]["correct"] == 2 and report["extra_output_documents"] == []


def test_misread_number_on_a_repeat_is_an_extra_document_and_absent_fields_are_abstention_checks():
    truth = truth_video(truth_sheet("t1", 0, "ttn", "ttn:2591", document_number=label("2591")),
                        truth_sheet("t2", 4, "ttn", "ttn:2591", document_number=label("2591")))
    result = video(scored_sheet("p1", 0, "ttn", document_number="2591", consignor='ООО "Ромашка"'),
                   scored_sheet("p2", 4, "ttn", document_number="2597"))
    report = evaluate_documents(result, truth)
    assert report["documents_found"]["correct"] == 1
    assert report["labeled_fields"]["correct"] == 1
    assert report["abstention_checks"]["total"] == 3 and report["abstention_checks"]["correct"] == 2
    assert report["abstention_checks"]["failures"][0]["field"] == "consignor"
    assert [(item["number"], item["reason"]) for item in report["extra_output_documents"]] == [
        ("2597", "split_from_matched_document")]
    assert report["details"][0]["split_into_output_documents"] == ["doc-001", "doc-002"]


def test_document_score_requires_explicit_document_ids():
    truth = truth_video({**truth_sheet("t1", 0, "ttn", "x", document_number=label("1")), "document_id": None})
    try:
        evaluate_documents(video(scored_sheet("p1", 0, "ttn", document_number="1")), truth)
    except ValueError as error:
        assert "document_id" in str(error)
    else:
        raise AssertionError("missing document_id must fail")


def test_number_readings_that_differ_only_in_lookalike_letters_are_one_document():
    torg = dict(supplier='ООО "Русское молоко"', consignee='ООО "УМНЫЙ РИТЕЙЛ"', document_date="17.08.2026")
    result = video(sheet("p1", 0, "torg12", document_number="РРБ-128027", **torg),
                   sheet("p2", 4, "torg12", document_number="РРРБ-128027", **torg),
                   sheet("p3", 8, "torg12", document_number="РРРБ-128027", **torg),
                   sheet("p4", 12, "torg12", document_number="PРРВ-128027", **torg),   # Latin P
                   sheet("p5", 16, "torg12", document_number="РРРБ-128028", **torg))   # other digits: other document
    documents = assemble_documents(result)["documents"]
    assert [d["number"] for d in documents] == ["РРРБ-128027", "РРРБ-128028"]
    first = documents[0]
    assert [s["page_id"] for s in first["sheets"]] == ["p1", "p2", "p3", "p4"]
    assert sorted(first["number_readings_joined"]) == sorted(["РРБ-128027", "РРРБ-128027", "PРРВ-128027"])
    assert [c["raw_value"] for c in first["fields"]["document_number"]["candidates"]] == ["РРРБ-128027", "РРБ-128027", "PРРВ-128027"]
    assert {s["assignment"] for s in first["sheets"] if s["page_id"] in {"p1", "p4"}} == {"own_number_variant"}


def test_number_variants_with_different_dates_stay_separate_documents():
    result = video(sheet("p1", 0, "ttn", **{**TTN_FRONT, "document_number": "АБ-100"}),
                   sheet("p2", 4, "ttn", **{**TTN_FRONT, "document_number": "АВ-100", "document_date": "24.08.2026"}))
    assert len(assemble_documents(result)["documents"]) == 2


def test_documents_command_restores_the_previous_report_when_rebuilding_fails(tmp_path, monkeypatch):
    import sever.report

    result = video(sheet("page-0001", 0, "upd", **UPD))
    (tmp_path / "results.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "report.html").write_text("old", encoding="utf-8")

    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(sever.report, "write_report", broken)
    outcome = CliRunner().invoke(app, ["documents", str(tmp_path)])
    assert outcome.exit_code != 0
    assert (tmp_path / "report.html").read_text() == "old"
    assert (tmp_path / "documents.csv").is_file()


def test_lookalike_letters_across_sheets_are_one_value_not_a_conflict():
    # 20260818-3: one УПД read «С01-345853» (Cyrillic С) on one sheet and «C01-345853» (Latin C) on another.
    upd = {**UPD, "invoice_number": "С01-345853"}
    result = video(sheet("p1", 0, "upd", **upd), sheet("p2", 4, "upd", **{**upd, "invoice_number": "C01-345853"}))
    [record] = assemble_documents(result)["documents"]
    assert record["status"] == "complete"
    assert record["fields"]["invoice_number"]["status"] == "recognized"
    assert record["fields"]["invoice_number"]["candidates"] == []
    assert [s["page_id"] for s in record["sheets"]] == ["p1", "p2"]
