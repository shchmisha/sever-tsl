"""Joint pixel reviews remain distinct from original per-frame proposals."""
from copy import deepcopy
import hashlib
import html
from pathlib import Path
import re

from PIL import Image
import pytest

from sever.report import write_report


@pytest.fixture
def reviewed_run(tmp_path):
    run = tmp_path / "run"
    frames_dir = run / "video" / "frames"
    frames_dir.mkdir(parents=True)
    frames = []
    for index in range(3):
        path = frames_dir / f"frame-{index + 1}.png"
        Image.new("RGB", (80, 50), (30 + index * 60, 80, 130)).save(path)
        frames.append({"frame_id": f"frame-{index + 1}", "timestamp_s": index + .5, "path": str(path)})
    proposals = [{"frame_id": frame["frame_id"], "frame_path": frame["path"],
                  "observed_form": "УПД", "visible_text": "own heading",
                  "document": {"fields": {"consignee": {"raw_value": value, "status": "recognized"}}}}
                 for frame, value in zip(frames, ["old entity", "old entity + identifiers", "old third proposal"])]
    field = {"raw_value": "reviewed entity", "normalized_value": "reviewed entity", "status": "recognized",
             "reason": "Fresh reading of own role caption", "evidence": [
                 {"frame_id": frame["frame_id"], "text": "Own caption: reviewed entity"} for frame in frames]}
    panel = run / "image-reading-evidence" / "actual-joint-panel.png"
    panel.parent.mkdir()
    with Image.open(frames[2]["path"]) as source:
        source.crop((2, 3, 60, 40)).save(panel)
    proof = {"panel_id": "frame_03_zoom_01", "frame_id": "frame-3",
             "source_sha256": hashlib.sha256(Path(frames[2]["path"]).read_bytes()).hexdigest(),
             "upload_path": str(panel), "upload_sha256": hashlib.sha256(panel.read_bytes()).hexdigest()}
    review = {"phase": "initial", "status": "resolved", "frame_ids": [frame["frame_id"] for frame in frames],
              "field_names": ["consignee"], "answer": {"fields": {"consignee": field}, "panels": {"p": proof},
              "adjudication": {"field_reviews": {"consignee": {"model_review": {
                  "resolution_reason": "Role boundary confirmed from all supplied images"}}}}}}
    document = {"document_type": "upd", "errors": [], "notes": [], "fields": {"consignee": deepcopy(field)}}
    page = {"page_id": "page-1", "start_s": 0, "end_s": 4, "selected_frame_id": "frame-1",
            "frames": frames, "result": document,
            "image_recognition": {"views": proposals, "conflict_reviews": [review]}}
    result = {"source": {"path": "actual video.mp4", "sha256": "source hash"},
              "pages": [page], "elapsed_seconds": 3.4, "status": "processing"}
    return run, result, review, panel


def render(fixture):
    run, result, _, _ = fixture
    write_report(run, result)
    return (run / "report.html").read_text()


def review_section(report):
    return re.search(r'<details class="conflict-reviews">(.*?)</details>', report, re.S).group(1)


def test_joint_review_keeps_all_original_proposals_and_links_three_actual_sources(reviewed_run):
    _, result, _, _ = reviewed_run
    before = deepcopy(result)
    report = render(reviewed_run)
    section = review_section(report)
    assert "Повторная проверка спорных полей по изображениям" in section
    assert "новый совместный ответ модели" in section
    assert "Исходные ответы модели сохранены" in section
    assert "Разногласия разрешены" in section
    assert "reviewed entity" in section and "Fresh reading of own role caption" in section
    assert "Role boundary confirmed from all supplied images" in section
    for index, value in enumerate(["old entity", "old entity + identifiers", "old third proposal"], 1):
        assert value in report.split('<details class="conflict-reviews">')[0]
        assert f'href="video/frames/frame-{index}.png">frame-{index}</a>' in section
    assert 'href="image-reading-evidence/actual-joint-panel.png">frame-3: frame_03_zoom_01</a>' in section
    assert result == before


def test_review_escapes_model_text_and_omits_raw_transport_secrets(reviewed_run):
    _, _, review, _ = reviewed_run
    attack = '<script>alert("model")</script> & <img src=x onerror=alert(1)>'
    review["answer"]["fields"]["consignee"].update(raw_value=attack, reason=attack)
    review["answer"]["fields"]["consignee"]["evidence"][0]["text"] = attack
    review["answer"]["adjudication"]["field_reviews"]["consignee"]["model_review"]["resolution_reason"] = attack
    review["answer"]["recognition"] = {"metadata": {"api_key": "must-never-leak", "source_path": "/private/arbitrary"},
                                        "raw_responses": [{"body": "must-never-leak"}]}
    review["answer"]["adjudication"]["raw_trace"] = "must-never-leak"
    report = render(reviewed_run)
    assert html.escape(attack) in review_section(report)
    assert "<script>" not in report and "<img src=x" not in report
    assert "must-never-leak" not in report and "/private/arbitrary" not in report


@pytest.mark.parametrize("status", ["unresolved", "failed"])
def test_unresolved_and_failed_reviews_show_original_conflicts_without_claiming_success(reviewed_run, status):
    _, result, review, _ = reviewed_run
    review.update(status=status, phase="final")
    result["pages"][0]["result"]["fields"]["consignee"].update(raw_value=None, normalized_value=None, status="conflict")
    if status == "failed":
        review["error_type"] = 'ProviderError<script>bad</script>'
        review["answer"] = {}
    report = render(reviewed_run)
    section = review_section(report)
    assert "Итоговая повторная проверка" in section
    assert "Разногласия разрешены" not in section
    assert "old entity + identifiers" in report and "conflict" in report
    if status == "failed":
        assert "Повторная проверка не завершена" in section
        assert "ProviderError&lt;script&gt;bad&lt;/script&gt;" in section
        assert "Исходные разногласия сохранены" in section
    else:
        assert "Разногласия сохранились" in section


@pytest.mark.parametrize("invalid", ["outside", "symlink", "missing", "upload_hash", "source_hash", "unknown_frame", "not_image"])
def test_review_never_links_unbound_or_arbitrary_proof_paths(reviewed_run, tmp_path, invalid):
    run, _, review, panel = reviewed_run
    proof = review["answer"]["panels"]["p"]
    if invalid in {"outside", "symlink"}:
        outside = tmp_path / "outside.png"
        Image.new("RGB", (4, 4)).save(outside)
        target = outside
        if invalid == "symlink":
            target = run / "outside-link.png"
            target.symlink_to(outside)
        proof.update(upload_path=str(target), upload_sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    elif invalid == "missing":
        proof["upload_path"] = str(run / "missing.png")
    elif invalid == "upload_hash":
        proof["upload_sha256"] = "wrong"
    elif invalid == "source_hash":
        proof["source_sha256"] = "wrong"
    elif invalid == "unknown_frame":
        proof["frame_id"] = "invented-frame"
    else:
        panel.write_bytes(b"secret payload, not an image")
        proof["upload_sha256"] = hashlib.sha256(panel.read_bytes()).hexdigest()
    section = review_section(render(reviewed_run))
    assert "Переданные модели фрагменты" not in section
    assert "outside.png" not in section and "outside-link.png" not in section
    assert "missing.png" not in section


def test_unknown_and_outside_frame_citations_remain_plain_text(reviewed_run, tmp_path):
    _, result, review, _ = reviewed_run
    outside = tmp_path / "outside-frame.png"
    Image.new("RGB", (4, 4)).save(outside)
    result["pages"][0]["image_recognition"]["views"].append(
        {"frame_id": "outside-frame", "frame_path": str(outside), "observed_form": "УПД", "visible_text": ""})
    review["frame_ids"].extend(["outside-frame", "unknown-frame"])
    review["answer"]["fields"]["consignee"]["evidence"].extend(
        {"frame_id": name, "text": "unverified citation"} for name in ("outside-frame", "unknown-frame"))
    section = review_section(render(reviewed_run))
    assert "outside-frame" in section and "unknown-frame" in section
    assert '>outside-frame</a>' not in section and '>unknown-frame</a>' not in section
    assert '../outside-frame.png' not in section


def add_joint_readings(fixture):
    _, result, review, _ = fixture
    frames = result["pages"][0]["frames"]
    review["answer"]["recognition"] = {"pixel_proof": {"sources": {
        f"frame_{index:02d}": {"frame_id": frame["frame_id"],
            "source_sha256": hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest()}
        for index, frame in enumerate(frames, 1)}}}
    readings = [{"frame_id": f"frame_{index:02d}", "raw_value": value,
                 "status": "recognized", "evidence_text": f"Own role: {value}"}
                for index, value in enumerate(["joint value <A>", "joint value B & C", "joint third value"], 1)]
    review["answer"]["adjudication"]["field_reviews"]["consignee"]["model_review"]["readings"] = readings
    return readings


def test_unresolved_joint_model_per_frame_values_are_visible_without_hiding_prior_proposals(reviewed_run):
    _, result, review, _ = reviewed_run
    readings = add_joint_readings(reviewed_run)
    review["status"] = "unresolved"
    field = review["answer"]["fields"]["consignee"]
    field.update(raw_value=None, normalized_value=None, status="conflict")
    result["pages"][0]["result"]["fields"]["consignee"] = deepcopy(field)
    before = deepcopy(result)
    report = render(reviewed_run)
    section = review_section(report)
    assert "Разногласия сохранились" in section
    assert "Предложения совместного ответа по каждому кадру" in section
    assert "части одного совместного ответа модели" in section
    for index, reading in enumerate(readings, 1):
        assert html.escape(reading["raw_value"]) in section
        assert html.escape(reading["evidence_text"]) in section
        assert f'href="video/frames/frame-{index}.png">frame-{index}</a>' in section
    assert "old entity + identifiers" in report.split('<details class="conflict-reviews">')[0]
    assert result == before


@pytest.mark.parametrize("binding", ["wrong_hash", "outside_id", "unsupplied_label"])
def test_joint_label_is_never_relabelled_as_actual_source_without_bound_proof(reviewed_run, binding):
    _, _, review, _ = reviewed_run
    readings = add_joint_readings(reviewed_run)
    source = review["answer"]["recognition"]["pixel_proof"]["sources"]["frame_01"]
    if binding == "wrong_hash":
        source["source_sha256"] = "wrong"
    elif binding == "outside_id":
        source["frame_id"] = "invented-source"
    else:
        readings[0]["frame_id"] = "frame_99<script>bad</script>"
    report = render(reviewed_run)
    table = review_section(report).split("Предложения совместного ответа по каждому кадру", 1)[1]
    assert 'href="video/frames/frame-1.png">frame-1</a>' not in table
    assert "неподтверждённая метка" in table
    assert html.escape(readings[0]["raw_value"]) in table
    assert "<script>" not in report


def test_active_joint_review_has_explicit_processing_state(reviewed_run):
    _, _, review, _ = reviewed_run
    review.update(status="processing", answer={})
    section = review_section(render(reviewed_run))
    assert "Повторная проверка продолжается" in section
    assert "Статус проверки не указан" not in section
    assert "Разногласия разрешены" not in section


def add_type_review(fixture, *, status="resolved"):
    _, result, review, _ = fixture
    add_joint_readings(fixture)
    frames = result["pages"][0]["frames"]
    panels, readings = {}, []
    for index, frame in enumerate(frames, 1):
        identifier = f"frame_{index:02d}_context_r0"
        panels[identifier] = {"panel_id": identifier, "frame_id": frame["frame_id"],
            "source_sha256": hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest(),
            "upload_sha256": hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest(),
            "upload_path": frame["path"]}
        readings.append({"frame_id": f"frame_{index:02d}", "document_type": "ttn",
                         "heading_text": f"Literal own heading {index} <Ω>", "panel_ids": [identifier]})
    proof = review["answer"]["recognition"]["pixel_proof"]
    review.update(kind="document_type", status=status, field_names=[])
    review["answer"] = {"document_type": "ttn" if status == "resolved" else "unknown", "fields": {},
        "panels": panels, "recognition": {"pixel_proof": proof},
        "classification_evidence": [{"frame_id": frame["frame_id"], "text": reading["heading_text"]}
                                    for frame, reading in zip(frames, readings)],
        "adjudication": {"status": status, "model_review": {
            "resolution_reason": "Fresh own form confirmed on every image", "readings": readings}}}
    return readings


def test_type_review_exposes_actual_per_frame_type_heading_and_cited_image(reviewed_run):
    _, result, _, _ = reviewed_run
    readings = add_type_review(reviewed_run)
    before = deepcopy(result)
    report = render(reviewed_run)
    section = review_section(report)
    assert "Проверка формы основного документа" in section
    assert "Ответ совместной проверки по форме: <strong>ttn</strong>" in section
    assert "Буквальный собственный заголовок" in section and "Тип модели" in section
    assert "Fresh own form confirmed on every image" in section
    assert "части одного совместного ответа модели" in section
    assert "Ответ с проверенными полями не сохранён" not in section
    for index, reading in enumerate(readings, 1):
        assert html.escape(reading["heading_text"]) in section
        assert f'href="video/frames/frame-{index}.png">frame-{index}</a>' in section
        assert f'href="video/frames/frame-{index}.png">frame-{index}: frame_{index:02d}_context_r0</a>' in section
    assert "old entity + identifiers" in report.split('<details class="conflict-reviews">')[0]
    assert result == before


@pytest.mark.parametrize("status", ["unresolved", "failed"])
def test_unresolved_or_failed_type_review_keeps_literal_disagreement_and_originals(reviewed_run, status):
    _, _, review, _ = reviewed_run
    readings = add_type_review(reviewed_run, status=status)
    if status == "unresolved":
        readings[1].update(document_type="trn", heading_text="Other literal own heading")
        readings[2].update(document_type="unknown", heading_text="Unreadable own form")
    else:
        review.update(error_type="TypeReviewProviderError", answer={})
    report = render(reviewed_run)
    section = review_section(report)
    assert "Разногласия разрешены" not in section
    assert "old entity + identifiers" in report
    if status == "unresolved":
        assert "Разногласия сохранились" in section and "<strong>unknown</strong>" in section
        assert "Other literal own heading" in section and "Unreadable own form" in section
        assert "<td>ttn</td>" in section and "<td>trn</td>" in section and "<td>unknown</td>" in section
    else:
        assert "Повторная проверка не завершена" in section and "TypeReviewProviderError" in section
        assert "Исходные разногласия сохранены" in section


def test_type_review_citations_require_matching_source_and_panel_not_raw_metadata(reviewed_run):
    _, _, review, _ = reviewed_run
    readings = add_type_review(reviewed_run, status="unresolved")
    attack = '<script>bad()</script> & <img src=x onerror=bad()>'
    readings[0]["heading_text"] = attack
    readings[0]["panel_ids"] = [readings[1]["panel_ids"][0], "invented<script>panel</script>"]
    review["answer"]["adjudication"]["model_review"]["resolution_reason"] = attack
    review["answer"]["recognition"].update(raw_responses=[{"secret": "HIDDEN-TRANSPORT"}],
                                            metadata={"local_path": "/private/HIDDEN-PATH"})
    report = render(reviewed_run)
    section = review_section(report)
    type_table = section.split("Форма документа · Предложения", 1)[1]
    first_row = type_table.split("<tbody>", 1)[1].split("</tr>", 1)[0]
    assert html.escape(attack) in section and "<script>" not in report and "<img src=x" not in report
    assert "неподтверждённый фрагмент" in first_row
    assert '>frame-2: frame_02_context_r0</a>' not in first_row
    assert "HIDDEN-TRANSPORT" not in report and "HIDDEN-PATH" not in report


def test_invalidated_type_review_is_historical_and_cannot_claim_current_resolution(reviewed_run):
    _, _, review, _ = reviewed_run
    add_type_review(reviewed_run)
    review["invalidated_by"] = {"frame_id": "frame-3", "document_type": "trn",
        "reason": "New own-heading reading <conflicts>", "metadata": {"secret": "HIDDEN-INVALIDATION"}}
    section = review_section(render(reviewed_run))
    assert "Решение проверки оспорено новым чтением" in section
    assert "Разногласия разрешены" not in section
    assert "оно больше не подтверждает текущий результат" in section
    assert 'href="video/frames/frame-3.png">frame-3</a>' in section
    assert "New own-heading reading &lt;conflicts&gt;" in section
    assert "Ответ совместной проверки по форме: <strong>ttn</strong>" in section
    assert "HIDDEN-INVALIDATION" not in render(reviewed_run)


def test_runs_without_conflict_reviews_keep_report_unchanged(reviewed_run):
    _, result, _, _ = reviewed_run
    recognition = result["pages"][0]["image_recognition"]
    del recognition["conflict_reviews"]
    old = render(reviewed_run)
    recognition["conflict_reviews"] = []
    empty = render(reviewed_run)
    # Empty audit metadata appears only in the existing JSON details.
    assert old.split('<details><summary>Качество')[0] == empty.split('<details><summary>Качество')[0]
    assert 'class="conflict-reviews"' not in empty
