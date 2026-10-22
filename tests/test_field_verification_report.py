"""Fine-print history shows isolated readings and only bound native evidence."""
from copy import deepcopy
import hashlib
import html
from pathlib import Path
import re

from PIL import Image
import pytest

from sever.report import write_report
from test_documents import UPD, sheet, video


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def verified_run(tmp_path):
    run = tmp_path / "run"
    source_dir = run / "video" / "frames"
    source_dir.mkdir(parents=True)
    panel_dir = run / "image-reading-evidence" / "image-panels"
    panel_dir.mkdir(parents=True)
    frames, views, sources, panels, readings = [], [], {}, {}, []
    for index in range(2):
        frame_id, label = f"frame-{index}", f"frame_{index + 1:02d}"
        path = source_dir / f"{frame_id}.png"
        Image.new("RGB", (80, 100), (60 + index * 40, 90, 120)).save(path)
        frames.append({"frame_id": frame_id, "path": str(path), "timestamp_s": index + .5})
        views.append({"frame_id": frame_id, "frame_path": str(path), "observed_form": "УПД", "visible_text": "",
                      "document": {"fields": {"consignor_address": {"status": "recognized", "raw_value": f"original proposal {index}"}}}})
        tile_id = f"{label}_detail_01"
        tile = panel_dir / f"{tile_id}.png"
        with Image.open(path) as original:
            original.crop((5, 10, 70, 35)).save(tile)
        panel = {"panel_id": tile_id, "frame_id": frame_id, "source_sha256": digest(path),
                 "upload_path": str(tile), "upload_sha256": digest(tile), "verification_purpose": "transcription_detail"}
        panels[tile_id] = panel
        sources[label] = {"frame_id": frame_id, "source_path": str(path), "source_sha256": digest(path),
                          "panels": {tile_id: panel}}
        readings.append({"frame_id": label, "status": "recognized", "raw_value": f"fresh literal {index}",
                         "panel_ids": [tile_id]})
    field = {"status": "recognized", "raw_value": "Москва, стр. 8", "normalized_value": "Москва, стр. 8", "evidence": []}
    entry = {"field_name": "consignor_address", "status": "resolved", "phase": "initial",
        "frame_ids": [frame["frame_id"] for frame in frames], "answer": {
            "fields": {"consignor_address": field}, "panels": panels,
            "recognition": {"pixel_proof": {"sources": sources, "panels": panels},
                            "metadata": {"api_key": "caller-secret"}, "raw_responses": [{"body": "caller-secret"}]},
            "adjudication": {"field_reviews": {"consignor_address": {"model_review": {"readings": readings}}}}}}
    page = sheet("page-1", 0, "upd", **UPD)
    page.update(frames=frames, selected_frame_id=frames[0]["frame_id"],
                image_recognition={"views": views, "detail_verifications": [entry]})
    return run, video(page), entry, panels


def render(fixture):
    run, result, _, _ = fixture
    write_report(run, result)
    return (run / "report.html").read_text()


def sections(report):
    return re.findall(r'<details class="field-verifications">(.*?)</details>', report, re.S)


def test_leaf_history_shows_independent_literals_and_actual_native_tiles(verified_run):
    _, result, _, panels = verified_run
    before = deepcopy(result)
    report = render(verified_run)
    [section] = sections(report)
    assert "Проверка мелкого текста" in section and "Каждый кадр прочитан отдельным новым запросом" in section
    assert "Адрес грузоотправителя" in section and "Подтверждено" in section
    for index, panel in enumerate(panels.values()):
        assert f"fresh literal {index}" in section
        assert f'href="image-reading-evidence/image-panels/{Path(panel["upload_path"]).name}"' in section
        assert f'original proposal {index}' in report.split('<details class="field-verifications">')[0]
    assert "caller-secret" not in report and "raw_responses" not in report and "api_key" not in report
    assert result == before


def test_document_history_is_below_document_cards_and_before_sheets(verified_run):
    _, result, entry, _ = verified_run
    result["pages"][0]["image_recognition"].pop("detail_verifications")
    entry.update(document_type="upd", source_page_ids=["page-1"])
    result["document_field_verifications"] = [entry]
    report = render(verified_run)
    [section] = sections(report)
    assert "Листы: page-1" in section and "fresh literal 0" in section
    assert report.index('class="doc-grid"') < report.index('class="field-verifications"') < report.index('Листы видео подробно')
    assert "caller-secret" not in report


def test_detail_text_is_escaped_and_raw_transport_is_not_rendered(verified_run):
    _, _, entry, _ = verified_run
    attack = '<script>alert("bad")</script> & <img src=x onerror=alert(1)>'
    entry["answer"]["fields"]["consignor_address"]["raw_value"] = attack
    readings = entry["answer"]["adjudication"]["field_reviews"]["consignor_address"]["model_review"]["readings"]
    readings[0]["raw_value"] = attack
    entry["answer"]["adjudication"]["provider_request"] = {"Authorization": "caller-secret"}
    report = render(verified_run)
    assert html.escape(attack) in sections(report)[0]
    assert "<script>" not in report and "<img src=x" not in report and "caller-secret" not in report


@pytest.mark.parametrize("invalid", ["outside", "symlink", "tampered_tile", "tampered_source", "foreign_sha", "foreign_frame", "locator_only"])
def test_unbound_or_altered_images_never_become_detail_links(verified_run, tmp_path, invalid):
    run, _, entry, panels = verified_run
    panel = next(iter(panels.values()))
    tile = Path(panel["upload_path"])
    if invalid in {"outside", "symlink"}:
        outside = tmp_path / "external-private.png"
        Image.new("RGB", (4, 4)).save(outside)
        target = outside
        if invalid == "symlink":
            target = run / "external-link.png"
            target.symlink_to(outside)
        panel.update(upload_path=str(target), upload_sha256=digest(outside))
    elif invalid == "tampered_tile":
        Image.new("RGB", (10, 10), "red").save(tile)
    elif invalid == "tampered_source":
        source = entry["answer"]["recognition"]["pixel_proof"]["sources"]["frame_01"]
        Image.new("RGB", (80, 100), "red").save(source["source_path"])
    elif invalid == "foreign_sha":
        panel["source_sha256"] = "f" * 64
    elif invalid == "foreign_frame":
        panel["frame_id"] = "other-frame"
    else:
        panel["verification_purpose"] = "localization"
    [section] = sections(render(verified_run))
    assert f'href="image-reading-evidence/image-panels/{tile.name}"' not in section
    assert "external-private.png" not in section and "external-link.png" not in section
    assert "fresh literal 0" in section


@pytest.mark.parametrize("status", ["failed", "unresolved", "insufficient_source_views"])
def test_incomplete_verification_history_does_not_claim_confirmation(verified_run, status):
    _, _, entry, _ = verified_run
    entry.update(status=status, answer={})
    [section] = sections(render(verified_run))
    assert "Подтверждено" not in section
    assert "Отдельные чтения кадров не сохранены" in section
