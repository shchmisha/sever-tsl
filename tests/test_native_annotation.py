from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import pytest

from tools.annotate_native_readings import build_plan, render


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(frame_id, text):
    return {"frame_id": frame_id, "region_index": None, "text": text, "bbox": [0., 0., 1., 1.]}


def field(frame_id, raw, text):
    return {"status": "recognized", "raw_value": raw, "normalized_value": raw,
            "evidence": [evidence(frame_id, text)], "confidence": None, "reason": None}


@pytest.fixture
def native_run(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic-source-video-for-proof-tests")
    source_id = digest(source)
    values = {"supplier": ('ООО «Печатный поставщик»', 'Поставщик: ООО «Печатный поставщик»'),
              "consignee": ('ООО «Печатный получатель»', 'Грузополучатель: ООО «Печатный получатель»'),
              "document_number": ('0042', 'ТОРГ-12 Номер документа: 0042'),
              "document_date": ('15.02.2020', 'Дата составления: 15.02.2020')}
    frames, views, readings = [], [], []
    for index, timestamp in enumerate((.5, 1.2), 1):
        frame_id = f"frame-{index}"
        path = tmp_path / (frame_id + ".png")
        Image.new("RGB", (320, 180), (10 * index, 40, 180)).save(path)
        frame = {"frame_id": frame_id, "path": str(path), "timestamp_s": timestamp,
                 "source_pts": round(timestamp * 10), "source_time_base": "1/10", "display_rotation_applied": True}
        frames.append(frame)
        panel = {"panel_id": "context_r270", "frame_id": frame_id, "bbox": [0., 0., 1., 1.],
                 "rotation_clockwise_degrees": 270}
        document = {"source_id": source_id, "page_id": "leaf-1", "document_type": "torg12", "type_confidence": None,
                    "classification_evidence": [evidence(frame_id, "Унифицированная форма ТОРГ-12")],
                    "fields": {name: field(frame_id, raw, text) for name, (raw, text) in values.items()},
                    "errors": [], "notes": []}
        readings.append(document)
        views.append({"frame_id": frame_id, "frame_path": str(path), "timestamp_s": timestamp,
                      "origin": "model_pixel_transcription", "document": deepcopy(document),
                      "reading": {"document_type": "torg12", "heading_panel_ids": ["context_r270"],
                                  "fields": {name: {"raw_value": raw, "status": "recognized",
                                                     "panel_ids": ["context_r270"]}
                                             for name, (raw, _) in values.items()}},
                      "pixel_proof": {"frame_id": frame_id, "source_path": str(path), "source_sha256": digest(path),
                                      "timestamp_s": timestamp, "source_pixel_size": [320, 180],
                                      "panels": {"context_r270": panel}}})
    final = deepcopy(readings[0])
    final["classification_evidence"] += deepcopy(readings[1]["classification_evidence"])
    for name in values:
        final["fields"][name]["evidence"] += deepcopy(readings[1]["fields"][name]["evidence"])
    data = {"source": {"path": str(source), "sha256": source_id},
            "configuration": {"document_reader": "native_image", "local_ocr_used": False, "vision_model": "synthetic-native-test-model"},
            "pages": [{"page_id": "leaf-1", "frames": frames, "result": final,
                       "image_recognition": {"engine": "native_image", "views": views}}]}
    save(tmp_path, data)
    return tmp_path, data


def save(run, data):
    (run / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_plan_values_and_two_frames_come_only_from_actual_native_results(native_run):
    run, data = native_run
    original = deepcopy(data)
    plan = build_plan(run)
    assert plan["source_video_sha256"] == data["source"]["sha256"]
    assert plan["results_sha256"] == digest(run / "results.json")
    assert plan["inserted_seconds"] == 3.
    assert plan["display_rotation_clockwise_degrees"] == 270
    assert [view["timestamp_s"] for view in plan["selected_views"]] == [.5, 1.2]
    assert all(set(ids) == {"frame-1", "frame-2"} for ids in plan["field_support"].values())
    for view in plan["selected_views"]:
        for name, actual in view["fields"].items():
            assert actual["raw_value"] == data["pages"][0]["result"]["fields"][name]["raw_value"]
            assert {item["frame_id"] for item in actual["evidence"]} == {view["frame_id"]}
    assert plan["manual_ground_truth_read"] is False
    assert plan["manual_values_accepted"] is False
    assert plan["moving_frame_values_drawn"] is False
    assert plan["word_boxes_drawn"] is False
    assert data == original


def test_render_proof_accepts_formatting_equivalence_but_keeps_actual_literals(native_run):
    run, data = native_run
    page = data["pages"][0]
    second = page["image_recognition"]["views"][1]
    raw = 'ООО "Печатный поставщик"'
    second["document"]["fields"]["supplier"]["raw_value"] = raw
    second["document"]["fields"]["supplier"]["normalized_value"] = raw
    second["document"]["fields"]["supplier"]["evidence"][0]["text"] = "Поставщик: " + raw
    second["reading"]["fields"]["supplier"]["raw_value"] = raw
    page["result"]["fields"]["supplier"]["evidence"][1]["text"] = "Поставщик: " + raw
    save(run, data)

    plan = build_plan(run)

    assert plan["field_support"]["supplier"] == ["frame-1", "frame-2"]
    assert plan["selected_views"][1]["fields"]["supplier"]["raw_value"] == raw
    assert plan["selected_views"][1]["fields"]["supplier"]["fused_raw_value"] == 'ООО «Печатный поставщик»'


def test_source_video_tamper_is_rejected_before_render(native_run):
    run, _ = native_run
    (run / "source.mp4").write_bytes(b"different source")
    with pytest.raises(ValueError, match="Source video bytes"):
        build_plan(run)


def test_model_read_frame_tamper_is_rejected_before_render(native_run):
    run, _ = native_run
    Image.new("RGB", (320, 180), "yellow").save(run / "frame-1.png")
    with pytest.raises(ValueError, match="frame bytes"):
        build_plan(run)


@pytest.mark.parametrize("change", ["proof_path", "view_path", "proof_frame", "frame_timestamp", "proof_timestamp", "dimensions", "document_source", "document_page"])
def test_source_frame_identity_timestamp_and_leaf_binding_are_checked(native_run, change):
    run, data = native_run
    page = data["pages"][0]
    view = page["image_recognition"]["views"][0]
    if change == "proof_path":
        view["pixel_proof"]["source_path"] = str(run / "other.png")
    elif change == "view_path":
        view["frame_path"] = str(run / "other.png")
    elif change == "proof_frame":
        view["pixel_proof"]["frame_id"] = "foreign"
    elif change == "frame_timestamp":
        page["frames"][0]["timestamp_s"] += .1
    elif change == "proof_timestamp":
        view["pixel_proof"]["timestamp_s"] += .1
    elif change == "dimensions":
        view["pixel_proof"]["source_pixel_size"] = [180, 320]
    elif change == "document_source":
        view["document"]["source_id"] = "foreign"
    else:
        view["document"]["page_id"] = "other-leaf"
    save(run, data)
    with pytest.raises(ValueError):
        build_plan(run)


@pytest.mark.parametrize("location", ["view", "fused"])
def test_foreign_evidence_frame_is_rejected(native_run, location):
    run, data = native_run
    page = data["pages"][0]
    document = page["result"] if location == "fused" else page["image_recognition"]["views"][0]["document"]
    document["fields"]["supplier"]["evidence"][0]["frame_id"] = "foreign"
    save(run, data)
    with pytest.raises(ValueError, match="foreign|actual read frame"):
        build_plan(run)


def test_multiple_panel_citations_from_one_frame_cannot_claim_two_readings(native_run):
    run, data = native_run
    page = data["pages"][0]
    page["result"]["fields"]["supplier"]["evidence"] = [deepcopy(page["result"]["fields"]["supplier"]["evidence"][0])] * 3
    save(run, data)
    with pytest.raises(ValueError, match="supplier lacks two distinct"):
        build_plan(run)


def test_fused_value_absent_from_actual_model_readings_cannot_be_shown(native_run):
    run, data = native_run
    data["pages"][0]["result"]["fields"]["supplier"]["raw_value"] = "Hand-added foreign entity"
    save(run, data)
    with pytest.raises(ValueError, match="supplier lacks two distinct"):
        build_plan(run)


def test_disagreeing_second_actual_value_cannot_be_used_as_confirmation(native_run):
    run, data = native_run
    value = data["pages"][0]["image_recognition"]["views"][1]["document"]["fields"]["supplier"]
    value["raw_value"] = "Other company"
    data["pages"][0]["image_recognition"]["views"][1]["reading"]["fields"]["supplier"]["raw_value"] = "Other company"
    value["evidence"][0]["text"] = "Поставщик: Other company"
    save(run, data)
    with pytest.raises(ValueError, match="supplier lacks two distinct"):
        build_plan(run)


@pytest.mark.parametrize("change", ["foreign_heading_panel", "different_rotation", "fake_word_box", "ocr_region", "foreign_field_panel"])
def test_only_recorded_native_panels_and_original_model_readings_are_accepted(native_run, change):
    run, data = native_run
    view = data["pages"][0]["image_recognition"]["views"][0]
    if change == "foreign_heading_panel":
        view["reading"]["heading_panel_ids"] = ["not-supplied"]
    elif change == "different_rotation":
        view["pixel_proof"]["panels"]["context_r270"]["rotation_clockwise_degrees"] = 90
    elif change == "fake_word_box":
        view["document"]["fields"]["supplier"]["evidence"][0]["bbox"] = [.1, .2, .3, .04]
    elif change == "ocr_region":
        view["document"]["fields"]["supplier"]["evidence"][0]["region_index"] = 4
    else:
        view["reading"]["fields"]["supplier"]["panel_ids"] = ["not-supplied"]
    save(run, data)
    with pytest.raises(ValueError):
        build_plan(run)


def test_duplicate_view_does_not_count_as_another_reading(native_run):
    run, data = native_run
    views = data["pages"][0]["image_recognition"]["views"]
    views.append(deepcopy(views[0]))
    save(run, data)
    with pytest.raises(ValueError, match="Repeated frame IDs"):
        build_plan(run)


def test_display_uses_actual_per_view_formatting_and_records_fused_value(native_run):
    run, data = native_run
    value = data["pages"][0]["image_recognition"]["views"][1]["document"]["fields"]["document_date"]
    value["raw_value"] = " 15.02.2020\n"
    data["pages"][0]["image_recognition"]["views"][1]["reading"]["fields"]["document_date"]["raw_value"] = value["raw_value"]
    value["evidence"][0]["text"] = "Дата составления: 15.02.2020"
    data["pages"][0]["result"]["fields"]["document_date"]["evidence"][1]["text"] = value["evidence"][0]["text"]
    save(run, data)
    plan = build_plan(run)
    assert plan["selected_views"][1]["fields"]["document_date"]["raw_value"] == " 15.02.2020\n"
    assert plan["selected_views"][1]["fields"]["document_date"]["fused_raw_value"] == "15.02.2020"


@pytest.mark.parametrize("freeze_seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_pause_duration_is_rejected(native_run, freeze_seconds):
    run, _ = native_run
    with pytest.raises(ValueError, match="duration"):
        build_plan(run, freeze_seconds=freeze_seconds)


def test_errors_are_preserved_but_legacy_ocr_is_not_promoted_to_native_demo(native_run):
    run, data = native_run
    data["pages"][0]["result"]["errors"] = ["actual reading failure"]
    save(run, data)
    plan = build_plan(run)
    assert plan["recorded_result_errors"] == ["actual reading failure"]
    data["configuration"]["document_reader"] = "ocr_guided"
    save(run, data)
    with pytest.raises(ValueError, match="direct image"):
        build_plan(run)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="local ffmpeg required")
def test_offline_render_preserves_source_playback_and_inserts_only_actual_reading_pauses(native_run):
    run, data = native_run
    source = run / "source.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=2", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
    source_id = digest(source)
    data["source"]["sha256"] = source_id
    page = data["pages"][0]
    page["result"]["source_id"] = source_id
    for view in page["image_recognition"]["views"]:
        view["document"]["source_id"] = source_id
    save(run, data)
    output = run / "native-demo.mp4"
    manifest = render(run, output)
    assert output.is_file() and output.stat().st_size > 1000
    assert manifest["output_sha256"] == digest(output)
    assert manifest["actual_output_duration_seconds"] == pytest.approx(5., abs=.15)
    assert manifest["inserted_seconds"] == 3.
    assert [segment["kind"] for segment in manifest["timeline"]] == ["source_playback", "actual_model_read_frame_freeze", "source_playback", "actual_model_read_frame_freeze", "source_playback"]
    assert sum(segment["planned_output_end_s"] - segment["planned_output_start_s"] for segment in manifest["timeline"] if segment["kind"] == "source_playback") == pytest.approx(2.)
    assert all(segment["speed"] == 1. and not segment["recognition_values_overlaid"] for segment in manifest["timeline"] if segment["kind"] == "source_playback")
    assert len(manifest["freeze_images"]) == 2
    assert all(item["source_crop"] is None for item in manifest["freeze_images"])
    assert all(digest(Path(item["path"])) == item["sha256"] for item in manifest["freeze_images"])
    saved = json.loads(output.with_suffix(".provenance.json").read_text())
    assert saved["results_sha256"] == digest(run / "results.json")
    assert saved["audio"] == "source audio preserved during playback; silence during inserted pauses"


def _fake_encoding(monkeypatch):
    """Exercise publication faults without unrelated encoder process failures."""
    import tools.annotate_native_readings as module
    monkeypatch.setattr(module.shutil, "which", lambda name: "/synthetic/" + name)
    monkeypatch.setattr(module, "_probe", lambda path: {
        "format": {"duration": "2" if Path(path).name == "source.mp4" else "5"},
        "streams": [{"codec_type": "video"}]})
    def encode(command, **_):
        target = Path(command[-1])
        assert ".native-stage-" in str(target.parent)
        assert target.name == "video.mp4"
        target.write_bytes(b"synthetic-encoded-output-for-publication-tests")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module.subprocess, "run", encode)
    return module


def _prior_artifact(run):
    video = run / "prior-successful.mp4"
    proof = video.with_suffix(".provenance.json")
    assets = video.with_name(video.stem + ".native-reading-assets")
    assets.mkdir()
    snapshot = assets / "freeze-1.png"
    video.write_bytes(b"prior-video")
    proof.write_bytes(b"prior-proof")
    snapshot.write_bytes(b"prior-snapshot")
    return {path: digest(path) for path in (video, proof, snapshot)}


def _assert_no_partial_demo(run, output):
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()
    assert not output.with_name(output.stem + ".native-reading-assets").exists()
    assert not list(run.glob(f".{output.stem}.native-stage-*"))


@pytest.mark.parametrize("existing", ["video", "provenance", "assets", "dangling_video_symlink"])
def test_existing_artifact_paths_are_rejected_before_any_render_or_modification(native_run, monkeypatch, existing):
    run, _ = native_run
    output = run / "native-demo.mp4"
    provenance = output.with_suffix(".provenance.json")
    assets = output.with_name(output.stem + ".native-reading-assets")
    if existing == "video":
        output.write_bytes(b"existing-video")
        preserved = {output: digest(output)}
    elif existing == "provenance":
        provenance.write_bytes(b"existing-proof")
        preserved = {provenance: digest(provenance)}
    elif existing == "assets":
        assets.mkdir()
        snapshot = assets / "freeze-1.png"
        snapshot.write_bytes(b"existing-snapshot")
        preserved = {snapshot: digest(snapshot)}
    else:
        output.symlink_to(run / "absent-target.mp4")
        preserved = {}
    def forbidden(*_, **__):
        raise AssertionError("Fresh-output guard must run before invoking any encoder")
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(FileExistsError, match="fresh output/provenance/assets"):
        render(run, output)
    assert all(digest(path) == original for path, original in preserved.items())
    assert not list(run.glob(f".{output.stem}.native-stage-*"))
    if existing == "dangling_video_symlink":
        assert output.is_symlink() and not (run / "absent-target.mp4").exists()


def test_staged_provenance_write_failure_publishes_nothing_and_preserves_prior_artifact(native_run, monkeypatch):
    run, _ = native_run
    prior = _prior_artifact(run)
    output = run / "new-native-demo.mp4"
    _fake_encoding(monkeypatch)
    original_write = Path.write_text
    def failed_proof(path, *args, **kwargs):
        if path.name == "provenance.json" and ".native-stage-" in str(path.parent):
            raise OSError("injected staged provenance write failure")
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", failed_proof)
    with pytest.raises(OSError, match="staged provenance write failure"):
        render(run, output)
    _assert_no_partial_demo(run, output)
    assert all(digest(path) == original for path, original in prior.items())


def test_provenance_publication_failure_rolls_back_new_video_and_assets(native_run, monkeypatch):
    run, _ = native_run
    prior = _prior_artifact(run)
    output = run / "new-native-demo.mp4"
    module = _fake_encoding(monkeypatch)
    original_link = module.os.link
    publication_checks = []
    def failed_publish(source, destination, *args, **kwargs):
        source, destination = Path(source), Path(destination)
        staging = source.parent if source.name in {"video.mp4", "provenance.json"} else source.parent.parent
        proof = staging / "provenance.json"
        assert proof.is_file(), "Complete provenance must exist before the first published asset"
        publication_checks.append(destination)
        if destination == output.with_suffix(".provenance.json"):
            assert output.is_file()
            assert json.loads(proof.read_text())["output_sha256"] == digest(output)
            raise OSError("injected provenance publication failure")
        return original_link(source, destination, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", failed_publish)
    with pytest.raises(OSError, match="provenance publication failure"):
        render(run, output)
    assert publication_checks[-1] == output.with_suffix(".provenance.json")
    _assert_no_partial_demo(run, output)
    assert all(digest(path) == original for path, original in prior.items())


def test_encoding_failure_leaves_no_published_video_proof_or_assets(native_run, monkeypatch):
    run, _ = native_run
    prior = _prior_artifact(run)
    output = run / "new-native-demo.mp4"
    module = _fake_encoding(monkeypatch)
    def failed_encode(command, **_):
        Path(command[-1]).write_bytes(b"partial encoding")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(module.subprocess, "run", failed_encode)
    with pytest.raises(subprocess.CalledProcessError):
        render(run, output)
    _assert_no_partial_demo(run, output)
    assert all(digest(path) == original for path, original in prior.items())


def test_concurrent_foreign_video_is_never_overwritten_or_removed(native_run, monkeypatch):
    run, _ = native_run
    output = run / "new-native-demo.mp4"
    module = _fake_encoding(monkeypatch)
    original_link = module.os.link
    def publish_with_foreign_writer(source, destination, *args, **kwargs):
        if Path(destination) == output:
            output.write_bytes(b"concurrently-created-foreign-video")
        return original_link(source, destination, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", publish_with_foreign_writer)
    with pytest.raises(FileExistsError):
        render(run, output)
    assert output.read_bytes() == b"concurrently-created-foreign-video"
    assert not output.with_suffix(".provenance.json").exists()
    assert not output.with_name(output.stem + ".native-reading-assets").exists()
    assert not list(run.glob(f".{output.stem}.native-stage-*"))


def test_rollback_preserves_foreign_replacement_of_just_published_video(native_run, monkeypatch):
    run, _ = native_run
    output = run / "new-native-demo.mp4"
    module = _fake_encoding(monkeypatch)
    original_link = module.os.link
    def replace_then_fail(source, destination, *args, **kwargs):
        if Path(destination) == output.with_suffix(".provenance.json"):
            output.unlink()
            output.write_bytes(b"foreign-replacement")
            raise OSError("injected failure after foreign replacement")
        return original_link(source, destination, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", replace_then_fail)
    with pytest.raises(OSError, match="foreign replacement"):
        render(run, output)
    assert output.read_bytes() == b"foreign-replacement"
    assert not output.with_suffix(".provenance.json").exists()
    assert not output.with_name(output.stem + ".native-reading-assets").exists()
    assert not list(run.glob(f".{output.stem}.native-stage-*"))


def _set_read_values(data, name, first, second=None):
    """Change synthetic model outputs and their corresponding saved projections."""
    page = data["pages"][0]
    second = first if second is None else second
    page["result"]["fields"][name]["raw_value"] = first
    page["result"]["fields"][name]["normalized_value"] = None
    for view, raw in zip(page["image_recognition"]["views"], (first, second), strict=True):
        view["reading"]["fields"][name]["raw_value"] = raw
        view["document"]["fields"][name]["raw_value"] = raw
        view["document"]["fields"][name]["normalized_value"] = None


def test_empty_model_quotes_remain_valid_with_actual_frame_panel_provenance(native_run):
    run, data = native_run
    page = data["pages"][0]
    for document in [page["result"], *(view["document"] for view in page["image_recognition"]["views"])]:
        for item in [*document["classification_evidence"], *(e for value in document["fields"].values() for e in value["evidence"])]:
            item["text"] = ""
    save(run, data)
    plan = build_plan(run)
    assert plan["semantic_validation"] == "model"
    assert len(plan["selected_views"]) == 2
    assert all(not item["text"] for view in plan["selected_views"] for value in view["fields"].values()
               for item in value["evidence"])


@pytest.mark.parametrize("name,raw,quote", [
    ("supplier", 'ООО «Береза»', "Грузополучатель: другое название"),
    ("consignee", 'ООО «Ива»', "Неполная подпись без значения"),
    ("document_number", "2020", "Номер документа: 0042 от 15.02.2020"),
    ("document_number", "№", ""),
    ("document_date", "31.02.2020", "Дата составления: 01.03.2020"),
    ("document_date", "02.06.99", "Заказ (заявка) от 02.06.99"),
])
def test_model_values_are_displayed_without_caption_calendar_or_role_judgment(native_run, name, raw, quote):
    run, data = native_run
    _set_read_values(data, name, raw)
    page = data["pages"][0]
    for document in [page["result"], *(view["document"] for view in page["image_recognition"]["views"])]:
        for item in document["fields"][name]["evidence"]:
            item["text"] = quote
    save(run, data)
    plan = build_plan(run)
    assert all(view["fields"][name]["raw_value"] == raw for view in plan["selected_views"])
    assert all(set(view["fields"][name]["confirmed_frame_ids"]) == {"frame-1", "frame-2"}
               for view in plan["selected_views"])


@pytest.mark.parametrize("name,first,second", [
    ("supplier", 'ООО «Береза»', '  ооо «БЕРЕЗА»\n'),
    ("document_number", "0042", "００４２"),
    ("document_date", "3 мая 2022 г.", "3\tмая\n2022 г."),
])
def test_unicode_case_whitespace_agreement_preserves_actual_each_frame_raw(native_run, name, first, second):
    run, data = native_run
    _set_read_values(data, name, first, second)
    save(run, data)
    plan = build_plan(run)
    assert plan["selected_views"][0]["fields"][name]["raw_value"] == first
    assert plan["selected_views"][1]["fields"][name]["raw_value"] == second
    assert plan["selected_views"][1]["fields"][name]["fused_raw_value"] == first


@pytest.mark.parametrize("name,first,second", [
    ("supplier", 'ООО «Береза»', 'Общество с ограниченной ответственностью «Береза»'),
    ("document_date", "15.02.2020", "15 февраля 2020"),
    ("document_date", "31.02.2020", "30.02.2020"),
    ("document_date", "02.06.99", "03.06.99"),
    ("document_number", "12/3", "123"),
    ("document_number", "А-42", "A-42"),
])
def test_different_literal_readings_cannot_confirm_fused_value_through_field_specific_normalization(native_run, name, first, second):
    run, data = native_run
    _set_read_values(data, name, first, second)
    save(run, data)
    with pytest.raises(ValueError, match=f"{name} lacks two distinct"):
        build_plan(run)


@pytest.mark.parametrize("change", ["raw", "status", "type", "panel_id", "panel_frame", "missing_field_panels"])
def test_rendered_value_is_bound_to_original_model_status_and_supplied_panel_ids(native_run, change):
    run, data = native_run
    view = data["pages"][0]["image_recognition"]["views"][0]
    proposal = view["reading"]["fields"]["supplier"]
    if change == "raw":
        view["document"]["fields"]["supplier"]["raw_value"] = "Hand edited value"
    elif change == "status":
        proposal["status"] = "unreadable"
    elif change == "type":
        view["reading"]["document_type"] = "ttn"
    elif change == "panel_id":
        view["pixel_proof"]["panels"]["context_r270"]["panel_id"] = "another-panel"
    elif change == "panel_frame":
        view["pixel_proof"]["panels"]["context_r270"]["frame_id"] = "foreign-frame"
    else:
        proposal["panel_ids"] = []
    save(run, data)
    with pytest.raises(ValueError):
        build_plan(run)


def test_empty_quotes_still_need_two_distinct_actual_frames(native_run):
    run, data = native_run
    page = data["pages"][0]
    page["result"]["fields"]["supplier"]["evidence"] = [evidence("frame-1", "")] * 3
    for view in page["image_recognition"]["views"]:
        view["document"]["fields"]["supplier"]["evidence"][0]["text"] = ""
    save(run, data)
    with pytest.raises(ValueError, match="supplier lacks two distinct"):
        build_plan(run)


def _add_nonselected_view(run, data, *, kind="unknown", errors=None):
    page = data["pages"][0]
    frame = deepcopy(page["frames"][0])
    view = deepcopy(page["image_recognition"]["views"][0])
    path = run / "earlier-frame.png"
    shutil.copyfile(frame["path"], path)
    frame.update(frame_id="earlier-frame", path=str(path), timestamp_s=.2)
    view.update(frame_id="earlier-frame", frame_path=str(path), timestamp_s=.2)
    view["pixel_proof"].update(frame_id="earlier-frame", source_path=str(path), timestamp_s=.2,
                              source_sha256=digest(path))
    for panel in view["pixel_proof"]["panels"].values():
        panel["frame_id"] = "earlier-frame"
    view["reading"]["document_type"] = kind
    view["document"]["document_type"] = kind
    view["document"]["errors"] = errors or []
    for item in [*view["document"]["classification_evidence"], *(e for value in view["document"]["fields"].values()
                                                               for e in value["evidence"])]:
        item["frame_id"] = "earlier-frame"
    if kind == "unknown":
        view["document"]["fields"] = {}
        view["reading"]["fields"] = {}
    page["frames"].insert(0, frame)
    page["image_recognition"]["views"].insert(0, view)
    page["result"]["classification_evidence"].extend(deepcopy(view["document"]["classification_evidence"]))
    page["result"]["errors"].extend(view["document"]["errors"])
    return view


@pytest.mark.parametrize("kind,errors", [("unknown", None), ("torg12", ["structural model output error"])])
def test_nonselected_unknown_or_errored_view_does_not_veto_two_proved_model_readings(native_run, kind, errors):
    run, data = native_run
    _add_nonselected_view(run, data, kind=kind, errors=errors)
    save(run, data)
    plan = build_plan(run)
    assert [view["frame_id"] for view in plan["selected_views"]] == ["frame-1", "frame-2"]
    assert all(set(ids) == {"frame-1", "frame-2"} for ids in plan["field_support"].values())


@pytest.mark.parametrize("change", ["source", "page", "hash"])
def test_excluded_unknown_views_still_have_strict_source_provenance_checks(native_run, change):
    run, data = native_run
    view = _add_nonselected_view(run, data)
    if change == "source":
        view["document"]["source_id"] = "foreign-source"
    elif change == "page":
        view["document"]["page_id"] = "foreign-leaf"
    else:
        view["pixel_proof"]["source_sha256"] = "0" * 64
    save(run, data)
    with pytest.raises(ValueError):
        build_plan(run)


@pytest.mark.parametrize("kind,errors", [("unknown", None), ("torg12", ["structural model output error"])])
def test_unknown_or_errored_view_cannot_supply_second_confirmed_type_reading(native_run, kind, errors):
    run, data = native_run
    view = _add_nonselected_view(run, data, kind=kind, errors=errors)
    data["pages"][0]["result"]["classification_evidence"] = [
        deepcopy(data["pages"][0]["image_recognition"]["views"][1]["document"]["classification_evidence"][0]),
        deepcopy(view["document"]["classification_evidence"][0]),
    ]
    save(run, data)
    with pytest.raises(ValueError, match="two distinct actual own-heading"):
        build_plan(run)
