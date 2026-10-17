from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from sever import ocr, video


@pytest.fixture(autouse=True)
def _isolate_configured_tesseract_models(monkeypatch):
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)


def _paper(number: int) -> Image.Image:
    image = Image.new("RGB", (320, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 45), f"INVOICE {number} / 2026", fill="black")
    rng = np.random.default_rng(number)
    for row in range(12):
        y = 95 + row * 25
        draw.line((20, y, 300, y), fill="black")
        for column in range(3):
            x = 25 + column * 85
            text = f"{number}-{rng.integers(100000, 999999)}"
            draw.text((x, y + 4), text, fill="black")
    return image


def _shift(image: Image.Image) -> Image.Image:
    shifted = Image.new("RGB", image.size, "white")
    shifted.paste(image, (5, 4))
    return shifted


def test_showinfo_retains_signed_pts_and_time_base():
    entries, time_base = video._showinfo("""
[Parsed_showinfo_2 @ x] config in time_base: 1/90000, frame_rate: 30/1
[Parsed_showinfo_2 @ x] n: 0 pts: -3000 pts_time:-0.0333333 fmt:rgb
[Parsed_showinfo_2 @ x] n: 1 pts: 42000 pts_time:0.466667 fmt:rgb
""")
    assert time_base == "1/90000"
    assert [e["source_pts"] for e in entries] == [-3000, 42000]


def test_ffmpeg_exact_pts_selection_over_150_frames(tmp_path):
    """Exercise FFmpeg's actual expression parser, not just the string shape."""
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    source = tmp_path / "many-frames.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=80x60:rate=100:duration=4.5", "-c:v", "ffv1", str(source)], check=True)
    expected = [20 + 20 * index for index in range(211)]
    expression = video._pts_selection_expression(expected)
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-copyts", "-i", str(source),
                             "-an", "-vf", f"select='{expression}',showinfo", "-fps_mode", "vfr",
                             "-f", "null", "-"], capture_output=True, text=True, check=True)
    actual, time_base = video._showinfo(result.stderr)
    assert time_base == "1/1000"
    assert [frame["source_pts"] for frame in actual] == expected
    # No decoded image sequence is written by this validation.
    assert not list(tmp_path.glob("*.png"))


def test_stable_segmentation_ignores_turning_and_camera_shift():
    a, b = _paper(1), _paper(2)
    turn = Image.new("RGB", a.size, "#b5b5b5")
    images = [a, _shift(a), a, turn, b, _shift(b), b]
    features = [video._features(i) for i in images]
    assert video._pair_evidence(features[0], features[1])["same_page"]
    assert not video._pair_evidence(features[0], features[4])["same_page"]
    groups, discarded = video._segment_samples(features)
    assert len(groups) == 2
    assert groups[0] == [0, 1, 2]
    assert groups[1] == [4, 5, 6]
    assert discarded == [3]


def test_first_two_real_fixtures_are_different_pages():
    root = Path(__file__).resolve().parents[1] / "analysis/sample-01/selected-frames"
    if not (root / "page-01.jpg").exists():
        pytest.skip("Local video fixtures are not distributed with the package")
    features = []
    for name in ("page-01.jpg", "page-02.jpg"):
        with Image.open(root / name) as image:
            image.thumbnail((384, 700))
            features.append(video._features(image))
    assert not video._pair_evidence(*features)["same_page"]


@pytest.mark.parametrize("degrees,expected_size", [(0, (100, 200)), (90, (200, 100)), (180, (100, 200)), (270, (200, 100))])
def test_rotation_provenance_maps_all_image_corners(degrees, expected_size):
    matrix = ocr._rotation_matrix(100, 200, degrees)
    corners = np.asarray([[0, 0, 1], [100, 0, 1], [100, 200, 1], [0, 200, 1]])
    transformed = (matrix @ corners.T).T
    assert tuple(transformed[:, :2].max(axis=0)) == expected_size
    np.testing.assert_allclose((np.linalg.inv(matrix) @ transformed.T).T, corners)


def test_orientation_layout_distinguishes_upright_from_upside_down():
    header = {"text": "СЧЕТ-ФАКТУРА № 25", "confidence": .5, "bbox": [.1, .1, .7, .05]}
    top = ocr._orientation_score({"regions": [header]}, (1000, 700))["score"]
    header["bbox"][1] = .85
    bottom = ocr._orientation_score({"regions": [header]}, (1000, 700))["score"]
    assert top > bottom


def test_read_image_rotation_cache_and_source_boxes(tmp_path, monkeypatch):
    source = tmp_path / "sideways.png"
    Image.new("RGB", (100, 200), "white").save(source)
    calls = []
    def backend(path, cache_dir):
        calls.append(path.name)
        if "probe-" in path.name:
            degrees = int(path.stem.split("-")[-1])
            box = [.1, .1 if degrees == 90 else .85, .7, .05]
            if degrees in (0, 180):
                box = [.1, .1, .05, .7]
        else:
            box = [.1, .1, .7, .05]
        return {"engine": "test_local", "text": "СЧЕТ-ФАКТУРА 25", "regions": [{"text": "СЧЕТ-ФАКТУРА 25", "confidence": .5, "bbox": box}]}
    monkeypatch.setattr(ocr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ocr, "_tesseract_result", backend)
    result = ocr.read_image(source, tmp_path / "cache")
    assert result["rotation_degrees"] == 90
    assert result["processing"]["ocr_pixel_size"] == [200, 100]
    assert len(calls) == 5
    np.testing.assert_allclose(result["regions"][0]["source_bbox_pixels"], [10, 40, 5, 140])
    again = ocr.read_image(source, tmp_path / "cache")
    assert again["processing"]["cache_hit"]
    assert len(calls) == 5
    enhanced = ocr.read_image(source, tmp_path / "cache", enhance_header=True)
    assert len(calls) == 8  # Header variant reuses four persisted small probe responses.
    header_region = next(r for r in enhanced["regions"] if r["processing_pass"] == "header")
    np.testing.assert_allclose(header_region["source_bbox_pixels"], [6, 40, 3, 140])
    assert len(enhanced["processing"]["enhancement_passes"]) == 2


def test_jpeg_storage_keeps_png_cache_and_source_provenance(tmp_path, monkeypatch):
    """Only new JPEG reads run OCR; old PNG artifacts retain their exact key."""
    source = tmp_path / "sideways.png"
    image = Image.new("RGB", (100, 200), "white")
    ImageDraw.Draw(image).text((5, 5), "Source detail", fill="black")
    image.save(source)
    source_bytes = source.read_bytes()
    cache = tmp_path / "cache"
    calls = []

    def backend(path, cache_dir):
        calls.append(path.name)
        degrees = int(path.stem.split("-")[-1]) if "probe-" in path.name else 90
        box = [.1, .1 if degrees == 90 else .85, .7, .05]
        if degrees in (0, 180):
            box = [.1, .1, .05, .7]
        return {"engine": "test_local", "text": "СЧЕТ-ФАКТУРА 25",
                "regions": [{"text": "СЧЕТ-ФАКТУРА 25", "confidence": .5, "bbox": box}]}

    monkeypatch.setattr(ocr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ocr, "_tesseract_result", backend)
    png = ocr.read_image(source, cache, enhance_header=True)
    assert len(calls) == 7
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    identity = f"tesseract_rus_eng::{source_hash}:header=True"
    historical_key = hashlib.sha256(f"{ocr._VERSION}:{identity}".encode()).hexdigest()
    assert Path(png["processing"]["ocr_image_path"]).name == f"{historical_key}-oriented.png"
    # Simulate an existing pre-option cache without the new metadata.
    existing = json.loads((cache / f"{historical_key}.json").read_text())
    for container in [existing["processing"], *existing["processing"]["enhancement_passes"]]:
        for key in ("storage_format", "mime_type", "jpeg_quality", "jpeg_subsampling"):
            container.pop(key)
    (cache / f"{historical_key}.json").write_text(json.dumps(existing))
    jpeg = ocr.read_image(source, cache, enhance_header=True, storage_format="jpeg")
    assert len(calls) == 10  # Shared orientation probes; independent full image and two crops.
    assert not jpeg["processing"]["cache_hit"]
    assert jpeg["processing"]["orientation_probe_cache_hit"]
    assert jpeg["rotation_degrees"] == png["rotation_degrees"] == 90
    assert jpeg["processing"]["ocr_image_path"] != png["processing"]["ocr_image_path"]
    assert jpeg["processing"]["source_sha256"] == png["processing"]["source_sha256"] == source_hash
    assert jpeg["processing"]["source_to_ocr_matrix"] == png["processing"]["source_to_ocr_matrix"]
    assert jpeg["processing"]["ocr_to_source_matrix"] == png["processing"]["ocr_to_source_matrix"]
    assert jpeg["regions"] == png["regions"]
    assert "JPEG_artifacts_may_reduce_OCR_readability" in jpeg["processing"]["limitations"]
    for entry in [jpeg["processing"], *jpeg["processing"]["enhancement_passes"]]:
        artifact = Path(entry["ocr_image_path"])
        assert artifact.suffix == ".jpg"
        assert entry["storage_format"] == "jpeg"
        assert entry["mime_type"] == "image/jpeg"
        assert entry["jpeg_quality"] == 98
        assert entry["jpeg_subsampling"] == 0
        with Image.open(artifact) as decoded:
            assert decoded.format == "JPEG"
    again_jpeg = ocr.read_image(source, cache, enhance_header=True, storage_format="jpeg")
    again_png = ocr.read_image(source, cache, enhance_header=True)
    assert len(calls) == 10
    assert again_jpeg["processing"]["cache_hit"] and again_png["processing"]["cache_hit"]
    assert again_png["processing"]["storage_format"] == "png"
    assert again_png["processing"]["mime_type"] == "image/png"
    assert again_png["processing"]["ocr_image_path"] == png["processing"]["ocr_image_path"]
    assert source.read_bytes() == source_bytes


def test_invalid_ocr_storage_format_has_no_writes(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="storage_format"):
        ocr.read_image(tmp_path / "absent.png", cache, storage_format="webp")
    assert not cache.exists()


def test_overlapping_title_cannot_outweigh_the_main_sideways_document():
    # Large title on a folded neighboring leaf is horizontal, but most text
    # belongs to a sideways UPD below it, as in the actual last video5 frame.
    wrong = [{"text": "ТОВАРНО-ТРАНСПОРТНАЯ НАКЛАДНАЯ", "confidence": 1., "bbox": [0., .4, .25, .055]}]
    wrong += [{"text": "Товар получен результаты работ приняты " * 2, "confidence": 1.,
               "bbox": [.3 + i * .02, .3, .012, .4]} for i in range(12)]
    upright = [{"text": "Универсальный передаточный документ № 777 от 1 января 2026", "confidence": 1.,
                "bbox": [.1, .12, .75, .03]}]
    upright += [{"text": "Товар получен результаты работ приняты " * 2, "confidence": 1.,
                 "bbox": [.2, .3 + i * .02, .7, .012]} for i in range(12)]
    bad = ocr._orientation_score({"regions": wrong}, (900, 1600))
    good = ocr._orientation_score({"regions": upright}, (1600, 900))
    assert bad["header_layout_bonus"] > 100
    assert bad["layout_anchor_dominance_gate"] == 0
    assert good["score"] > bad["score"]
    upside_down = [{**r, "bbox": [r["bbox"][0], 1 - r["bbox"][1] - r["bbox"][3], *r["bbox"][2:]]} for r in upright]
    assert good["score"] > ocr._orientation_score({"regions": upside_down}, (1600, 900))["score"]


def test_version4_cache_reprobes_small_images_but_reuses_full_ocr(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 200), "white").save(source)
    cache = tmp_path / "cache"
    calls = []
    def backend(path, cache_dir):
        calls.append(path.name)
        degrees = int(path.stem.split("-")[-1]) if "probe-" in path.name else 90
        box = [.1, .1 if degrees == 90 else .85, .7, .05]
        if degrees in (0, 180):
            box = [.1, .1, .05, .7]
        return {"engine": "test_local", "text": "СЧЕТ-ФАКТУРА 25", "regions": [{"text": "СЧЕТ-ФАКТУРА 25", "confidence": .5, "bbox": box}]}
    monkeypatch.setattr(ocr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ocr, "_tesseract_result", backend)
    original = ocr.read_image(source, cache, enhance_header=True)
    assert len(calls) == 7
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = f"tesseract_rus_eng::{source_hash}:header=True"
    current_key = hashlib.sha256(f"{ocr._VERSION}:{identity}".encode()).hexdigest()
    old_key = hashlib.sha256(f"4:{identity}".encode()).hexdigest()
    (cache / f"{old_key}.json").write_text(json.dumps(original))
    (cache / f"{current_key}.json").unlink()
    for path in cache.glob("*-orientation-probes.json"):
        path.unlink()  # Version4 did not keep these responses.
    migrated = ocr.read_image(source, cache, enhance_header=True)
    assert len(calls) == 11  # Four small probes, zero full-resolution/crop OCR.
    assert migrated["processing"]["cache_migration"]["from_version"] == 4
    assert migrated["processing"]["cache_version"] == ocr._VERSION
    assert migrated["regions"] == original["regions"]
    assert migrated["processing"]["ocr_image_path"] == original["processing"]["ocr_image_path"]
    assert migrated["processing"]["source_to_ocr_matrix"] == original["processing"]["source_to_ocr_matrix"]


def test_mixed_case_form_title_beats_small_upside_down_ocr_advantage():
    rows = [{"text": "Продукция количество цена сумма товар без налога " * 3, "confidence": .9,
             "bbox": [.1, .35 + i * .025, .8, .009]} for i in range(12)]
    upright = {"regions": [{"text": "Товарная накладная", "confidence": .9,
                             "bbox": [.27, .263, .121, .009]}, *rows]}
    inverted = {"regions": [{"text": "Товарная накладная", "confidence": 1.,
                              "bbox": [.60, .728, .123, .01]},
                             {"text": "Товарная накладная имеет приложение на", "confidence": 1.,
                              "bbox": [.59, .559, .19, .009]},
                             *[{**r, "confidence": .905} for r in rows]]}
    normal = ocr._orientation_score(upright, (1131, 1600))
    reverse = ocr._orientation_score(inverted, (1131, 1600))
    assert reverse["horizontal_character_score"] > normal["horizontal_character_score"]
    assert normal["document_title_layout_bonus"] > 80
    assert reverse["document_title_layout_bonus"] == 0
    assert normal["score"] > reverse["score"]


def test_video_retains_bounded_exact_pts_originals(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    for index in range(16):
        _paper(1 if index < 8 else 2).save(tmp_path / f"input-{index:03}.png")
    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "4", "-i", str(tmp_path / "input-%03d.png"),
                    "-c:v", "ffv1", str(source)], check=True)
    pages = video.analyze_video(source, tmp_path / "result", sample_interval=.25)
    assert len(pages) == 2
    assert sum(len(p["frames"]) for p in pages) <= 6
    assert all(1 <= len(p["frames"]) <= 3 for p in pages)
    assert all(any(f["frame_id"] == p["selected_frame_id"] for f in p["frames"]) for p in pages)
    assert all(f["pixel_size"] == [320, 560] for p in pages for f in p["frames"])
    assert all(f["source_pts_time_s"] == pytest.approx(f["timestamp_s"]) for p in pages for f in p["frames"])
    assert len(list((tmp_path / "result/frames").glob("*.png"))) <= 6
    assert video.analyze_video(source, tmp_path / "result", sample_interval=.25) == pages
    capped = video.analyze_video(source, tmp_path / "capped", sample_interval=.25, max_pages=1)
    assert len(capped) == 1
    assert capped[0]["end_s"] == pages[0]["end_s"]
    assert capped[0]["end_s"] < 4


def test_large_uppercase_title_outweighs_repeated_body_mentions():
    regions = [{"text": "ТОВАРНАЯ НАКЛАДНАЯ", "confidence": .5, "bbox": [.1, .3, .7, .04]}]
    regions += [{"text": "С товаром переданы документы: удостоверение о качестве", "confidence": .5,
                 "bbox": [.1, .6 + i * .01, .7, .02]} for i in range(8)]
    upright = ocr._orientation_score({"regions": regions}, (1000, 1600))["score"]
    inverted = [{**r, "bbox": [r["bbox"][0], 1 - r["bbox"][1] - r["bbox"][3], *r["bbox"][2:]]} for r in regions]
    assert upright > ocr._orientation_score({"regions": inverted}, (1000, 1600))["score"]


def test_quality_visibility_and_unavailable_glare_or_hand_confirmation():
    page = Image.new("RGB", (400, 600), "#303030")
    draw = ImageDraw.Draw(page)
    draw.polygon([(50, 40), (360, 70), (350, 560), (30, 570)], fill="white")
    quality = video.analyze_image_quality(page)
    assert quality["full_page_visible"] is True
    assert quality["paper_geometry"]["perspective_indicator"] is not None
    assert quality["exposure"]["glare_confirmed"] is None
    assert quality["possible_hand_occlusion"]["hand_occlusion_confirmed"] is None
    clipped = video.analyze_image_quality(Image.new("RGB", (400, 600), "white"))
    assert clipped["full_page_visible"] is False
    assert clipped["paper_geometry"]["clipped_edges"]


def test_adaptive_number_row_crop_does_not_use_the_number_value():
    regions = [{"text": "Счет-фактура Nº", "bbox": [.18, .072, .05, .016]}]
    boxes = ocr._header_line_boxes(regions, (3840, 2160))
    assert len(boxes) == 1
    name, (left, top, right, bottom) = boxes[0]
    assert name == "header_number_line_0"
    assert left < .18 * 3840 < .30 * 3840 < right
    assert top < .08 * 2160 < bottom
    # Already recognized number lines are not treated as empty number labels.
    assert ocr._header_line_boxes([{**regions[0], "text": "Счет-фактура Nº 777"}], (3840, 2160)) == []
