from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from sever import ocr
from sever.classifier import _own_forms
from sever.vision import _role_sources, _seller_address_literal


@pytest.fixture(autouse=True)
def _isolate_configured_tesseract_models(monkeypatch):
    # A deployed service's best-model profile must not alter fixtures that
    # explicitly exercise default cache identities. Profile tests set their
    # own actual temporary model directory below.
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)


def _tsv(words):
    stream = io.StringIO()
    columns = ["level", "page_num", "block_num", "par_num", "line_num", "word_num",
               "left", "top", "width", "height", "conf", "text"]
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", quoting=csv.QUOTE_NONE,
                            quotechar=None, lineterminator="\n")
    writer.writeheader()
    for word in words:
        writer.writerow({"level": 5, "page_num": 1, "block_num": 1, "par_num": 1,
                         "line_num": 1, "height": 20, "conf": 90, **word})
    return stream.getvalue()


def test_compound_heading_has_one_literal_line_and_actual_word_box_union():
    words = [{"word_num": 1, "left": 10, "top": 12, "width": 170, "text": "Универсальный"},
             {"word_num": 2, "left": 192, "top": 10, "width": 165, "text": "передаточный"},
             {"word_num": 3, "left": 369, "top": 11, "width": 130, "text": "документ"}]
    assert all(not _own_forms(word["text"]) for word in words)
    regions = ocr._tesseract_line_regions(_tsv(words), (1000, 500))
    assert len(regions) == 1
    region = regions[0]
    assert region["text"] == "Универсальный передаточный документ"
    assert _own_forms(region["text"]) == {"upd"}
    assert region["bbox"] == pytest.approx([.01, .02, .489, .044])
    assert region["engine_line_id"] == [1, 1, 1, 1]
    assert region["engine_word_observations"][1]["bbox_pixels"] == [192, 10, 165, 20]
    assert region["engine_word_observations"][1]["text"] == "передаточный"


def test_caption_values_and_separate_columns_do_not_become_one_party_band():
    words = [{"word_num": 1, "left": 10, "top": 50, "width": 58, "text": "Адрес:"},
             {"word_num": 2, "left": 75, "top": 50, "width": 80, "text": "765432,"},
             {"word_num": 3, "left": 163, "top": 50, "width": 120, "text": "Новый"},
             {"word_num": 4, "left": 291, "top": 50, "width": 70, "text": "город"},
             {"word_num": 5, "left": 600, "top": 50, "width": 98, "text": "Покупатель:"},
             {"word_num": 6, "left": 706, "top": 50, "width": 45, "text": "ООО"},
             {"word_num": 7, "left": 760, "top": 50, "width": 90, "text": '"Пример"'}]
    regions = ocr._tesseract_line_regions(_tsv(words), (1000, 500))
    assert [region["text"] for region in regions] == ["Адрес:", "765432, Новый город",
                                                    "Покупатель:", 'ООО "Пример"']
    assert regions[1]["bbox"] == pytest.approx([.075, .1, .286, .04])
    lookup = {f"r{i}": {**region, "processing_pass": "full_image"} for i, region in enumerate(regions)}
    seller = {"ids": ["r0", "r1"]}
    assert ocr._tesseract_line_regions(_tsv(words), (1000, 500)) == regions
    literal = _seller_address_literal(seller, lookup)
    assert literal == ("765432, Новый город", ["r0", "r1"])


def test_reference_words_join_without_inventing_same_page_values():
    words = [{"word_num": 1, "left": 10, "top": 100, "width": 160,
              "text": "Грузоотправитель:"},
             {"word_num": 2, "left": 182, "top": 100, "width": 25, "text": "он"},
             {"word_num": 3, "left": 215, "top": 100, "width": 25, "text": "же"}]
    regions = ocr._tesseract_line_regions(_tsv(words), (1000, 1000))
    assert [r["text"] for r in regions] == ["Грузоотправитель:", "он же"]
    panels = _role_sources({"regions": regions}, "upd")
    assert panels["consignor"]["explicit_seller_reference"] is True
    assert "seller" not in panels


def test_actual_tsv_line_ids_confidence_and_literal_alternatives_are_preserved():
    rows = [{"word_num": 2, "left": 80, "top": 12, "width": 70, "conf": 40, "text": "later"},
            {"word_num": 1, "left": 10, "top": 10, "width": 58, "conf": 100, "text": "first"},
            {"word_num": 1, "left": 10, "top": 50, "width": 150, "line_num": 2,
             "text": "OCR-01"},
            {"word_num": 1, "left": 10, "top": 10, "width": 150, "block_num": 2,
             "text": "OCR-O1"},
            {"word_num": 3, "left": 165, "top": 10, "width": 80, "conf": -1, "text": "discard"},
            {"word_num": 4, "left": 250, "top": 10, "width": 80, "level": 4, "text": "not-word"}]
    regions = ocr._tesseract_line_regions(_tsv(rows), (1000, 500))
    assert [region["text"] for region in regions] == ["first later", "OCR-01", "OCR-O1"]
    assert regions[0]["confidence"] == pytest.approx(.7)
    assert [region["region_index"] for region in regions] == [0, 1, 2]
    assert regions[1]["engine_line_id"] != regions[2]["engine_line_id"]


def test_ocr_company_quote_is_literal_and_does_not_swallow_following_tsv_rows():
    rows = [{"word_num": 1, "left": 10, "top": 10, "width": 60, "text": "ООО"},
            {"word_num": 2, "left": 78, "top": 10, "width": 130, "text": '"Новая'},
            {"word_num": 3, "left": 216, "top": 10, "width": 130, "text": 'компания"'},
            {"word_num": 1, "left": 10, "top": 50, "width": 130, "line_num": 2, "text": "765432,"},
            {"word_num": 2, "left": 148, "top": 50, "width": 130, "line_num": 2, "text": "Адрес"}]
    tsv = _tsv(rows)
    assert '\t"Новая\n' in tsv  # Same unescaped field format produced by Tesseract.
    regions = ocr._tesseract_line_regions(tsv, (1000, 500))
    assert [region["text"] for region in regions] == ['ООО "Новая компания"', "765432, Адрес"]
    assert all("\t" not in region["text"] and "\n" not in region["text"] for region in regions)
    assert sum(len(region["engine_word_observations"]) for region in regions) == 5


def test_tesseract_backend_checks_languages_and_exports_line_provenance(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    Image.new("RGB", (1000, 500), "white").save(image)
    tsv = _tsv([{"word_num": 1, "left": 10, "top": 10, "width": 130, "text": "Транспортная"},
                {"word_num": 2, "left": 150, "top": 10, "width": 120, "text": "накладная"}])
    calls = []
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/usr/bin/tesseract")

    def run(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "List of available languages:\nrus\neng\n"
                                           if "--list-langs" in arguments else tsv, "")

    monkeypatch.setattr(ocr.subprocess, "run", run)
    result = ocr._tesseract_result(image, tmp_path)
    assert calls[-1] == ["/usr/bin/tesseract", str(image), "stdout", "-l", "rus+eng", "--psm", "3", "tsv"]
    assert result["text"] == "Транспортная накладная"
    assert result["region_granularity"] == "line_segments"
    assert ocr._compatible_tesseract_layout(result)
    assert not ocr._compatible_tesseract_layout({"engine": "tesseract_rus_eng", "regions": []})


def test_word_only_linux_image_and_orientation_caches_are_not_reused(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (100, 200), "white").save(source)
    cache = tmp_path / "cache"
    calls = []

    def backend(path, cache_dir):
        calls.append(path.name)
        degrees = int(path.stem.split("-")[-1]) if "probe-" in path.name else 90
        box = [.1, .1 if degrees == 90 else .85, .7, .05]
        if degrees in (0, 180):
            box = [.1, .1, .05, .7]
        return {"engine": "tesseract_rus_eng", "tesseract_layout_version": ocr._TESSERACT_LAYOUT_VERSION,
                "region_granularity": "line_segments", "text": "СЧЕТ-ФАКТУРА 31",
                "regions": [{"text": "СЧЕТ-ФАКТУРА 31", "confidence": .9, "bbox": box}]}

    monkeypatch.setattr(ocr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ocr, "_tesseract_result", backend)
    first = ocr.read_image(source, cache)
    assert len(calls) == 5
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = f"tesseract_rus_eng::{source_hash}:header=False"
    current = cache / (hashlib.sha256(f"{ocr._VERSION}:{identity}".encode()).hexdigest() + ".json")
    legacy = cache / (hashlib.sha256(f"6:{identity}".encode()).hexdigest() + ".json")
    first.pop("tesseract_layout_version")
    first.pop("region_granularity")
    legacy.write_text(json.dumps(first))
    current.unlink()
    for artifact in cache.glob("*-orientation-probes.json"):
        data = json.loads(artifact.read_text())
        for response in data["responses"]:
            response["result"].pop("tesseract_layout_version")
            response["result"].pop("region_granularity")
        artifact.write_text(json.dumps(data))
    fresh = ocr.read_image(source, cache)
    assert len(calls) == 10  # Four new probes and actual full-resolution reading.
    assert not fresh["processing"]["cache_hit"]
    assert not fresh["processing"]["orientation_probe_cache_hit"]
    assert "cache_migration" not in fresh["processing"]
    assert legacy.is_file()  # Historical evidence is retained.
    again = ocr.read_image(source, cache)
    assert len(calls) == 10
    assert again["processing"]["cache_hit"]


def test_custom_model_profiles_change_full_and_orientation_cache_keys(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (100, 200), "white").save(source)
    directories = [tmp_path / "models-a", tmp_path / "models-b"]
    for index, directory in enumerate(directories):
        directory.mkdir()
        (directory / "rus.traineddata").write_bytes(f"rus-model-{index}".encode())
        (directory / "eng.traineddata").write_bytes(b"eng-model")
    calls = []

    def backend(path, cache_dir):
        calls.append(path.name)
        degrees = int(path.stem.split("-")[-1]) if "probe-" in path.name else 90
        box = [.1, .1 if degrees == 90 else .85, .7, .05]
        if degrees in (0, 180):
            box = [.1, .1, .05, .7]
        return {"engine": "tesseract_rus_eng", "tesseract_layout_version": ocr._TESSERACT_LAYOUT_VERSION,
                "text": "СЧЕТ-ФАКТУРА 31", "regions": [{"text": "СЧЕТ-ФАКТУРА 31", "confidence": .9, "bbox": box}]}

    monkeypatch.setattr(ocr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ocr, "_tesseract_result", backend)
    cache = tmp_path / "cache"
    monkeypatch.setenv("TESSDATA_PREFIX", str(directories[0]))
    first = ocr.read_image(source, cache)
    assert len(calls) == 5
    assert ocr.read_image(source, cache)["processing"]["cache_hit"]
    assert len(calls) == 5
    monkeypatch.setenv("TESSDATA_PREFIX", str(directories[1]))
    second = ocr.read_image(source, cache)
    assert len(calls) == 10
    assert not second["processing"]["cache_hit"]
    assert not second["processing"]["orientation_probe_cache_hit"]
    for key in ("ocr_image_path", "orientation_probe_artifact"):
        assert first["processing"][key] != second["processing"][key]
    model = second["processing"]["tesseract_model_profile"]["models"]["rus"]
    assert model["path"] == str((directories[1] / "rus.traineddata").resolve())
    assert model["sha256"] == hashlib.sha256(b"rus-model-1").hexdigest()
    # Replacing content at the same configured location invalidates both caches.
    path = directories[1] / "rus.traineddata"
    old_stat = path.stat()
    path.write_bytes(b"rus-model-2")
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1))
    third = ocr.read_image(source, cache)
    assert len(calls) == 15
    assert not third["processing"]["cache_hit"]
    assert not third["processing"]["orientation_probe_cache_hit"]
    assert third["processing"]["tesseract_model_profile"]["models"]["rus"]["sha256"] == hashlib.sha256(b"rus-model-2").hexdigest()


def test_model_bytes_are_hashed_once_per_file_revision_and_missing_files_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    with pytest.raises(RuntimeError, match="local rus.traineddata"):
        ocr._tesseract_model_profile()
    for language in ("rus", "eng"):
        (tmp_path / f"{language}.traineddata").write_bytes(language.encode())
    actual = Path.read_bytes
    reads = []

    def read_bytes(path):
        reads.append(path)
        return actual(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert ocr._tesseract_model_profile() == ocr._tesseract_model_profile()
    assert len(reads) == 2
    assert ocr._tesseract_profile_identity(None) == ""
    monkeypatch.delenv("TESSDATA_PREFIX")
    assert ocr._tesseract_model_profile() is None


def test_native_vision_does_not_read_a_linux_model_profile(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (100, 200), "white").save(source)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "missing-models"))
    monkeypatch.setattr(ocr.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "swiftc")

    def backend(path, cache_dir):
        degrees = int(path.stem.split("-")[-1]) if "probe-" in path.name else 90
        box = [.1, .1 if degrees == 90 else .85, .7, .05]
        if degrees in (0, 180):
            box = [.1, .1, .05, .7]
        return {"engine": "apple_vision_test", "text": "СЧЕТ-ФАКТУРА 31",
                "regions": [{"text": "СЧЕТ-ФАКТУРА 31", "confidence": .9, "bbox": box}]}

    monkeypatch.setattr(ocr, "_native_result", backend)
    result = ocr.read_image(source, tmp_path / "cache")
    assert "tesseract_model_profile" not in result["processing"]
