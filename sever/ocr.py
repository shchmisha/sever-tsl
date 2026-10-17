"""Cached, on-device OCR with explicit content-orientation provenance.

Apple Vision uses the repository's Swift utility; Linux can use an already
installed Tesseract with rus+eng data. This module downloads no models and calls
no remote service. OCR confidence and orientation scores are not field accuracy.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageOps


_VERSION = 8
_TESSERACT_LAYOUT_VERSION = 2
_HEADER_WORDS = re.compile(r"НАКЛАДН|СЧ[ЕЁ]Т|ФАКТУР|УДОСТОВЕРЕНИ|СЕРТИФИКАТ|РЕЕСТР|ПРИЛОЖЕНИ|УНИВЕРСАЛЬНЫЙ\s+ПЕРЕДАТОЧНЫЙ|INVOICE|CERTIFICATE", re.I)
_DOCUMENT_TITLE = re.compile(
    r"^(?:ТОВАРНО[-\s]+ТРАНСПОРТНАЯ\s+НАКЛАДНАЯ|ТОВАРНАЯ\s+НАКЛАДНАЯ|ТРАНСПОРТНАЯ\s+НАКЛАДНАЯ|"
    r"СЧ[ЕЁ]Т[-\s]+ФАКТУРА|УНИВЕРСАЛЬНЫЙ\s+ПЕРЕДАТОЧНЫЙ\s+ДОКУМЕНТ|INVOICE|CERTIFICATE)"
    r"\s*(?=$|[№N\d])", re.I)
_FOOTER_WORDS = re.compile(r"СТРАНИЦА|РУКОВОДИТЕЛ|БУХГАЛТЕР|ПОДПИСЬ|SIGNATURE|PAGE", re.I)


def _native_binary(cache_dir: Path) -> Path:
    source = Path(__file__).resolve().parent / "native" / "vision_ocr.swift"
    if not source.is_file():
        raise RuntimeError("Native Vision requires the repository tools/vision_ocr.swift utility")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = cache_dir / f"vision-ocr-{digest}"
    if target.is_file():
        return target
    swift = shutil.which("swiftc")
    if not swift:
        raise RuntimeError("Swift compiler is unavailable for local Apple Vision OCR")
    with tempfile.TemporaryDirectory(prefix="sever-swift-", dir=cache_dir) as temporary:
        compiled = Path(temporary) / "vision-ocr"
        process = subprocess.run([swift, "-O", str(source), "-o", str(compiled)], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"Native OCR compilation failed: {process.stderr[-2000:]}")
        compiled.replace(target)
    return target


def _native_result(path: Path, cache_dir: Path) -> dict:
    process = subprocess.run([str(_native_binary(cache_dir)), "--languages", "ru-RU,en-US", "--no-correction",
                              "--min-text-height", "0.003", str(path)], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Native Vision OCR failed: {process.stderr[-2000:]}")
    batch = json.loads(process.stdout)
    image = batch["images"][0]
    if image.get("error"):
        raise RuntimeError(image["error"])
    regions = []
    for index, region in enumerate(image["regions"]):
        b = region["boundingBoxNormalizedBottomLeft"]
        regions.append({"text": region["text"], "confidence": region["confidence"],
                        "bbox": [b["x"], 1 - b["y"] - b["height"], b["width"], b["height"]],
                        "region_index": index, "engine_region_index": region.get("visionIndex", index),
                        "candidates": region.get("candidates", [])})
    return {"engine": f"apple_vision_revision_{batch['configuration']['revision']}", "text": image["text"], "regions": regions}


def _tesseract_line_regions(tsv: str, pixel_size: tuple[int, int]) -> list[dict]:
    """Keep real TSV words together, without bridging separated page columns.

    Downstream form and role checks consume text regions, not individual words.
    Tesseract's page/block/paragraph/line identifiers establish which words may
    be joined. A large printed gap or a colon ending a caption keeps adjacent
    values in separate regions. All boxes are unions of actual word boxes; the
    original word readings and recognizer coordinates remain in provenance.
    """
    width, height = pixel_size
    lines: dict[tuple[int, ...], list[dict]] = {}
    # TSV text is a literal OCR field. Quotes in a company name are not CSV
    # quoting and must not consume the following rows until another quote.
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        confidence = float(row.get("conf") or -1)
        if confidence < 0:
            continue
        box = [int(row[k]) for k in ("left", "top", "width", "height")]
        if box[2] <= 0 or box[3] <= 0:
            continue
        key = tuple(int(row[k]) for k in ("page_num", "block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append({"text": text,
            "confidence": min(1., max(0., confidence / 100)),
            "bbox_pixels": box, "word_num": int(row["word_num"])})
    regions = []
    for key, words in sorted(lines.items()):
        words.sort(key=lambda word: (word["word_num"], word["bbox_pixels"][0]))
        typical_height = float(np.median([word["bbox_pixels"][3] for word in words]))
        groups: list[list[dict]] = [[]]
        for word in words:
            previous = groups[-1][-1] if groups[-1] else None
            if previous:
                x, _, w, _ = previous["bbox_pixels"]
                gap = word["bbox_pixels"][0] - x - w
                if gap > 1.8 * typical_height or previous["text"].endswith((":", "：")):
                    groups.append([])
            groups[-1].append(word)
        for group in groups:
            left = min(word["bbox_pixels"][0] for word in group)
            top = min(word["bbox_pixels"][1] for word in group)
            right = max(word["bbox_pixels"][0] + word["bbox_pixels"][2] for word in group)
            bottom = max(word["bbox_pixels"][1] + word["bbox_pixels"][3] for word in group)
            count = sum(len(word["text"]) for word in group)
            regions.append({"text": " ".join(word["text"] for word in group),
                "confidence": sum(len(word["text"]) * word["confidence"] for word in group) / max(1, count),
                "bbox": [left / width, top / height, (right - left) / width, (bottom - top) / height],
                "region_index": len(regions), "engine_region_index": len(regions),
                "engine_line_id": list(key), "engine_pixel_size": [width, height],
                "engine_word_observations": group})
    return regions


def _compatible_tesseract_layout(result: dict) -> bool:
    return (result.get("engine") != "tesseract_rus_eng"
            or result.get("tesseract_layout_version") == _TESSERACT_LAYOUT_VERSION)


@lru_cache(maxsize=32)
def _traineddata_sha256(path: str, mtime_ns: int, size: int, ctime_ns: int, inode: int) -> str:
    """Hash deployed model bytes once per file revision, never download them."""
    del mtime_ns, size, ctime_ns, inode  # These fields form the cache revision key.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tesseract_model_profile() -> dict | None:
    prefix = os.environ.get("TESSDATA_PREFIX")
    if not prefix:
        return None
    directory = Path(prefix).expanduser().resolve()
    models = {}
    for language in ("rus", "eng"):
        path = (directory / f"{language}.traineddata").resolve()
        if not path.is_file():
            raise RuntimeError(f"Configured TESSDATA_PREFIX requires a local {language}.traineddata file: {path}")
        stat = path.stat()
        models[language] = {"path": str(path), "size_bytes": stat.st_size,
                           "sha256": _traineddata_sha256(str(path), stat.st_mtime_ns,
                                                         stat.st_size, stat.st_ctime_ns, stat.st_ino)}
    return {"configuration": "TESSDATA_PREFIX", "directory": str(directory), "models": models}


def _tesseract_profile_identity(profile: dict | None) -> str:
    if profile is None:
        return ""
    digests = {language: model["sha256"] for language, model in profile["models"].items()}
    digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return f":tesseract_models={digest}"


def _tesseract_result(path: Path, cache_dir: Path) -> dict:
    del cache_dir
    profile = _tesseract_model_profile()
    binary = shutil.which("tesseract")
    if not binary:
        raise RuntimeError("Install/configure a local OCR engine: Apple Vision on macOS or Tesseract with rus+eng on Linux")
    installed = subprocess.run([binary, "--list-langs"], capture_output=True, text=True, check=True).stdout.splitlines()
    if not {"rus", "eng"}.issubset({s.strip() for s in installed}):
        raise RuntimeError("Local Tesseract requires installed rus and eng language data; no download is attempted")
    process = subprocess.run([binary, str(path), "stdout", "-l", "rus+eng", "--psm", "3", "tsv"], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Tesseract OCR failed: {process.stderr[-2000:]}")
    with Image.open(path) as image:
        width, height = image.size
    regions = _tesseract_line_regions(process.stdout, (width, height))
    return {"engine": "tesseract_rus_eng", "text": "\n".join(region["text"] for region in regions),
            "regions": regions, "region_granularity": "line_segments",
            "tesseract_layout_version": _TESSERACT_LAYOUT_VERSION,
            "tesseract_model_profile": profile}


def _orientation_score(result: dict, pixel_size: tuple[int, int]) -> dict:
    """Prefer horizontal text and headers above footers; report ambiguity separately."""
    width, height = pixel_size
    character_score, header_bonus, footer_bonus, document_title_bonus = 0., 0., 0., 0.
    heights = [r["bbox"][3] for r in result["regions"] if len(r["text"].strip()) > 4 and r["bbox"][2] * width >= 1.2 * r["bbox"][3] * height]
    typical_height = float(np.median(heights)) if heights else .01
    horizontal_chars, total_chars = 0, 0
    for region in result["regions"]:
        text = region["text"]
        count = len(re.sub(r"\s", "", text))
        x, y, w, h = region["bbox"]
        horizontal = w * width >= 1.2 * h * height
        weight = max(.15, float(region.get("confidence", .5)))
        character_score += min(count, 160) * weight * (1 if horizontal else .15)
        total_chars += count
        horizontal_chars += count if horizontal else 0
        if horizontal:
            prominent = min(5., max(.3, h / max(.001, typical_height))) ** 2
            letters = [c for c in text if c.isalpha()]
            uppercase_title = len(letters) > 4 and sum(c.isupper() for c in letters) / len(letters) > .78
            title_weight = 5. if uppercase_title else .2
            heading = sum(len(m.group()) for m in _HEADER_WORDS.finditer(text)) * max(0., 1 - y - h / 2) * 18 * prominent * title_weight
            header_bonus = max(header_bonus, heading)
            footer_bonus += sum(len(m.group()) for m in _FOOTER_WORDS.finditer(text)) * (y + h / 2) * 5
            if _DOCUMENT_TITLE.match(text.strip()):
                # Exact mixed-case form titles can be the same size as table
                # text. Their placement above the body is still stronger
                # directional evidence than small character-count differences
                # between upright and upside-down recognition. Anchoring the
                # title excludes body references such as 'накладная имеет...'.
                document_title_bonus = max(document_title_bonus, 400 * max(0., .5 - y - h / 2))
    horizontal_fraction = horizontal_chars / max(1, total_chars)
    # A folded/overlapping sheet can expose a large horizontal title while the
    # main leaf is sideways. Only trust layout anchors along the dominant text
    # axis; the character score remains useful when neither axis is dominant.
    dominance_gate = float(np.clip((horizontal_fraction - .4) / .25, 0., 1.))
    return {"score": round(.4 * character_score + dominance_gate * (header_bonus + footer_bonus + document_title_bonus), 3), "recognized_characters": total_chars,
            "horizontal_character_fraction": round(horizontal_fraction, 5),
            "horizontal_character_score": round(.4 * character_score, 3),
            "layout_anchor_dominance_gate": round(dominance_gate, 5),
            "document_title_layout_bonus": round(document_title_bonus, 3),
            "header_layout_bonus": round(header_bonus, 3), "footer_layout_bonus": round(footer_bonus, 3)}


def _write_cache(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.stem, suffix=".tmp", delete=False) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _storage_metadata(storage_format: str) -> dict:
    return {"storage_format": storage_format,
            "mime_type": "image/jpeg" if storage_format == "jpeg" else "image/png",
            "jpeg_quality": 98 if storage_format == "jpeg" else None,
            "jpeg_subsampling": 0 if storage_format == "jpeg" else None}


def _save_ocr_image(image: Image.Image, path: Path, storage_format: str) -> None:
    if storage_format == "jpeg":
        image.save(path, format="JPEG", quality=98, subsampling=0)
    else:
        image.save(path, format="PNG")


def _rotation_matrix(width: int, height: int, degrees: int) -> np.ndarray:
    """Continuous pixel-edge coordinates; clockwise rotation, top-left origin."""
    matrices = {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                90: [[0, -1, height], [1, 0, 0], [0, 0, 1]],
                180: [[-1, 0, width], [0, -1, height], [0, 0, 1]],
                270: [[0, 1, 0], [-1, 0, width], [0, 0, 1]]}
    return np.asarray(matrices[degrees], dtype=float)


def _exif_matrix(width: int, height: int, orientation: int) -> np.ndarray:
    matrices = {1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                2: [[-1, 0, width], [0, 1, 0], [0, 0, 1]],
                3: _rotation_matrix(width, height, 180),
                4: [[1, 0, 0], [0, -1, height], [0, 0, 1]],
                5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                6: _rotation_matrix(width, height, 90),
                7: [[0, -1, height], [-1, 0, width], [0, 0, 1]],
                8: _rotation_matrix(width, height, 270)}
    return np.asarray(matrices.get(orientation, matrices[1]), dtype=float)


def _source_boxes(regions: list[dict], size: tuple[int, int], inverse: np.ndarray) -> None:
    width, height = size
    for region in regions:
        x, y, w, h = region["bbox"]
        corners = np.asarray([[x * width, y * height, 1], [(x + w) * width, y * height, 1],
                              [(x + w) * width, (y + h) * height, 1], [x * width, (y + h) * height, 1]])
        mapped = (inverse @ corners.T).T[:, :2]
        low, high = mapped.min(axis=0), mapped.max(axis=0)
        region["source_quad_pixels"] = mapped.tolist()
        region["source_bbox_pixels"] = [float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1])]


def _header_line_boxes(regions: list[dict], pixel_size: tuple[int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Expand an incomplete number-label row, independently of its unknown value."""
    width, height = pixel_size
    label = re.compile(r"^(?:сч[её]т[- ]фактура|накладная|удостоверение|сертификат)\s*(?:№|N[º°o]?)?\s*$", re.I)
    boxes = []
    for region in regions:
        x, y, w, h = region["bbox"]
        if y > .35 or not label.match(region["text"].strip()):
            continue
        box = (max(0, int(np.floor((x - .025) * width))),
               max(0, int(np.floor((y - .35 * h) * height))),
               min(width, int(np.ceil((x + w + .40) * width))),
               min(height, int(np.ceil((y + 3 * h) * height))))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((f"header_number_line_{len(boxes)}", box))
        if len(boxes) == 2:
            break
    return boxes


def read_image(path: Path, cache_dir: Path, *, enhance_header: bool = False,
               storage_format: str = "png") -> dict:
    """Read a full-resolution frame locally; bbox is normalized top-left after rotation.

    Content orientation is chosen by four downsampled OCR probes. It is a layout
    heuristic; absent headers/symmetric layouts can remain ambiguous. Source
    coordinates remain recoverable through matrices and per-region source boxes.
    Optional JPEG quality98 storage saves disk for new oriented images/crops.
    Its separate cache identity leaves existing default PNG caches reusable.
    """
    if storage_format not in {"png", "jpeg"}:
        raise ValueError("storage_format must be 'png' or 'jpeg'")
    storage = _storage_metadata(storage_format)
    extension = "jpg" if storage_format == "jpeg" else "png"
    path, cache_dir = Path(path).resolve(), Path(cache_dir).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    native = platform.system() == "Darwin" and bool(shutil.which("swiftc"))
    backend = _native_result if native else _tesseract_result
    engine_family = "apple_vision" if native else "tesseract_rus_eng"
    model_profile = None if native else _tesseract_model_profile()
    model_identity = _tesseract_profile_identity(model_profile)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    # Swift changes invalidate image caches too.
    swift_source = Path(__file__).resolve().parent / "native" / "vision_ocr.swift"
    utility_hash = hashlib.sha256(swift_source.read_bytes()).hexdigest() if native and swift_source.exists() else ""
    cache_identity = f"{engine_family}:{utility_hash}:{source_hash}:header={enhance_header}{model_identity}"
    # Preserve the exact historical PNG identity. JPEG recognition sees a
    # lossy image, so it must never reuse full-resolution PNG/crop OCR results.
    if storage_format != "png":
        cache_identity += f":storage={storage_format}:quality=98:subsampling=0"
    key = hashlib.sha256(f"{_VERSION}:{cache_identity}".encode()).hexdigest()
    result_file = cache_dir / f"{key}.json"
    if result_file.exists():
        cached = json.loads(result_file.read_text())
        if _compatible_tesseract_layout(cached) and Path(cached["processing"]["ocr_image_path"]).is_file():
            cached["processing"]["cache_hit"] = True
            cached["processing"]["source_path"] = str(path)
            cached["processing"].update(storage)
            if model_profile is not None:
                cached["processing"]["tesseract_model_profile"] = model_profile
            for enhancement in cached["processing"].get("enhancement_passes", []):
                enhancement.update(storage)
            return cached
    with Image.open(path) as original:
        raw_size = original.size
        exif_orientation = original.getexif().get(274, 1)
        upright = ImageOps.exif_transpose(original).convert("RGB")
    probe = upright.copy()
    probe.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    probe_key = hashlib.sha256(f"orientation-probes:1:{engine_family}:{utility_hash}:{source_hash}{model_identity}".encode()).hexdigest()
    probe_file = cache_dir / f"{probe_key}-orientation-probes.json"
    probe_cache_hit = probe_file.is_file()
    probe_responses = json.loads(probe_file.read_text())["responses"] if probe_cache_hit else []
    if any(not _compatible_tesseract_layout(response["result"]) for response in probe_responses):
        # Word-only Linux probes cannot be rescored as text-line observations.
        probe_cache_hit, probe_responses = False, []
    orientation_scores = []
    if not probe_cache_hit:
        # Keep compact recognition responses for future scoring changes, but
        # remove all four diagnostic images after their local recognition.
        with tempfile.TemporaryDirectory(prefix="sever-orientation-", dir=cache_dir) as temporary:
            for degrees in (0, 90, 180, 270):
                candidate = probe.rotate(-degrees, expand=True)
                probe_path = Path(temporary) / f"probe-{degrees}.png"
                candidate.save(probe_path)
                probe_responses.append({"rotation_degrees": degrees, "pixel_size": list(candidate.size),
                                        "result": backend(probe_path, cache_dir)})
        _write_cache(probe_file, {"source_sha256": source_hash, "probe_pixel_size": list(probe.size),
                                 "network_used": False, "responses": probe_responses})
    for response in probe_responses:
        orientation_scores.append({"rotation_degrees": response["rotation_degrees"],
                                   **_orientation_score(response["result"], tuple(response["pixel_size"]))})
    ranked = sorted(orientation_scores, key=lambda x: (-x["score"], x["rotation_degrees"]))
    degrees = ranked[0]["rotation_degrees"]
    margin = (ranked[0]["score"] - ranked[1]["score"]) / max(1., ranked[0]["score"])
    ambiguous = ranked[0]["recognized_characters"] < 20 or margin < .035
    # Version4 needs small probes once; version5 already retained their raw
    # responses. Both can reuse full-resolution recognition/crop evidence when
    # the new rules select exactly the same content orientation.
    legacy_files = [(version, cache_dir / (hashlib.sha256(f"{version}:{cache_identity}".encode()).hexdigest() + ".json"))
                    for version in (7, 6, 5, 4)]
    for legacy_version, legacy_file in legacy_files:
        if not legacy_file.is_file():
            continue
        legacy = json.loads(legacy_file.read_text())
        if not _compatible_tesseract_layout(legacy):
            continue
        processing = legacy.get("processing", {})
        artifacts = [processing.get("ocr_image_path", "")]
        artifacts += [entry["ocr_image_path"] for entry in processing.get("enhancement_passes", [])]
        if legacy.get("rotation_degrees") == degrees and all(Path(p).is_file() for p in artifacts):
            processing.update(source_path=str(path), cache_hit=True, cache_version=_VERSION,
                              orientation_method="four_local_OCR_probes_dominant_axis_document_title_header_footer_layout",
                              orientation_scores=orientation_scores, orientation_score_margin=round(margin, 5),
                              orientation_ambiguous=ambiguous, orientation_probe_cache_hit=probe_cache_hit,
                              orientation_probe_artifact=str(probe_file),
                              cache_migration={"from_version": legacy_version, "full_resolution_OCR_reused": True,
                                               "reason": "new local probe scores choose the same content rotation"})
            processing.update(storage)
            if model_profile is not None:
                processing["tesseract_model_profile"] = model_profile
            for enhancement in processing.get("enhancement_passes", []):
                enhancement.update(storage)
            processing["limitations"] = [entry for entry in processing.get("limitations", []) if entry != "content_orientation_ambiguous"]
            if ambiguous:
                processing["limitations"].append("content_orientation_ambiguous")
            _write_cache(result_file, legacy)
            return legacy
    final_image = upright.rotate(-degrees, expand=True)
    ocr_path = cache_dir / f"{key}-oriented.{extension}"
    _save_ocr_image(final_image, ocr_path, storage_format)
    result = backend(ocr_path, cache_dir)
    for region in result["regions"]:
        region["processing_pass"] = "full_image"
    enhancement_passes = []
    if enhance_header:
        width, height = final_image.size
        # Alternate local readings preserve conflicts; they do not silently overwrite fields.
        crops = [("header", (0, 0, width, int(height * .60))),
                 ("left_roles", (0, 0, int(width * .68), int(height * .68)))]
        crops.extend(_header_line_boxes(result["regions"], final_image.size))
        for name, box in crops:
            crop = final_image.crop(box)
            scale = min(1.5, max(1., 3200 / max(crop.size)))
            resized_size = (round(crop.width * scale), round(crop.height * scale))
            if resized_size != crop.size:
                crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
            crop_path = cache_dir / f"{key}-{name}.{extension}"
            _save_ocr_image(crop, crop_path, storage_format)
            alternate = backend(crop_path, cache_dir)
            added = 0
            for region in alternate["regions"]:
                x, y, w, h = region["bbox"]
                region["bbox"] = [(box[0] + x * (box[2] - box[0])) / width,
                                  (box[1] + y * (box[3] - box[1])) / height,
                                  w * (box[2] - box[0]) / width, h * (box[3] - box[1]) / height]
                duplicate = any(r["text"] == region["text"] and np.max(np.abs(np.asarray(r["bbox"]) - region["bbox"])) < .015 for r in result["regions"])
                if duplicate:
                    continue
                region["region_index"] = len(result["regions"])
                region["processing_pass"] = name
                result["regions"].append(region)
                result["text"] += "\n" + region["text"]
                added += 1
            enhancement_passes.append({"name": name, "crop_xyxy_in_ocr_image": list(box), "recognizer_pixel_size": list(crop.size),
                                       "ocr_image_path": str(crop_path), "added_alternate_regions": added,
                                       **storage,
                                       "resampling": "LANCZOS upscaling for recognizer; adds no source detail",
                                       "crop_to_ocr_matrix": [[(box[2] - box[0]) / crop.width, 0, box[0]],
                                                              [0, (box[3] - box[1]) / crop.height, box[1]], [0, 0, 1]]})
    from .video import analyze_image_quality
    quality = analyze_image_quality(final_image)
    forward = _rotation_matrix(*upright.size, degrees) @ _exif_matrix(*raw_size, exif_orientation)
    inverse = np.linalg.inv(forward)
    _source_boxes(result["regions"], final_image.size, inverse)
    limits = ["OCR_text_and_confidence_require_validation", "table_row_column_relationships_not_reconstructed",
              "perspective_blur_glare_and_occlusion_not_corrected", "orientation_is_text_layout_heuristic"]
    if ambiguous:
        limits.append("content_orientation_ambiguous")
    if not result["regions"]:
        limits.append("no_readable_text_detected")
    if storage_format == "jpeg":
        limits.append("JPEG_artifacts_may_reduce_OCR_readability")
    result.update(rotation_degrees=degrees, processing={
        **storage,
        "source_path": str(path), "source_sha256": source_hash, "source_pixel_size": list(raw_size),
        "exif_orientation": exif_orientation, "exif_orientation_applied": True,
        "content_rotation_clockwise_degrees": degrees, "ocr_pixel_size": list(final_image.size),
        "ocr_image_path": str(ocr_path), "orientation_probe_pixel_size": list(probe.size),
        "orientation_method": "four_local_OCR_probes_dominant_axis_document_title_header_footer_layout",
        "orientation_scores": orientation_scores, "orientation_score_margin": round(margin, 5),
        "orientation_probe_cache_hit": probe_cache_hit, "orientation_probe_artifact": str(probe_file), "cache_version": _VERSION,
        "orientation_ambiguous": ambiguous, "source_to_ocr_matrix": forward.tolist(), "ocr_to_source_matrix": inverse.tolist(),
        "matrix_coordinates": "continuous pixel-edge coordinates, top-left origin, homogeneous [x,y,1]",
        "bbox_coordinates": "normalized top-left in full-resolution OCR-oriented image",
        "languages": ["ru-RU", "en-US"] if native else ["rus", "eng"], "language_correction": False if native else "engine_default",
        "network_used": False, "cache_hit": False, "enhancement_passes": enhancement_passes,
        "quality": quality, "full_page_visible": quality["full_page_visible"],
        "full_page_visibility_reason": quality["full_page_visibility_reason"], "limitations": limits,
    })
    if model_profile is not None:
        result["processing"]["tesseract_model_profile"] = model_profile
    _write_cache(result_file, result)
    return result
