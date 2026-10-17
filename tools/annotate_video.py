"""Draw actual recognition values/evidence on the processed video's pixels.

No manual field labels or expected-value files are read by this renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABELS = {
    "invoice_number": "Номер счёт-фактуры", "invoice_date": "Дата счёт-фактуры",
    "consignor": "Грузоотправитель", "consignor_address": "Адрес грузоотправителя",
    "consignee": "Грузополучатель", "consignee_address": "Адрес грузополучателя",
}


def tokens(text):
    return set(re.findall(r"[\w]+", text.casefold().replace("ё", "е")))


def inside(box, panel):
    x, y, w, h = box
    a, b, c, d = panel
    return a <= x + w / 2 <= a + c and b <= y + h / 2 <= b + d


def valid_box(box):
    """Only boxes in the saved OCR-oriented coordinate system are drawable."""
    try:
        values = np.asarray(box, dtype=float)
        return bool(values.shape == (4,) and np.isfinite(values).all()
                    and (values[:2] >= 0).all() and (values[2:] > 0).all()
                    and values[0] + values[2] <= 1.000001
                    and values[1] + values[3] <= 1.000001)
    except (TypeError, ValueError):
        return False


def anchors(key, field, regions, reference_frame_id):
    """Locate evidence only in the actual observation used as the reference.

    A neighbor's normalized box belongs to its own image, even when it shows
    the same leaf. It cannot be used in the reference image's homography.
    """
    found = []
    target = tokens(field.get("raw_value") or field.get("normalized_value") or "")
    by_index = {region["region_index"]: region for region in regions if "region_index" in region}
    for evidence in field.get("evidence", []):
        if evidence.get("frame_id") != reference_frame_id:
            continue
        if evidence.get("region_index") is not None:
            region = by_index.get(evidence["region_index"])
            if region is None or not valid_box(region.get("bbox")):
                continue
            # Stale region indices must not silently relocate an annotation.
            if evidence.get("bbox") is not None and (
                not valid_box(evidence["bbox"])
                or np.max(np.abs(np.asarray(evidence["bbox"]) - region["bbox"])) > .006
            ):
                continue
            found.append({**evidence, "text": region["text"], "bbox": region["bbox"],
                          "kind": "recorded_ocr_region"})
            continue
        if not valid_box(evidence.get("bbox")):
            continue
        matches = []
        for region in regions:
            if not valid_box(region.get("bbox")) or "region_index" not in region:
                continue
            words = tokens(region.get("text", ""))
            overlap = words & target
            if inside(region["bbox"], evidence["bbox"]) and words and (
                words == target or (len(overlap) >= 2 and len(overlap) / len(words) >= .65)
            ):
                matches.append({"frame_id": reference_frame_id, "region_index": region["region_index"],
                                "text": region["text"], "bbox": region["bbox"],
                                "kind": "literal_ocr_inside_recorded_pixel_panel"})
        found.extend(matches or [{**evidence, "kind": "recorded_pixel_reading_panel"}])
    unique = []
    for item in found:
        if not any(np.max(np.abs(np.asarray(item["bbox"]) - old["bbox"])) < .006 for old in unique):
            unique.append(item)
    return unique


def select_reference(page, observations, *, require_all=False):
    """Choose one real OCR view; never combine coordinates across source views.

    The renderer localizes saved recognized results, not their correctness.
    Unsupported results remain in results.json and are explicitly excluded
    from the video rather than being placed at a neighbor's coordinates.
    """
    fields = [(key, field) for key, field in page["result"]["fields"].items()
              if field["status"] == "recognized"]
    if not fields:
        raise ValueError("No recognized fields to annotate")
    frames = {frame["frame_id"]: frame for frame in page["frames"]}
    candidates = []
    for observation in observations:
        frame_id = observation.get("frame_id")
        frame = frames.get(frame_id)
        processing = observation.get("processing", {})
        # The OCR coordinate system must be tied to this page's actual frame.
        if frame is None or not processing.get("source_path") or not processing.get("ocr_image_path"):
            continue
        if Path(processing["source_path"]).resolve() != Path(frame["path"]).resolve():
            continue
        evidence = {key: anchors(key, field, observation.get("regions", []), frame_id)
                    for key, field in fields}
        supported = [(key, field) for key, field in fields if evidence[key]]
        candidates.append({"observation": observation, "frame": frame, "fields": supported,
                           "evidence": {key: evidence[key] for key, _ in supported}})
    if not candidates:
        raise ValueError("No OCR observation is bound to an actual page frame")
    candidates.sort(key=lambda candidate: (
        -len(candidate["fields"]), candidate["frame"]["frame_id"] != page.get("selected_frame_id"),
        candidate["frame"].get("timestamp_s", float("inf")), candidate["frame"]["frame_id"]))
    chosen = candidates[0]
    frame_id = chosen["frame"]["frame_id"]
    supported_names = {key for key, _ in chosen["fields"]}
    excluded = {}
    for key, field in fields:
        if key in supported_names:
            continue
        sources = sorted({item["frame_id"] for item in field.get("evidence", []) if item.get("frame_id")})
        excluded[key] = {
            "reason": ("no_field_evidence" if not sources else "no_evidence_on_reference_frame"
                       if frame_id not in sources else "invalid_or_unresolved_reference_geometry"),
            "evidence_frame_ids": sources,
        }
    if not supported_names:
        raise ValueError("No recognized field has drawable evidence on one actual OCR frame")
    if require_all and excluded:
        raise ValueError("No single OCR frame grounds all recognized fields; best reference "
                         + frame_id + " excludes " + ", ".join(excluded))
    chosen.update(excluded_fields=excluded, all_recognized_fields_grounded=not excluded,
                  candidate_support=[{"frame_id": candidate["frame"]["frame_id"],
                                      "supported_fields": [key for key, _ in candidate["fields"]]}
                                     for candidate in candidates])
    return chosen


def face(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    candidates = [Path(f"/System/Library/Fonts/Supplemental/{name}"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                       if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("A Cyrillic Arial or DejaVu Sans font is required for video annotation")


def lines(draw, value, font, width):
    output, current = [], ""
    for word in value.split():
        proposed = (current + " " + word).strip()
        if current and draw.textlength(proposed, font=font) > width:
            output.append(current)
            current = word
        else:
            current = proposed
    return output + [current]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/demo-send/processing/input-3s"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo-send/annotated.mp4"))
    parser.add_argument("--require-all-grounded", action="store_true",
                        help="Fail unless one actual OCR observation grounds every recognized field")
    args = parser.parse_args()
    run = args.run.resolve()
    data = json.loads((run / "results.json").read_text())
    source = Path(data["source"]["path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == data["source"]["sha256"]
    page = next(p for p in data["pages"] if p["result"]["document_type"] == "upd")
    observations = json.loads((run / "ocr" / f"{page['page_id']}.json").read_text())
    chosen = select_reference(page, observations, require_all=args.require_all_grounded)
    frame, ocr = chosen["frame"], chosen["observation"]
    fields, evidence = chosen["fields"], chosen["evidence"]
    reference = cv2.imread(ocr["processing"]["ocr_image_path"])
    if reference is None:
        raise ValueError("Chosen OCR reference image cannot be read")
    reference = cv2.resize(reference, (1920, 1080))
    rotation = ocr["processing"]["content_rotation_clockwise_degrees"]
    boxes = [a["bbox"] for v in evidence.values() for a in v]
    low = np.min([[b[0], b[1]] for b in boxes], axis=0)
    high = np.max([[b[0] + b[2], b[1] + b[3]] for b in boxes], axis=0)
    roi = np.r_[np.maximum(low - [.045, .035], 0), np.minimum(high + [.045, .035], 1)]
    x0, y0, x1, y1 = np.rint(roi * [1920, 1080, 1920, 1080]).astype(int)

    orb = cv2.ORB_create(nfeatures=2500)
    reference_keys, reference_desc = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1920x1080", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)
    ], stdin=subprocess.PIPE)
    tracking, rendered = [], 0
    repeat = 4
    while True:
        ok, image = capture.read()
        if not ok:
            break
        image = {0: lambda x: x, 90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                 180: lambda x: cv2.rotate(x, cv2.ROTATE_180),
                 270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)}[rotation](image)
        image = cv2.resize(image, (1920, 1080))
        keys, desc = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        good = []
        if reference_desc is not None and desc is not None:
            good = [m for pair in matcher.knnMatch(reference_desc, desc, k=2)
                    if len(pair) == 2 for m, n in [pair] if m.distance < .72 * n.distance]
        matrix, mask = None, None
        if len(good) >= 12:
            a = np.float32([reference_keys[m.queryIdx].pt for m in good])
            b = np.float32([keys[m.trainIdx].pt for m in good])
            matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, 3)
        inliers = int(mask.sum()) if mask is not None else 0
        usable = matrix is not None and inliers >= 15 and inliers / max(len(good), 1) >= .45
        tracking.append({"input_frame": rendered, "matches": len(good), "inliers": inliers,
                         "overlay_registered": bool(usable), "reference_frame_id": frame["frame_id"],
                         "reference_to_current_homography": matrix.tolist() if usable else None})
        crop = image[y0:y1, x0:x1]
        scale = min(1920 / crop.shape[1], 735 / crop.shape[0])
        preview = cv2.resize(crop, None, fx=scale, fy=scale)
        px, py = (1920 - preview.shape[1]) // 2, 92 + (735 - preview.shape[0]) // 2
        for repetition in range(repeat):
            step = min(len(fields) - 1, int((rendered * repeat + repetition) * len(fields) / (nframes * repeat)))
            key, field = fields[step]
            canvas = Image.new("RGB", (1920, 1080), "#101c2d")
            canvas.paste(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), (px, py))
            draw = ImageDraw.Draw(canvas)
            draw.text((40, 21), "УПД — найденные реквизиты на исходном видео", font=face(39, True), fill="white")
            if usable:
                for anchor in evidence[key]:
                    bx, by, bw, bh = anchor["bbox"]
                    quad = np.float32([[[bx*1920, by*1080], [(bx+bw)*1920, by*1080],
                                       [(bx+bw)*1920, (by+bh)*1080], [bx*1920, (by+bh)*1080]]])
                    points = cv2.perspectiveTransform(quad, matrix)[0]
                    points = [(float((x-x0)*scale+px), float((y-y0)*scale+py)) for x, y in points]
                    draw.line(points + [points[0]], fill="#06dfb0", width=5)
            draw.rectangle((0, 840, 1920, 1080), fill="#14263e")
            label = LABELS.get(key, key)
            if key in {"consignor", "consignor_address"} and any(
                re.search(r"\bон\s*же\b", item.get("text", ""), re.I) for item in evidence[key]
            ):
                label += " · «он же» → реквизиты продавца"
            draw.text((40, 855), f"{step + 1}/{len(fields)} · {label}", font=face(34), fill="#06dfb0")
            value = field["raw_value"] or field["normalized_value"]
            if key == "invoice_date" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", field.get("normalized_value") or ""):
                value = ".".join(reversed(field["normalized_value"].split("-")))
            for j, line in enumerate(lines(draw, value, face(40, True), 1840)):
                draw.text((40, 905+j*48), line, font=face(40, True), fill="white")
            footer = "Значения из результата распознавания · фрагмент увеличен и замедлен ×4"
            if not usable:
                footer = "Положение области не подтверждено на этом кадре: рамки отключены"
            draw.text((40, 1045), footer, font=face(23), fill="#a8b8ca")
            process.stdin.write(canvas.tobytes())
        rendered += 1
    capture.release()
    process.stdin.close()
    if process.wait():
        raise RuntimeError("Video encoding failed")
    manifest = {"annotation_values_source": str(run / "results.json"),
                "results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
                "source_video_sha256": data["source"]["sha256"], "manual_ground_truth_read": False,
                "selected_frame_id": page["selected_frame_id"], "display_crop_normalized": roi.tolist(),
                "annotation_reference_frame_id": frame["frame_id"],
                "annotation_reference_timestamp_s": frame.get("timestamp_s"),
                "annotation_reference_source_path": ocr["processing"]["source_path"],
                "annotation_reference_ocr_image_path": ocr["processing"]["ocr_image_path"],
                "reference_selection_method": "maximum_recognized_fields_with_frame_local_evidence_primary_on_tie",
                "all_recognized_fields_grounded_on_reference": chosen["all_recognized_fields_grounded"],
                "excluded_recognized_fields": chosen["excluded_fields"],
                "reference_candidates": chosen["candidate_support"],
                "slowdown_factor": repeat, "fields": {k: {"value": f["raw_value"], "evidence_anchors": evidence[k]} for k,f in fields},
                "tracking": tracking, "frames_with_registered_overlays": sum(t["overlay_registered"] for t in tracking),
                "input_frames": rendered}
    args.out.with_suffix(".provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"video": str(args.out), "frames": rendered,
                      "registered_frames": manifest["frames_with_registered_overlays"],
                      "reference_frame_id": frame["frame_id"], "rendered_fields": [key for key, _ in fields],
                      "excluded_recognized_fields": list(chosen["excluded_fields"]),
                      "manual_labels_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
