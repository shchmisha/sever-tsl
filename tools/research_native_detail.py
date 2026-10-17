"""Bounded fresh local reading with generic upright 2x2 native-detail panels.

This diagnostic changes image preparation only. It uses the installed reader's
unchanged schema/instructions and never loads previous answers or ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sever.agents import AgentConfig
from sever import image_reader


RECIPE = "upright-context-and-overlapping-2x2-native-details-v1"
REQUEST_LIMIT = 2
TOKEN_LIMIT = 150_000


def _rotate(image, rotation):
    if rotation == 0:
        return image.copy()
    return image.transpose({90: Image.Transpose.ROTATE_270,
                            180: Image.Transpose.ROTATE_180,
                            270: Image.Transpose.ROTATE_90}[rotation])


def _upright_boxes(width, height):
    return [(f"detail_{row}_{column}", (left, top, right, bottom))
            for row, (top, bottom) in enumerate(((0, max(1, round(.6*height))),
                                                 (min(height-1, round(.4*height)), height)))
            for column, (left, right) in enumerate(((0, max(1, round(.6*width))),
                                                    (min(width-1, round(.4*width)), width)))]


def _source_box(box, size, rotation):
    """Map an upright crop's exact pixel edges back to the saved source."""
    left, top, right, bottom = box
    width, height = size
    if rotation == 0:
        return box
    if rotation == 90:
        return top, height-right, bottom, height-left
    if rotation == 180:
        return width-right, height-bottom, width-left, height-top
    return width-bottom, left, width-top, right


def prepare_panels(frame, evidence_dir, *, rotation):
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180 or 270")
    source = Path(frame["path"]).resolve()
    source_bytes = source.read_bytes()
    with Image.open(BytesIO(source_bytes)) as original:
        original.load()
        exif_orientation = original.getexif().get(274)
        if original.mode in {"RGBA", "LA"} or "transparency" in original.info:
            rgba = original.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = original.convert("RGB")
    width, height = image.size
    upright_size = (height, width) if rotation % 180 else (width, height)
    rectangles = [("context", (0, 0, width, height), False)]
    rectangles.extend((name, _source_box(box, image.size, rotation), True)
                      for name, box in _upright_boxes(*upright_size))
    folder = Path(evidence_dir)/"image-panels"
    folder.mkdir(parents=True, exist_ok=True)
    frame_id = str(frame["frame_id"])
    uploads, panels = [], {}
    for name, box, native in rectangles:
        uploaded = _rotate(image.crop(box), rotation)
        if not native:
            uploaded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        uploaded.save(buffer, "JPEG", quality=95, subsampling=0)
        data = buffer.getvalue()
        digest = image_reader._sha(data)
        target = folder/f"{digest}.jpg"
        target.write_bytes(data)
        identifier = f"{name}_r{rotation}"
        left, top, right, bottom = box
        transform = image_reader._source_transform(box, rotation, uploaded.size)
        panels[identifier] = {"panel_id": identifier, "frame_id": frame_id,
            "bbox": [left/width, top/height, (right-left)/width, (bottom-top)/height],
            "source_bbox_pixels_top_left": list(box),
            "rotation_clockwise_degrees": rotation,
            "upload_pixel_size": list(uploaded.size), "upload_sha256": digest,
            "upload_path": str(target), "source_to_upload_pixels": transform,
            "upload_to_source_pixels": image_reader._inverse_affine(transform),
            "prep": {"format": "JPEG", "quality": 95, "subsampling": 0,
                     "resize": "none" if native else "LANCZOS",
                     "max_edge": None if native else 1024,
                     "native_source_detail": native, "exif_orientation_applied": False,
                     "diagnostic_recipe": RECIPE}}
        uploads.append((identifier, data))
    return uploads, {"frame_id": frame_id, "source_path": str(source),
        "source_sha256": image_reader._sha(source_bytes), "source_pixel_size": [width, height],
        "source_exif_orientation": exif_orientation,
        "coordinate_system": "normalized_top_left_saved_source_frame",
        "panels": panels, "timestamp_s": frame.get("timestamp_s"),
        "diagnostic_recipe": RECIPE, "diagnostic_rotation_clockwise_degrees": rotation}


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def geometry_self_test():
    """Independent pixel-edge and crop checks; no model/client is constructed."""
    checked = 0
    with tempfile.TemporaryDirectory(prefix="sever-native-detail-geometry-") as temporary:
        folder = Path(temporary)
        source = folder/"source.png"
        image = Image.new("RGB", (13, 19))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), (x*11, y*7, (x+y)*5))
        image.save(source)
        frame = {"frame_id": "synthetic-geometry", "path": str(source)}
        for rotation in (0, 90, 180, 270):
            upright = _rotate(image, rotation)
            for _, box in _upright_boxes(*upright.size):
                cropped = _rotate(image.crop(_source_box(box, image.size, rotation)), rotation)
                assert ImageChops.difference(cropped, upright.crop(box)).getbbox() is None
            uploads, proof = prepare_panels(frame, folder/str(rotation), rotation=rotation)
            assert len(uploads) == 5 and proof["source_sha256"] == image_reader._sha(source.read_bytes())
            for identifier, data in uploads:
                panel = proof["panels"][identifier]
                assert data == Path(panel["upload_path"]).read_bytes()
                assert panel["upload_sha256"] == image_reader._sha(data)
                left, top, right, bottom = panel["source_bbox_pixels_top_left"]
                points = ((left, top), (right, top), (right, bottom), (left, bottom))
                target_width, target_height = panel["upload_pixel_size"]
                mapped = []
                for x, y in points:
                    matrix = panel["source_to_upload_pixels"]
                    u, v = (matrix[axis][0]*x+matrix[axis][1]*y+matrix[axis][2] for axis in (0, 1))
                    mapped.append((round(u, 8), round(v, 8)))
                    inverse = panel["upload_to_source_pixels"]
                    restored = [inverse[axis][0]*u+inverse[axis][1]*v+inverse[axis][2] for axis in (0, 1)]
                    assert abs(restored[0]-x) < 1e-7 and abs(restored[1]-y) < 1e-7
                assert set(mapped) == {(0, 0), (target_width, 0), (0, target_height), (target_width, target_height)}
                checked += 1
    return {"geometry_checks": checked, "status": "passed", "model_requests": 0}


async def run(args):
    sources = [Path(path).resolve() for path in args.input]
    if not 1 <= len(sources) <= REQUEST_LIMIT or len(set(sources)) != len(sources):
        raise ValueError("Provide one or two distinct actual source frame paths")
    source_hashes = [image_reader._sha(source.read_bytes()) for source in sources]
    out = args.out.resolve()
    config = AgentConfig(model_name=args.model, vision_model_name=args.model, base_url=args.base_url,
        provider_kind="local", document_reader="native_image", timeout_seconds=1800,
        max_output_tokens=8192, request_limit=REQUEST_LIMIT, max_requests=REQUEST_LIMIT,
        input_tokens_limit=TOKEN_LIMIT, output_tokens_limit=TOKEN_LIMIT, max_tokens=TOKEN_LIMIT,
        allow_tools=False, verify_inn=False)
    out.mkdir(parents=True, exist_ok=False)
    summary = {"run_kind": "isolated_fresh_native_detail_diagnostic", "status": "processing",
        "recipe": RECIPE, "rotation_clockwise_degrees": args.rotation,
        "configuration": {"reader_version": image_reader.IMAGE_READER_VERSION,
            "schema_version": image_reader.IMAGE_SCHEMA_VERSION, "model": args.model,
            "endpoint": args.base_url, "max_requests": REQUEST_LIMIT,
            "aggregate_token_limit": TOKEN_LIMIT, "output_max_tokens": 8192,
            "instructions_sha256": image_reader._sha(image_reader.INSTRUCTIONS.encode())},
        "ground_truth_read": False, "saved_answers_input": False, "response_reuse": False,
        "production_reader_changed": False, "independent_frame_corroboration": False,
        "accuracy_evaluated": False, "readings": []}
    save(out/"summary.json", summary)
    def prepare(frame, evidence_dir):
        return prepare_panels(frame, evidence_dir, rotation=args.rotation)
    with patch.object(image_reader, "_prepare_panels", prepare):
        for index, (source, digest) in enumerate(zip(sources, source_hashes), 1):
            folder = out/f"frame-{index:02d}"
            folder.mkdir()
            frame = {"path": str(source), "frame_id": f"source-frame-{digest}", "timestamp_s": None}
            entry = {"source_path": str(source), "source_sha256": digest,
                     "frame_id": frame["frame_id"], "status": "processing"}
            summary["readings"].append(entry)
            save(out/"summary.json", summary)
            before = asdict(config.usage)
            started = time.monotonic()
            try:
                if image_reader._sha(source.read_bytes()) != digest:
                    raise ValueError("Source pixels changed before the fresh read")
                result = await image_reader.read_frame(frame, source_id=digest,
                    page_id=f"diagnostic-frame-{index:02d}", config=config,
                    cache_dir=folder/"evidence", trace_path=folder/"trace.jsonl")
                if image_reader._sha(source.read_bytes()) != digest:
                    raise ValueError("Source pixels changed during the fresh read")
                save(folder/"reading.json", {**result, "document": result["document"].model_dump(mode="json")})
                entry.update(status="completed", document_type=result["document"].document_type,
                             structural_errors=result["document"].errors,
                             reading_path=str(folder/"reading.json"))
            except Exception as error:
                entry.update(status="failed", error_type=type(error).__name__)
            entry.update(elapsed_seconds=round(time.monotonic()-started, 3),
                usage_delta={key: asdict(config.usage)[key]-before[key]
                             for key in ("requests", "input_tokens", "output_tokens", "tool_calls")})
            summary["usage"] = config.usage.model_dump()
            save(out/"summary.json", summary)
            print(json.dumps({"frame_index": index, "status": entry["status"],
                              "usage_delta": entry["usage_delta"]}), flush=True)
    summary["status"] = "completed" if all(entry["status"] == "completed"
        and not entry.get("structural_errors") for entry in summary["readings"]) else "completed_with_errors"
    save(out/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--model", default="qwen38-27b-q6")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(geometry_self_test()))
        return
    if not args.input or args.out is None:
        parser.error("--input and a fresh --out are required for model reading")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
