"""Prepare one source frame as upright context and four native detail panels.

Rotation is an explicit caller decision. Crops are generic overlapping 2x2
source rectangles, never inferred text/role boxes or independent source views.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .agents import AnalysisError


DETAIL_RECIPE = "upright-context-and-overlapping-2x2-native-details-v1"
ORIENTATION_RECIPE = "four-native-source-context-rotations-v1"


def _rotate(image: Image.Image, rotation: int) -> Image.Image:
    if rotation == 0:
        return image.copy()
    return image.transpose({90: Image.Transpose.ROTATE_270,
                            180: Image.Transpose.ROTATE_180,
                            270: Image.Transpose.ROTATE_90}[rotation])


def _upright_boxes(width: int, height: int):
    return [(f"detail_{row}_{column}", (left, top, right, bottom))
            for row, (top, bottom) in enumerate(((0, max(1, round(.6*height))),
                                                 (min(height-1, round(.4*height)), height)))
            for column, (left, right) in enumerate(((0, max(1, round(.6*width))),
                                                    (min(width-1, round(.4*width)), width)))]


def _source_box(box, size, rotation: int):
    left, top, right, bottom = box
    width, height = size
    if rotation == 0:
        return box
    if rotation == 90:
        return top, height-right, bottom, height-left
    if rotation == 180:
        return width-right, height-bottom, width-left, height-top
    return width-bottom, left, width-top, right


def _prepare_panels(frame: dict, evidence_dir: Path, *, rotations: tuple[int, ...], details: bool):
    # Imported at call time so the reader can use this preparation module
    # without a module-initialization cycle.
    from .image_reader import _sha, _source_transform, _inverse_affine

    try:
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
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AnalysisError(f"Cannot read saved source frame ({type(error).__name__}); no model call made") from None
    digest = _sha(source_bytes)
    expected = frame.get("source_sha256")
    if expected is not None and expected != digest:
        raise AnalysisError("Saved source frame SHA does not match supplied provenance; no model call made")
    width, height = image.size
    recipe = DETAIL_RECIPE if details else ORIENTATION_RECIPE
    rectangles = []
    for rotation in rotations:
        rectangles.append(("context", (0, 0, width, height), False, rotation))
        if details:
            upright_size = (height, width) if rotation % 180 else (width, height)
            rectangles.extend((name, _source_box(box, image.size, rotation), True, rotation)
                              for name, box in _upright_boxes(*upright_size))
    folder = Path(evidence_dir)/"image-panels"
    folder.mkdir(parents=True, exist_ok=True)
    frame_id = str(frame.get("frame_id") or frame.get("id") or f"source-sha256-{digest}")
    uploads, panels = [], {}
    for name, box, native, rotation in rectangles:
        uploaded = _rotate(image.crop(box), rotation)
        if not native:
            uploaded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        uploaded.save(buffer, "JPEG", quality=95, subsampling=0)
        data = buffer.getvalue()
        upload_digest = _sha(data)
        target = folder/f"{upload_digest}.jpg"
        temporary = folder/f".{upload_digest}.{uuid4().hex}.tmp"
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        identifier = f"{name}_r{rotation}"
        left, top, right, bottom = box
        transform = _source_transform(box, rotation, uploaded.size)
        panels[identifier] = {"panel_id": identifier, "frame_id": frame_id,
            "bbox": [left/width, top/height, (right-left)/width, (bottom-top)/height],
            "source_bbox_pixels_top_left": list(box),
            "rotation_clockwise_degrees": rotation,
            "upload_pixel_size": list(uploaded.size), "upload_sha256": upload_digest,
            "upload_path": str(target), "source_to_upload_pixels": transform,
            "upload_to_source_pixels": _inverse_affine(transform),
            "prep": {"format": "JPEG", "quality": 95, "subsampling": 0,
                     "resize": "none" if native else "LANCZOS",
                     "max_edge": None if native else 1024,
                     "native_source_detail": native, "exif_orientation_applied": False,
                     "recipe": recipe}}
        uploads.append((identifier, data))
    try:
        unchanged = _sha(source.read_bytes()) == digest
    except OSError:
        unchanged = False
    if not unchanged:
        raise AnalysisError("Source pixels changed during detail preparation; no model call made")
    return uploads, {"frame_id": frame_id, "source_path": str(source),
        "source_sha256": digest, "source_pixel_size": [width, height],
        "source_exif_orientation": exif_orientation,
        "coordinate_system": "normalized_top_left_saved_source_frame",
        "panels": panels, "timestamp_s": frame.get("timestamp_s"),
        "display_rotation_applied": frame.get("display_rotation_applied"),
        "recipe": recipe, "rotation_clockwise_degrees": rotations[0] if len(rotations) == 1 else None,
        "rotations_clockwise_degrees": list(rotations),
        "source_unchanged_during_preparation": True}


def prepare_detail_panels(frame: dict, evidence_dir: Path, *, rotation: int):
    """Return a context and four native quarters at the explicit orientation.

    Only whole-frame context shrinks to 1024 pixels on its long edge. The source
    remains byte-identical throughout preparation. No saved answers are read.
    """
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be an integer 0, 90, 180 or 270")
    return _prepare_panels(frame, evidence_dir, rotations=(rotation,), details=True)


def prepare_orientation_panels(frame: dict, evidence_dir: Path):
    """Prepare only four context rotations; no detail image is encoded or saved."""
    return _prepare_panels(frame, evidence_dir, rotations=(0, 90, 180, 270), details=False)
