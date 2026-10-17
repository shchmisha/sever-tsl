"""Fresh image orientation followed by one native detail reading of that frame.

Orientation selects only pixel rotation. It neither determines a document type
nor gates recognition, and never supplies text or field values to the reader.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from .agents import AnalysisError, _trace
from .image_detail import prepare_detail_panels
from .image_orientation import OrientationSchemaError, choose_reading_rotation
from .image_reader import _prepare_panels, _sha, read_frame


ADAPTIVE_READER_VERSION = "fresh-orientation-native-details-v1"


async def read_frame_adaptive(frame: dict, *, source_id: str, page_id: str,
                              config, cache_dir: Path, trace_path: Path,
                              model=None, orientation_model=None) -> dict:
    """Read current pixels, charging both calls to the same caller allowance.

Unknown or malformed orientation falls back to all four actual rotations.
Provider, budget and proof-storage failures stop the operation. Saved answers
are never input, including the orientation answer beyond its rotation number.
    """
    try:
        path = Path(frame["path"]).resolve()
        digest = _sha(path.read_bytes())
    except (KeyError, TypeError, OSError) as exc:
        raise AnalysisError(f"Cannot bind source pixels ({type(exc).__name__})") from None
    if frame.get("source_sha256") not in (None, digest):
        raise AnalysisError("Source pixels disagree with supplied provenance")
    frame_id = str(frame.get("frame_id") or frame.get("id") or f"source-sha256-{digest}")
    bound = {**frame, "frame_id": frame_id, "source_sha256": digest}
    orientation = None
    fallback = None
    try:
        orientation = await choose_reading_rotation(bound, source_id=source_id,
            page_id=page_id, config=config, cache_dir=cache_dir, trace_path=trace_path,
            model=orientation_model)
    except OrientationSchemaError:
        fallback = "orientation_response_schema_failed"
    rotation = orientation["rotation_clockwise"] if orientation else None
    if orientation:
        proof = orientation["proof"]
        if proof["source_sha256"] != digest or proof["frame_id"] != frame_id:
            raise AnalysisError("Orientation proof belongs to different source pixels")
        if rotation is None:
            fallback = "orientation_unknown"

    def prepare(same_frame, directory):
        # Recheck even the schema-error fallback, whose result has no proof.
        try:
            unchanged = _sha(path.read_bytes()) == digest
        except OSError:
            unchanged = False
        if not unchanged:
            raise AnalysisError("Source pixels changed after orientation; no field call made")
        preparer = (_prepare_panels if rotation is None else
                    partial(prepare_detail_panels, rotation=rotation))
        uploads, proof = preparer(same_frame, directory)
        if proof["source_sha256"] != digest or proof["frame_id"] != frame_id:
            raise AnalysisError("Field panels belong to different source pixels")
        if orientation and proof["source_pixel_size"] != orientation["proof"]["source_pixel_size"]:
            raise AnalysisError("Field panel dimensions differ from the oriented source")
        return uploads, proof

    result = await read_frame(bound, source_id=source_id, page_id=page_id,
        config=config, cache_dir=cache_dir, trace_path=trace_path, model=model,
        panel_preparer=prepare, expected_source_sha256=digest)
    preparation = {"version": ADAPTIVE_READER_VERSION, "rotation_clockwise": rotation,
        "fallback": fallback, "orientation_evidence_path": orientation["evidence_path"] if orientation else None,
        "response_reuse": False, "independent_evidence_frames": 1}
    result["recognition"]["image_preparation"] = preparation
    _trace(trace_path, "adaptive_image_reading", source_id, page_id,
           frame_id=frame_id, source_sha256=digest, **preparation)
    return result
