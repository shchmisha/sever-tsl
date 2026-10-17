"""Fresh orientation and a source-zoom agent, with one-frame provenance."""
from __future__ import annotations

from pathlib import Path

from .agents import AnalysisError, _trace
from .image_agent import read_frame_agent
from .image_orientation import choose_reading_rotation, OrientationSchemaError
from .image_reader import read_frame, _sha


ZOOM_READER_VERSION = "fresh-orientation-and-source-zoom-v1"


async def read_frame_zoom(frame, *, source_id, page_id, config, cache_dir, trace_path,
                          model=None, orientation_model=None):
    try:
        path = Path(frame['path']).resolve()
        digest = _sha(path.read_bytes())
    except (KeyError, TypeError, OSError) as exc:
        raise AnalysisError(f'Cannot bind source pixels ({type(exc).__name__})') from None
    if frame.get('source_sha256') not in (None, digest):
        raise AnalysisError('Source pixels disagree with supplied provenance')
    frame_id = str(frame.get('frame_id') or frame.get('id') or f'source-sha256-{digest}')
    bound = {**frame, 'frame_id': frame_id, 'source_sha256': digest}
    fallback = None
    try:
        orientation = await choose_reading_rotation(bound, source_id=source_id, page_id=page_id,
            config=config, cache_dir=cache_dir, trace_path=trace_path, model=orientation_model)
    except OrientationSchemaError:
        orientation = None
        fallback = 'orientation_response_schema_failed'
    if orientation and (orientation['proof']['source_sha256'] != digest
                        or orientation['proof']['frame_id'] != frame_id):
        raise AnalysisError('Orientation proof belongs to a different source frame')
    try:
        unchanged = _sha(path.read_bytes()) == digest
    except OSError:
        unchanged = False
    if not unchanged:
        raise AnalysisError('Source changed after orientation; no field reading made')
    rotation = orientation['rotation_clockwise'] if orientation else None
    if rotation is None:
        fallback = fallback or 'orientation_unknown'
        result = await read_frame(bound, source_id=source_id, page_id=page_id, config=config,
            cache_dir=cache_dir, trace_path=trace_path, expected_source_sha256=digest, model=model)
    else:
        result = await read_frame_agent(bound, source_id=source_id, page_id=page_id, config=config,
            cache_dir=cache_dir, trace_path=trace_path, rotation=rotation, model=model)
    proof = result['recognition']['pixel_proof']
    if proof['source_sha256'] != digest or proof['frame_id'] != frame_id:
        raise AnalysisError('Field reading belongs to a different source frame')
    metadata = {'version': ZOOM_READER_VERSION, 'rotation_clockwise': rotation,
        'fallback': fallback, 'orientation_evidence_path': orientation['evidence_path'] if orientation else None,
        'response_reuse': False, 'independent_evidence_frames': 1}
    result['recognition']['image_preparation'] = metadata
    _trace(trace_path, 'source_zoom_image_reading', source_id, page_id,
           frame_id=frame_id, source_sha256=digest, **metadata)
    return result
