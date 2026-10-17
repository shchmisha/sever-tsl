"""Leaf-shared orientation and a source-zoom agent, with one-frame provenance."""
from __future__ import annotations

from pathlib import Path

from .agents import AnalysisError, _trace
from .image_agent import read_frame_agent
from .image_orientation import choose_reading_rotation, OrientationSchemaError
from .image_reader import read_frame, _sha
from .image_triage import TriageSchemaError, triage_frame


ZOOM_READER_VERSION = "leaf-orientation-and-source-zoom-v2"


async def read_frame_zoom(frame, *, source_id, page_id, config, cache_dir, trace_path,
                          model=None, orientation_model=None, leaf_orientation=None,
                          leaf_triage=None, triage_model=None):
    """Read one frame; `leaf_orientation` shares one sheet's upright rotation.

    All views of a leaf show the same handheld sheet with display rotation
    already applied, so the first fresh orientation decision is reused for later
    views of that leaf. A reused rotation that yields an insufficient-evidence
    form is dropped, so the next view of the leaf asks for orientation afresh.

    With `leaf_triage`, an upright view first gets one short scope call while the
    leaf has not yet needed a full read. A clear out_of_scope answer with a
    literal heading becomes this view's unknown reading; any other answer
    (including a schema failure) switches the whole leaf to full reads.
    """
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
    reused = (dict(leaf_orientation) if leaf_orientation
              and leaf_orientation.get('rotation_clockwise') is not None else None)
    if reused is not None:
        orientation = {'rotation_clockwise': reused['rotation_clockwise'],
                       'evidence_path': reused['evidence_path'],
                       'proof': {'source_sha256': digest, 'frame_id': frame_id}}
    else:
        try:
            orientation = await choose_reading_rotation(bound, source_id=source_id, page_id=page_id,
                config=config, cache_dir=cache_dir, trace_path=trace_path, model=orientation_model)
        except OrientationSchemaError:
            orientation = None
            fallback = 'orientation_response_schema_failed'
    if orientation and (orientation['proof']['source_sha256'] != digest
                        or orientation['proof']['frame_id'] != frame_id):
        raise AnalysisError('Orientation proof belongs to a different source frame')
    if (reused is None and leaf_orientation is not None and orientation
            and orientation['rotation_clockwise'] is not None):
        leaf_orientation.clear()
        leaf_orientation.update(rotation_clockwise=orientation['rotation_clockwise'],
                                frame_id=frame_id, evidence_path=orientation['evidence_path'])
    try:
        unchanged = _sha(path.read_bytes()) == digest
    except OSError:
        unchanged = False
    if not unchanged:
        raise AnalysisError('Source changed after orientation; no field reading made')
    rotation = orientation['rotation_clockwise'] if orientation else None
    triage = None
    if rotation is not None and leaf_triage is not None and not leaf_triage.get('full_read_required'):
        try:
            triage = await triage_frame(bound, rotation=rotation, source_id=source_id, page_id=page_id,
                config=config, cache_dir=cache_dir, trace_path=trace_path, model=triage_model)
        except TriageSchemaError:
            triage = {'form_scope': None, 'observed_form': '', 'evidence_path': None, 'result': None,
                      'schema_failed': True}
        if triage['result'] is None:
            leaf_triage.update(full_read_required=True, decided_by_frame_id=frame_id)
    if triage is not None and triage['result'] is not None:
        result = triage['result']
    elif rotation is None:
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
        'rotation_source': 'same_leaf_frame' if reused else 'fresh_orientation_call' if orientation else None,
        'rotation_from_frame_id': reused['frame_id'] if reused else frame_id if orientation else None,
        'response_reuse': False, 'independent_evidence_frames': 1}
    if triage is not None:
        metadata['triage'] = {'form_scope': triage['form_scope'], 'evidence_path': triage['evidence_path'],
            'short_circuit': triage['result'] is not None, 'schema_failed': bool(triage.get('schema_failed'))}
    if (reused is not None and leaf_orientation is not None
            and result['document'].document_type == 'unknown'
            and result['recognition'].get('form_scope') == 'insufficient_evidence'):
        # The sheet may have been turned; the next view checks orientation afresh.
        leaf_orientation.clear()
        metadata['leaf_orientation_dropped'] = True
    result['recognition']['image_preparation'] = metadata
    _trace(trace_path, 'source_zoom_image_reading', source_id, page_id,
           frame_id=frame_id, source_sha256=digest, **metadata)
    return result
