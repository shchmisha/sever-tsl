"""Pure selection of disputed fields and their original source views.

These helpers never repair text, open source files, or call a model. The caller
must supply pages belonging to one assembled document. Returned frame records
contain source identity and geometry only, never the competing field values.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

from .image_fusion import _field_comparison
from .schema import FIELDS_BY_TYPE


_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _get(item: Any, name: str, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _recognized_key(field: Any, name: str) -> str | None:
    raw = _get(field, "raw_value")
    if _get(field, "status") != "recognized" or not isinstance(raw, str) or not raw.strip():
        return None
    return _field_comparison(name, raw)


def needs_detail_verification(proposed_field, readings, field_name: str, document_type: str) -> bool:
    """Whether a recognized proposal disagrees with a grounded direct reading.

    Agreement is tested per original frame reading, so an agreed suffix cannot
    be changed unnoticed merely because the readings differ elsewhere. Missing,
    printed-blank, and ungrounded readings do not create a verification target.
    ``readings`` accepts DocumentResult objects or their JSON dictionaries.
    """
    if field_name not in FIELDS_BY_TYPE.get(document_type, ()):
        return False
    proposed = _recognized_key(proposed_field, field_name)
    if proposed is None:
        return False
    for reading in readings:
        if _get(reading, "document_type") != document_type:
            continue
        field = (_get(reading, "fields", {}) or {}).get(field_name)
        observed = _recognized_key(field, field_name)
        if observed is None:
            continue
        evidence = _get(field, "evidence", ()) or ()
        if any(isinstance(_get(item, "frame_id"), str) and _get(item, "frame_id").strip()
               for item in evidence) and observed != proposed:
            return True
    return False


def _finite(value, default=0.0):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else default


def _source_frame(view, original, *, source_page_id: str, source_id) -> dict | None:
    proof = view.get("pixel_proof") or {}
    frame_id, path = proof.get("frame_id"), proof.get("source_path")
    digest, timestamp = proof.get("source_sha256"), proof.get("timestamp_s")
    rotation = proof.get("rotation_clockwise_degrees")
    if (not isinstance(frame_id, str) or not frame_id.strip() or frame_id != view.get("frame_id")
            or not isinstance(path, str) or not path.strip()
            or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            or type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0
            or type(rotation) is not int or rotation not in (0, 90, 180, 270)):
        return None
    document = view.get("document") or {}
    if ((source_id is not None and _get(document, "source_id") != source_id)
            or (_get(document, "page_id") not in (None, source_page_id))
            or (view.get("frame_path") not in (None, path))):
        return None
    if original:
        if (original.get("path") not in (None, path)
                or original.get("source_sha256") not in (None, digest)
                or original.get("timestamp_s") not in (None, timestamp)):
            return None
    return {"frame_id": frame_id, "path": path, "source_sha256": digest.lower(),
            "timestamp_s": timestamp, "rotation_clockwise_degrees": rotation,
            "source_page_id": source_page_id}


def frames_for_verification(pages: list[dict], field_name: str, limit: int = 3) -> list[dict]:
    """Choose 2–3 sharp original views covering every field-value dispute.

    Across pages, coverage uses final recognized page values: differences in
    older noisy readings must not manufacture extra document-level disputes.
    For a page whose final field is conflicting, its direct recognized values
    define the sides to cover instead. Unreadable/printed-blank fields are not
    verification targets. Every chosen frame has a distinct ID, source SHA and
    captured timestamp. If any side cannot be covered within the limit, return
    an empty list and leave the field unresolved.

    Membership in one assembled document is the caller's responsibility; this
    helper additionally rejects mixing form types or source-video identities.
    """
    if type(limit) is not int or not 2 <= limit <= 3:
        raise ValueError("Detail verification requires a limit of two or three frames")
    if not pages:
        return []
    kinds = {_get(page.get("result") or {}, "document_type") for page in pages}
    source_ids = {_get(page.get("result") or {}, "source_id") for page in pages}
    if len(kinds) != 1 or len(source_ids) != 1:
        return []
    kind, source_id = next(iter(kinds)), next(iter(source_ids))
    if field_name not in FIELDS_BY_TYPE.get(kind, ()):
        return []

    groups: dict[str, list] = defaultdict(list)
    required = set()
    order = 0
    for page in pages:
        result = page.get("result") or {}
        source_page_id = page.get("page_id") or _get(result, "page_id")
        if not isinstance(source_page_id, str) or not source_page_id.strip():
            return []
        field = (_get(result, "fields", {}) or {}).get(field_name)
        final_key = _recognized_key(field, field_name)
        if final_key is None and _get(field, "status") != "conflict":
            continue
        if final_key is not None:
            required.add(final_key)
        originals = {frame.get("frame_id"): frame for frame in page.get("frames", ())}
        for view in (page.get("image_recognition") or {}).get("views", ()):
            direct = view.get("document") or {}
            if (_get(direct, "document_type") != kind
                    or view.get("document_type", kind) != kind):
                continue
            observed = (_get(direct, "fields", {}) or {}).get(field_name)
            key = final_key if final_key is not None else _recognized_key(observed, field_name)
            if key is None:
                continue
            required.add(key)
            original = originals.get(view.get("frame_id")) or {}
            frame = _source_frame(view, original, source_page_id=source_page_id, source_id=source_id)
            if frame is None:
                continue
            quality = original.get("quality") or {}
            rank = (-_finite(quality.get("relative_selection_score")),
                    -_finite(quality.get("sharpness_laplacian_variance")), order)
            groups[key].append((frame, rank))
            order += 1
    if not required or len(required) > limit or any(not groups[key] for key in required):
        return []
    for candidates in groups.values():
        candidates.sort(key=lambda item: item[1])
    ranked = sorted((item for members in groups.values() for item in members), key=lambda item: item[1])

    def compatible(candidate, chosen):
        frame = candidate[0]
        return all(all(frame[key] != prior[0][key] for key in ("frame_id", "source_sha256", "timestamp_s"))
                   for prior in chosen)

    def cover(chosen, pending):
        if not pending:
            if len(chosen) >= 2:
                return chosen
            # Backtrack past a sharp view that has no independent partner.
            partner = next((item for item in ranked if compatible(item, chosen)), None)
            return [*chosen, partner] if partner is not None else None
        if len(chosen) >= limit:
            return None
        options = {key: [item for item in groups[key] if compatible(item, chosen)] for key in pending}
        key = min(options, key=lambda name: (len(options[name]), name))
        for candidate in options[key]:
            found = cover([*chosen, candidate], pending - {key})
            if found is not None:
                return found
        return None

    chosen = cover([], required)
    if chosen is None:
        return []
    for candidate in ranked:
        if len(chosen) >= limit:
            break
        if compatible(candidate, chosen):
            chosen.append(candidate)
    if len(chosen) < 2:
        return []
    return [dict(frame) for frame, _ in sorted(chosen, key=lambda item: item[1])]
