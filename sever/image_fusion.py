"""Deterministic corroboration of direct image readings from one visible leaf.

Only actual evidence frame identities count as separate views. This module
does not inspect OCR, repair source text, call a model, or read expected values.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import re
import unicodedata

from .schema import DocumentResult, Evidence, FIELDS_BY_TYPE, FieldValue


def _evidence(items: list[Evidence]) -> list[Evidence]:
    """Keep literal proofs without multiplying identical citations."""
    unique = {}
    for item in items:
        unique.setdefault(json.dumps(item.model_dump(mode="json"), sort_keys=True, ensure_ascii=False), item)
    return [item.model_copy(deep=True) for item in unique.values()]


def _frames(items: list[Evidence]) -> set[str]:
    return {item.frame_id for item in items if item.frame_id.strip()}


def _grounded_type_frames(reading: DocumentResult) -> set[str]:
    """Only supplied-frame type evidence can support or contradict a type.

    The raw model type and any structural failures remain in the saved reading.
    Field-specific errors do not invalidate independently grounded type evidence.
    """
    return (_frames(reading.classification_evidence)
            if reading.document_type != "unknown" else set())


def _literal_comparison(value: str) -> str:
    """Compare only Unicode, letter case and whitespace, independent of field."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_ORGANIZATION_FIELDS = frozenset({"consignor", "consignee", "supplier", "customer"})
_ADDRESS_FIELDS = frozenset({"consignor_address", "consignee_address"})
_QUOTE_GLYPHS = str.maketrans({character: '"' for character in '«»„“”‟″'}
                            | {character: "'" for character in "‘’‚‛"})
_TEXT_TOKENS = re.compile(r"[^\W_]+|_|[^\w\s]", re.UNICODE)
_SHORT_WORD_PERIOD = re.compile(r"(?<!\w)([^\W\d_]{1,4})\.(?=$|[\s,;\"'\)\]])", re.UNICODE)
_ADDRESS_WORD_PERIOD = re.compile(r"(?<!\w)([^\W\d_]{1,4})\.(?=\d)", re.UNICODE)


def _field_comparison(name: str, value: str) -> str:
    """Ignore presentation differences only in party names and addresses.

    Keep word boundaries, every letter/digit, internal periods, hyphens,
    slashes and separators next to numeric components. Numbers and dates use the unchanged
    literal comparison. This key never replaces a model's original value.
    """
    text = _literal_comparison(value)
    if name not in _ORGANIZATION_FIELDS | _ADDRESS_FIELDS:
        return text
    text = text.translate(_QUOTE_GLYPHS)
    text = _SHORT_WORD_PERIOD.sub(r"\1", text)
    if name in _ADDRESS_FIELDS:
        # A printed short prefix attached to a number keeps both tokens:
        # "корп.16" equals "корп 16", never "корп16" or "соор 16".
        text = _ADDRESS_WORD_PERIOD.sub(r"\1 ", text)
    tokens = _TEXT_TOKENS.findall(text)
    kept = []
    quoted = False
    for index, token in enumerate(tokens):
        if token == '"':
            quoted = not quoted
        if name in _ADDRESS_FIELDS and not quoted and token in {",", ";"}:
            # Only alphabetic address-component separators may vary. Keep
            # quoted-name punctuation and every separator next to a number,
            # including "12, А" versus "12 А". Quotes cannot hide a number.
            left = index - 1
            right = index + 1
            while left >= 0 and tokens[left] in {'"', "'", "(", ")", "[", "]"}:
                left -= 1
            while right < len(tokens) and tokens[right] in {'"', "'", "(", ")", "[", "]"}:
                right += 1
            if left >= 0 and right < len(tokens):
                before, after = tokens[left], tokens[right]
                if before.isalpha() and after.isalpha():
                    continue
        kept.append(token)
    return " ".join(kept)


def _blank_frames(items: list[Evidence]) -> set[str]:
    # Count the model's not_present claim only on actual located source views.
    # This does not inspect whether the pixels are semantically blank.
    return {item.frame_id for item in items if item.frame_id.strip()
            and item.bbox is not None and len(item.bbox) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in item.bbox)
            and item.bbox[2] > 0 and item.bbox[3] > 0
            and item.bbox[0] >= 0 and item.bbox[1] >= 0
            and item.bbox[0] + item.bbox[2] <= 1.000001
            and item.bbox[1] + item.bbox[3] <= 1.000001}


def fuse_readings(readings: list[DocumentResult], *, source_id: str,
                  page_id: str) -> tuple[DocumentResult, dict]:
    """Fuse same-leaf readings conservatively, preserving every raw transcript.

    Two distinct image views must support the type and each recognized field.
    Unknown or ungrounded-type views do not veto a supported type; disagreeing
    types with actual supplied-frame evidence do.
    Model confidence is not converted into a calibrated consensus probability.
    Matching not_present model statuses require located evidence on two views.
    Presentation equivalence for party names/addresses changes comparison
    keys only; no parser changes or rejects the model's raw value.
"""
    if any(item.source_id != source_id or item.page_id != page_id for item in readings):
        raise ValueError("Image fusion requires readings of the same source and visible leaf")
    classification = _evidence([e for item in readings for e in item.classification_evidence])
    notes = list(dict.fromkeys(note for item in readings for note in item.notes))
    errors = list(dict.fromkeys(error for item in readings for error in item.errors))
    by_type: dict[str, list[DocumentResult]] = defaultdict(list)
    type_frames: dict[str, set[str]] = defaultdict(set)
    all_frames = set()
    for item in readings:
        all_frames.update(_frames(item.classification_evidence))
        for field in item.fields.values():
            all_frames.update(_frames(field.evidence))
        grounded_frames = _grounded_type_frames(item)
        # A missing heading citation cannot erase separately located field
        # readings. Keep the model's declared field contract, while only actual
        # heading evidence contributes to form confirmation or disagreement.
        if item.document_type != "unknown":
            by_type[item.document_type].append(item)
        if grounded_frames:
            type_frames[item.document_type].update(grounded_frames)
    candidates = sorted(type_frames)
    kind = candidates[0] if len(candidates) == 1 else "unknown"
    type_confirmed = kind != "unknown" and len(type_frames[kind]) >= 2
    type_status = "confirmed" if type_confirmed else "pending" if candidates else "unconfirmed"
    if len(candidates) > 1:
        type_status = "conflict"
        errors.append("image_fusion:conflicting_document_types")
        notes.append("Direct image readings disagree on the supported document type; no type or fields were selected")
    elif not type_confirmed:
        notes.append("The model's document type is not corroborated by two distinct source evidence frame IDs")
    meta = {
        "method": "Same visible leaf; distinct source evidence frame IDs; Unicode/case/whitespace comparison with quote, abbreviation-period and component-punctuation presentation equivalence for party names/addresses only; no OCR, semantic repair or expected values",
        "minimum_distinct_frames": 2,
        "type_confirmation": {"status": type_status, "confirmed": type_confirmed,
            "candidate_type": kind if kind != "unknown" else None,
            "frame_ids_by_type": {name: sorted(type_frames[name]) for name in candidates}},
        "field_confirmations": {}, "remaining_fields": [],
        "readings": [item.model_dump(mode="json") for item in readings],
        "coverage": {"reading_count": len(readings), "distinct_frame_count": len(all_frames),
                     "frame_ids": sorted(all_frames), "confirmed_fields": 0, "required_fields": 0},
    }
    if not type_confirmed:
        meta["remaining_fields"] = list(FIELDS_BY_TYPE[kind]) if kind != "unknown" else []
        meta["coverage"]["required_fields"] = len(meta["remaining_fields"])
        return DocumentResult(source_id=source_id, page_id=page_id, document_type="unknown",
            classification_evidence=classification, notes=notes, errors=errors), meta

    fields = {}
    for name in FIELDS_BY_TYPE[kind]:
        observed = [item.fields[name] for item in by_type[kind]]
        evidence = _evidence([e for field in observed for e in field.evidence])
        values: dict[str, list[FieldValue]] = defaultdict(list)
        value_frames: dict[str, set[str]] = defaultdict(set)
        absence_frames = set()
        explicit_conflict = False
        for field in observed:
            if field.status == "recognized" and field.raw_value:
                key = _field_comparison(name, field.raw_value)
                frames = _frames(field.evidence)
                if frames:
                    values[key].append(field)
                    value_frames[key].update(frames)
            elif field.status == "not_present":
                absence_frames.update(_blank_frames(field.evidence))
            elif field.status == "conflict" and _frames(field.evidence):
                explicit_conflict = True
        conflict = explicit_conflict or len(values) > 1 or bool(values and absence_frames)
        if conflict:
            status = "conflict"
            reason = "Direct image readings disagree on this own field; all literal readings and evidence are preserved"
            fused = FieldValue(status=status, reason=reason, evidence=evidence)
        elif values and len(next(iter(value_frames.values()))) >= 2:
            original = next(iter(values.values()))[0]
            status = "recognized"
            reason = ("Model raw values agree across at least two distinct image views after "
                      "Unicode/case/whitespace and party-name/address presentation comparison; "
                      "all original raw values are preserved")
            fused = FieldValue(status=status, raw_value=original.raw_value,
                normalized_value=original.normalized_value, confidence=None,
                reason=reason, evidence=evidence)
        elif not values and len(absence_frames) >= 2:
            status = "not_present"
            reason = "Model readings agree this field is not present across at least two distinct image views"
            fused = FieldValue(status=status, reason=reason, evidence=evidence)
        else:
            status = "unreadable"
            reason = ("A literal value has only one distinct supporting image view; another view is required"
                      if values else "An absence claim has only one located blank image view; another view is required"
                      if absence_frames else "No corroborated literal value or two located blank own-field views")
            fused = FieldValue(status=status, reason=reason, evidence=evidence)
        fields[name] = fused
        confirmed = status in {"recognized", "not_present"}
        meta["field_confirmations"][name] = {
            "status": status, "confirmed": confirmed,
            "value_groups": [{"comparison": key, "frame_ids": sorted(value_frames[key]),
                              "raw_values": list(dict.fromkeys(value.raw_value for value in group))}
                             for key, group in values.items()],
            "blank_frame_ids": sorted(absence_frames), "reason": reason,
        }
        if confirmed:
            meta["coverage"]["confirmed_fields"] += 1
        else:
            meta["remaining_fields"].append(name)
    meta["coverage"]["required_fields"] = len(fields)
    return DocumentResult(source_id=source_id, page_id=page_id, document_type=kind,
        classification_evidence=classification, fields=fields, notes=notes, errors=errors), meta
