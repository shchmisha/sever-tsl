"""Same-leaf eligibility and evidence-preserving multi-frame fusion."""
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image

from .normalize import comparison
from .schema import DocumentResult, FieldValue, FIELDS_BY_TYPE


def unresolved_fields(document: DocumentResult, expected_type: str) -> list[str]:
    if document.document_type != expected_type:
        return list(FIELDS_BY_TYPE[expected_type])
    return [name for name, field in document.fields.items() if field.status in {"unreadable", "conflict"}]


def fields_needing_views(document: DocumentResult, expected_type: str) -> list[str]:
    """Read unresolved fields and corroborate answers in an independent source frame."""
    if document.document_type != expected_type:
        return list(FIELDS_BY_TYPE[expected_type])
    return [name for name, value in document.fields.items()
            if value.status in {"unreadable", "conflict"}
            or len({e.frame_id for e in value.evidence}) < 2]


def fields_to_read(document: DocumentResult, expected_type: str) -> list[str]:
    """A grounded dispute has its source evidence and awaits explicit review.

    Additional model votes cannot resolve it under this pipeline's rules.
    Keep reading other missing/unconfirmed fields without a frame-count cap.
    """
    return [name for name in fields_needing_views(document, expected_type)
            if name not in document.fields or document.fields[name].status != "conflict"]


def _identity(ocr: dict, kind: str) -> dict[str, str]:
    from .vision import _local_header_readings, _role_sources
    panels = _role_sources(ocr, kind)
    result = {}
    names = ("invoice_number", "invoice_date") if kind == "upd" else ("document_number", "document_date")
    for name in names:
        values = {comparison(name, raw) for raw, _ in _local_header_readings(ocr, kind, name, panels)} - {None}
        if len(values) == 1:
            result[name] = values.pop()
    return result


def _address_key(value: str | None) -> str | None:
    if value is None:
        return None
    # Typographic numero marks are formatting, not house/unit digits.
    value = re.sub(r"(?:№|\bN[oº](?=\s*\d))", "№", value, flags=re.I)
    return comparison("consignor_address", value)


def _address_continuation(short: FieldValue, long: FieldValue) -> bool:
    """An exact address prefix ending in a printed continuation separator.

    The shorter view does not corroborate the missing suffix. Only the longer
    source reading supplies it; differing words/digits remain real conflicts.
    """
    if not short.raw_value or not short.raw_value.rstrip().endswith((",", ";")):
        return False
    a, b = _address_key(short.raw_value), _address_key(long.raw_value)
    return bool(a and b and b.startswith(a + " "))


def neighbor_eligibility(primary: dict, neighbor: dict, kind: str, needed: list[str]) -> dict:
    """Require registered paper and compatible own headers, never adjacency alone."""
    from .video import _features, _pair_evidence
    from .vision import _heading_supports, _role_sources, _local_header_readings, _wrapped_upd_titles
    from .classifier import _clipped_upd_header, _own_forms, _vertical_form_anchor
    from .schema import Evidence
    if primary["frame_id"] == neighbor["frame_id"]:
        return {"eligible": False, "reason": "same_frame"}
    identities = [_identity(page, kind) for page in (primary, neighbor)]
    contradictions = [name for name in identities[0].keys() & identities[1].keys()
                      if identities[0][name] != identities[1][name]]
    if contradictions:
        return {"eligible": False, "reason": "different_own_header_values", "contradictions": contradictions,
                "identities": identities}
    # A nearby unrelated caption/footer cannot authorize cross-document fusion.
    regions = neighbor.get("regions", [])
    own_headings = []
    # These are exact printed words from one same-frame OCR pass; original
    # regions remain available to field validation and the registration below.
    heading_candidates = [*regions, *_wrapped_upd_titles(neighbor)]
    for region in heading_candidates:
        text = region.get("text", "")
        evidence = Evidence(frame_id=neighbor["frame_id"], text=text, bbox=region.get("bbox"))
        if region.get("bbox", [0, 1])[1] >= .4 or _vertical_form_anchor(evidence):
            continue
        if re.match(r"\s*(?:приложени\w*\s+к|согласно|основани\w*)\b", text, re.I):
            continue
        clipped = kind == "upd" and _clipped_upd_header(evidence)
        own_code = kind == "torg12" and re.search(r"\b0330212\b", text)
        if clipped or (kind in _own_forms(text) and (_heading_supports(kind, text) or own_code)):
            own_headings.append(region)
    if not own_headings:
        return {"eligible": False, "reason": "neighbor_own_form_heading_not_confirmed"}
    panels = _role_sources(neighbor, kind)
    candidates = []
    for name in needed:
        role = "document_header" if name.endswith(("number", "date")) else name.removesuffix("_address")
        panel = panels.get(role)
        if panel and panel.get("ids"):
            if role == "document_header":
                readings = _local_header_readings(neighbor, kind, name, panels)
                label_pattern = (r"\bдата\b|\bот\s*(?:\d|[«\"_])" if name.endswith("date") else
                                 r"сч[её]т.?фактур|номер\s+документ|[№N][ºoо°]?(?:\s|$)")
                own_label = any(re.search(label_pattern, regions[int(i[1:])].get("text", ""), re.I)
                                for i in panel["ids"])
                if not readings and not own_label:
                    continue
            # This is only source availability; the reader still decides blank/illegible.
            candidates.append(name)
    if not candidates:
        return {"eligible": False, "reason": "no_needed_field_source_in_neighbor"}
    features = []
    for page in (primary, neighbor):
        with Image.open(Path(page["processing"]["source_path"])) as image:
            height = max(2, round(image.height * 384 / image.width / 2) * 2)
            small = image.convert("RGB").resize((384, height), Image.Resampling.LANCZOS)
            features.append(_features(small))
    registered = _pair_evidence(*features)
    if not registered["same_page"]:
        return {"eligible": False, "reason": "same_leaf_registration_not_confirmed", "registration": registered}
    return {"eligible": True, "reason": "registered_same_leaf_with_needed_source",
            "registration": registered, "identities": identities, "candidate_fields": candidates}


def merge_documents(primary: DocumentResult, neighbor: DocumentResult, expected_type: str) -> tuple[DocumentResult, dict]:
    if primary.source_id != neighbor.source_id or primary.page_id != neighbor.page_id:
        raise ValueError("Cannot combine different sources or leaf identifiers")
    summary = {"recovered_fields": [], "conflicting_fields": [], "agreement_fields": [], "partial_observations": []}
    if neighbor.document_type != expected_type:
        summary["reason"] = "neighbor_reader_did_not_confirm_expected_type"
        return primary, summary
    if primary.document_type != expected_type:
        summary["recovered_fields"] = [name for name, value in neighbor.fields.items()
                                       if value.status in {"recognized", "not_present"}]
        summary["reason"] = "same_leaf_neighbor_confirmed_type"
        result = neighbor.model_copy(deep=True)
        result.notes.append("Own form recovered from a geometrically registered same-leaf neighbor; primary reading retained in the trace")
        return result, summary
    result = primary.model_copy(deep=True)
    for name, old in primary.fields.items():
        new = neighbor.fields[name]
        if old.status == "conflict":
            # A third reading is not permission to erase a real disagreement.
            # Preserve all source views for explicit later review.
            for evidence in new.evidence:
                if evidence not in result.fields[name].evidence:
                    result.fields[name].evidence.append(evidence)
        elif old.status == "unreadable" and new.status in {"recognized", "not_present"}:
            result.fields[name] = new.model_copy(deep=True)
            result.fields[name].reason = "Recovered from a registered neighboring frame of the same leaf; " + (new.reason or "grounded source reading")
            summary["recovered_fields"].append(name)
        elif old.status in {"recognized", "not_present"} and new.status in {"recognized", "not_present"}:
            key = _address_key if name.endswith("address") else lambda value: comparison(name, value)
            old_key, new_key = key(old.normalized_value or old.raw_value), key(new.normalized_value or new.raw_value)
            agrees = (old.status == new.status == "not_present" or
                      old.status == new.status == "recognized" and old_key is not None and old_key == new_key)
            if agrees:
                summary["agreement_fields"].append(name)
                for evidence in new.evidence:
                    if evidence not in result.fields[name].evidence:
                        result.fields[name].evidence.append(evidence)
            elif name.endswith("address") and old.status == new.status == "recognized" and _address_continuation(new, old):
                summary["partial_observations"].append({"field": name, "frame_ids": sorted({e.frame_id for e in new.evidence}),
                    "reason": "Exact prefix with continuation separator; missing suffix is not independent confirmation"})
            elif name.endswith("address") and old.status == new.status == "recognized" and _address_continuation(old, new):
                result.fields[name] = new.model_copy(deep=True)
                summary["recovered_fields"].append(name)
                summary["partial_observations"].append({"field": name, "frame_ids": sorted({e.frame_id for e in old.evidence}),
                    "reason": "Longer source reading supplies the suffix missing from the exact prefix"})
            else:
                result.fields[name] = FieldValue(status="conflict", evidence=[*old.evidence, *new.evidence],
                    reason="Two registered same-leaf frames give different grounded values; no value selected")
                summary["conflicting_fields"].append(name)
        elif new.status == "conflict":
            result.fields[name] = FieldValue(status="conflict", evidence=[*old.evidence, *new.evidence],
                reason="A grounded same-leaf reading reports incompatible source observations; no value selected")
            summary["conflicting_fields"].append(name)
    recovered = set(summary["recovered_fields"])
    result.errors = [error for error in result.errors if not any(error.startswith(name + ":") for name in recovered)]
    result.errors.extend(f"{name}: neighboring_frame_value_conflict" for name in summary["conflicting_fields"])
    note = "Same-leaf multi-frame comparison; source evidence retained and disagreements left unresolved"
    if note not in result.notes:
        result.notes.append(note)
    return result, summary
