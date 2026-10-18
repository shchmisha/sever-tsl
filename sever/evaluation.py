"""Provisional one-to-one visible-leaf evaluation, with annotation coverage explicit."""
from collections import Counter, defaultdict
import math

from .normalize import comparison
from .schema import FIELDS_BY_TYPE

NOT_VISIBLE = "not_visible_on_this_leaf"


def wilson_interval(correct: int, total: int, z: float = 1.959964) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when nothing was measured."""
    if total <= 0:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value for b A-only and c B-only successes; None without discordant pairs."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _metric(correct: int, total: int, **extra) -> dict:
    return {"correct": correct, "total": total, "rate": correct / total if total else None, **extra}


def _intervals(report: dict) -> dict:
    """95% Wilson intervals for every correct/total metric, kept beside the metrics."""
    names = ("candidate_coverage", "detection", "type_accuracy_all_expected", "type_accuracy_targets",
             "e2e", "e2e_on_fully_labeled", "unknown_rejection", "abstention_checks", "labeled_fields")
    return {name: wilson_interval(report[name]["correct"], report[name]["total"]) for name in names if name in report}


def _primary_frame(page: dict) -> dict | None:
    frames = page.get("frames", [])
    selected = page.get("selected_frame_id")
    if selected is not None:
        # A missing declared primary frame must not silently match via a neighbor.
        return next((frame for frame in frames if frame.get("frame_id") == selected), None)
    # Compatibility with reference-PDF and older results which have one frame.
    return frames[0] if frames else None


def _processing_failures(prediction: dict) -> list[str]:
    # Native image reading reports operational failures separately from semantic
    # form validation. Match the code before optional frame/exception suffixes;
    # a visible out-of-scope form or uncertain heading can still be a valid
    # unknown prediction, whereas an unread image cannot earn rejection credit.
    native_codes = {"quality_gate_rejected", "image_reading_failed",
                    "model_budget_or_provider_failure", "additional_frame_pool_failed",
                    "conflict_review_failed", "conflict_review_budget_or_provider_failure",
                    "type_conflict_review_failed", "type_conflict_review_budget_or_provider_failure"}

    def native_failure(error: str) -> bool:
        namespace, separator, reason = error.partition(":")
        return bool(separator) and namespace == "native_image" and reason.strip().split(":", 1)[0] in native_codes

    return [error for error in prediction.get("errors", []) if isinstance(error, str) and (
        error == "ocr_failed" or error.startswith("ocr_failed:") or error.startswith("quality_rejected:") or
        error == "extraction_failed" or error.startswith("extraction_failed:") or
        error.startswith("classification: no OCR evidence;") or
        error.startswith("classification: invalid evidence;") or
        error.startswith("classification: foreign or invalid OCR evidence ids rejected") or
        native_failure(error)
    )]


def _value_matches(name: str, found, target, value) -> bool:
    variants = [value]
    if isinstance(target, dict) and isinstance(target.get("accepted_variants"), list):
        variants.extend(target["accepted_variants"])
    if found is None:
        return value is None
    observed = comparison(name, found)
    return observed is not None and any(
        variant is not None and observed == comparison(name, variant) for variant in variants
    )


def _selected_frame_quality(actual: list[dict]) -> dict:
    """Summarize existing indicators without inventing quality thresholds or accuracy."""
    flags, visibility, glare, hand = Counter(), Counter(), Counter(), Counter()
    inspected = missing_frame = missing_quality = 0
    for page in actual:
        frame = _primary_frame(page)
        if frame is None:
            missing_frame += 1
            continue
        quality = frame.get("quality")
        if not isinstance(quality, dict) or not quality:
            missing_quality += 1
            continue
        inspected += 1
        flags.update(set(quality.get("limitations", [])))
        for counts, value in (
            (visibility, quality.get("full_page_visible")),
            (glare, quality.get("exposure", {}).get("glare_confirmed")),
            (hand, quality.get("possible_hand_occlusion", {}).get("hand_occlusion_confirmed")),
        ):
            counts["true" if value is True else "false" if value is False else "unknown"] += 1
    return {
        "method": "Existing heuristic flags on each prediction's primary selected frame; one frame per candidate leaf. Missing quality and unconfirmed glare/occlusion remain unknown.",
        "candidate_leaves": len(actual), "frames_with_quality": inspected,
        "missing_primary_frame": missing_frame, "missing_quality": missing_quality,
        "flags": dict(flags), "full_page_visible": dict(visibility),
        "glare_confirmed": dict(glare), "hand_occlusion_confirmed": dict(hand),
        "limitations": ["These are shooting/selection indicators, not calibrated quality probabilities or field accuracy.",
                        "A paper-outline flag does not establish field legibility or absence of internal occlusion."],
    }


def _label_window(item: dict, matching: str) -> tuple[float, float]:
    """The time window a prediction's primary frame must fall in to match this label.

    "strict" uses the hand-drawn interval. "bracket" widens it to the label's own
    boundary brackets (midpoints of adjacent sampled frames are approximate), so a
    primary frame just outside an approximate boundary still counts as this sheet.
    """
    start, end = item["start_s"], item["end_s"]
    if matching == "bracket":
        brackets = item.get("boundary_uncertainty") or {}
        start_bracket, end_bracket = brackets.get("start_bracket_s"), brackets.get("end_bracket_s")
        if isinstance(start_bracket, list) and len(start_bracket) == 2:
            start = min(start, float(start_bracket[0]))
        if isinstance(end_bracket, list) and len(end_bracket) == 2:
            end = max(end, float(end_bracket[1]))
    return start, end


def evaluate(result: dict, truth: dict, *, address_boundary: str = "current", matching: str = "strict") -> dict:
    """Score one run against one labelled video.

    address_boundary="legacy" scores a field against its ``legacy_value`` (the
    label under the pre-2026-10-11 postal-only boundary) when the label has one;
    use it only for runs whose instructions predate that rule.
    matching="bracket" lets a primary frame inside a label's boundary bracket
    match that label when no label's core interval contains it; the strict
    one-to-one rule and the preference for the core interval are kept.
    """
    if address_boundary not in {"current", "legacy"}:
        raise ValueError("address_boundary must be 'current' or 'legacy'")
    if matching not in {"strict", "bracket"}:
        raise ValueError("matching must be 'strict' or 'bracket'")
    if truth.get("source_sha256") and result.get("source", {}).get("sha256") != truth["source_sha256"]:
        raise ValueError("Ground truth source hash does not match the processed video")
    expected = truth.get("pages", [])
    actual = result.get("pages", [])
    primary_frames = [_primary_frame(page) for page in actual]
    used: set[int] = set()
    detected = typed = target_typed = success = fully_labeled_total = 0
    field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    errors = Counter()
    details = []
    # Rows a human marked as not visible on this leaf are abstention checks: the
    # system must not claim a value or a blank for them. Undetected intervals and
    # rows pending a semantic decision are counted separately, never as passes.
    abstention = {"correct": 0, "total": 0, "undetected": 0, "semantic_pending": 0,
                  "failures": [], "by_field": defaultdict(lambda: {"correct": 0, "total": 0})}
    target_total = unknown_total = unknown_correct = 0
    bracket_matches = []
    for item in expected:
        start, end = item["start_s"], item["end_s"]
        choices = []
        for i, frame in enumerate(primary_frames):
            if i in used or frame is None:
                continue
            timestamp = frame.get("timestamp_s")
            if timestamp is not None and start <= timestamp < end:
                choices.append((abs((start + end) / 2 - timestamp), i))
        index = min(choices)[1] if choices else None
        if index is None and matching == "bracket":
            # A primary inside this label's boundary bracket may be taken only
            # when the prediction's own leaf interval overlaps this label more
            # than any other label whose core contains that primary, so a
            # neighbour sheet is never stolen through an approximate boundary.
            wide_start, wide_end = _label_window(item, matching)
            fallback = []
            for i, frame in enumerate(primary_frames):
                if i in used or frame is None:
                    continue
                timestamp = frame.get("timestamp_s")
                if timestamp is None or not wide_start <= timestamp < wide_end:
                    continue
                page = actual[i]
                p_start, p_end = page.get("start_s"), page.get("end_s")

                def overlap(label):
                    if p_start is None or p_end is None:
                        return 0.0
                    return max(0.0, min(float(p_end), label["end_s"]) - max(float(p_start), label["start_s"]))
                owners = [other for other in expected if other is not item and other["start_s"] <= timestamp < other["end_s"]]
                if all(overlap(item) > overlap(other) for other in owners):
                    fallback.append((abs((start + end) / 2 - timestamp), i))
            if fallback:
                index = min(fallback)[1]
                bracket_matches.append({"truth_page_id": item.get("page_id"),
                                        "primary_timestamp_s": primary_frames[index].get("timestamp_s"),
                                        "prediction_interval_s": [actual[index].get("start_s"), actual[index].get("end_s")],
                                        "core_interval_s": [start, end], "bracket_window_s": [wide_start, wide_end]})
        prediction = actual[index].get("result", {}) if index is not None else {}
        failure = _processing_failures(prediction)
        if index is not None:
            used.add(index)
            detected += 1
        kind = item["document_type"]
        type_ok = not failure and prediction.get("document_type") == kind
        typed += int(type_ok)
        if kind == "unknown":
            unknown_total += 1
            unknown_correct += int(type_ok)
        else:
            target_total += 1
            target_typed += int(type_ok)
        checks = {}
        abstentions = {}
        # Labeled fields retain misses and wrong types in their denominators.
        for name, target in item.get("fields", {}).items():
            if isinstance(target, dict) and target.get("labeled", True) is False:
                if target.get("manual_visibility") != NOT_VISIBLE:
                    abstention["semantic_pending"] += 1
                    continue
                if index is None:
                    abstention["undetected"] += 1
                    continue
                found = prediction.get("fields", {}).get(name, {})
                claimed = found.get("status") in {"recognized", "not_present"}
                abstentions[name] = not claimed
                abstention["total"] += 1
                abstention["by_field"][f"{kind}.{name}"]["total"] += 1
                if claimed:
                    abstention["failures"].append({"truth_page_id": item.get("page_id"), "field": name,
                        "predicted_type": prediction.get("document_type"), "status": found.get("status"),
                        "raw_value": (found.get("raw_value") or "")[:120] or None})
                else:
                    abstention["correct"] += 1
                    abstention["by_field"][f"{kind}.{name}"]["correct"] += 1
                continue
            value = target.get("value") if isinstance(target, dict) else target
            if address_boundary == "legacy" and isinstance(target, dict) and "legacy_value" in target:
                value = target["legacy_value"]
            wanted_status = target.get("status", "recognized" if value is not None else "not_present") if isinstance(target, dict) else ("recognized" if value is not None else "not_present")
            found = prediction.get("fields", {}).get(name, {})
            ok = type_ok and found.get("status") == wanted_status and _value_matches(name, found.get("normalized_value"), target, value)
            checks[name] = ok
            field_stats[f"{kind}.{name}"]["total"] += 1
            field_stats[f"{kind}.{name}"]["correct"] += int(ok)
            if not ok:
                errors[f"field:{name}:{found.get('status', 'missing')}"] += 1
        complete = set(item.get("fields", {})) == set(FIELDS_BY_TYPE[kind]) and all(
            not isinstance(value, dict) or value.get("labeled", True)
            for value in item.get("fields", {}).values()
        )
        e2e = index is not None and type_ok and complete and all(checks.values())
        if kind != "unknown":
            fully_labeled_total += int(complete)
            success += int(e2e)
        if index is None:
            errors["missed_leaf"] += 1
        elif failure:
            errors["processing_failed"] += 1
        elif not type_ok:
            errors["wrong_type"] += 1
        if index is not None:
            errors.update(prediction.get("errors", []))
        frame = primary_frames[index] if index is not None else None
        details.append({
            "truth_page_id": item.get("page_id"),
            "prediction_page_id": actual[index].get("page_id") if index is not None else None,
            "primary_frame_id": frame.get("frame_id") if frame else None,
            "primary_timestamp_s": frame.get("timestamp_s") if frame else None,
            "detected": index is not None, "processing_failed": bool(failure),
            "type_correct": bool(type_ok), "field_checks": checks, "fully_labeled": complete,
            "e2e_evaluated": kind != "unknown" and complete,
            "e2e_success": bool(e2e) if kind != "unknown" and complete else None,
            "abstention_checks": abstentions,
        })
    usability = truth.get("selected_frame_usability", {})
    usable_correct = usable_total = usable_targets_correct = usable_targets_total = usable_uncertain = 0
    target_page_ids = {item.get("page_id") for item in expected if item["document_type"] != "unknown"}
    for detail in details:
        annotation = usability.get(detail.get("truth_page_id"))
        if not isinstance(annotation, dict) or annotation.get("independently_reviewed") is not True:
            continue
        # An annotation binds the judgement to this actual chosen frame. It is
        # not inferred from successful recognition, confidence or timestamp alone.
        if annotation.get("selected_frame_id") != detail.get("primary_frame_id"):
            continue
        usable_total += 1
        usable_correct += int(detail["detected"] and annotation.get("usable") is True)
        usable_uncertain += int(annotation.get("usable") is None)
        if detail.get("truth_page_id") in target_page_ids:
            usable_targets_total += 1
            field_scope = annotation.get("assessment") != "form_rejection"
            usable_targets_correct += int(detail["detected"] and annotation.get("usable") is True and field_scope)
    for counts in field_stats.values():
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else None
    conservative = _metric(
        success, target_total, unlabeled_targets=target_total - fully_labeled_total,
        interpretation="Conservative lower bound over all target leaf intervals; unlabeled targets are not measured extraction failures.",
    )
    processing_failures = Counter(error for page in actual for error in _processing_failures(page.get("result", {})))
    report = {
        "ground_truth_provenance": truth.get("provenance", "independent manual provisional"),
        "address_boundary": address_boundary,
        "matching": matching, "bracket_matches": bracket_matches,
        "customer_approved": truth.get("customer_approved", False),
        "unit": "visible leaf interval, including repeat showings; one-to-one match by primary selected-frame timestamp",
        "frame_document_detection_rate": _metric(usable_correct, usable_total,
            population_expected_leaves=len(expected), annotation_coverage=usable_total / len(expected) if expected else None,
            uncertain=usable_uncertain, confirmed_unusable=usable_total - usable_correct - usable_uncertain,
            interpretation="Confirmed-usable lower bound from independent frame reviews plus primary-frame availability misses. Null judgments stay uncertain. Unreviewed intervals are excluded. Contractual acceptance requires an agreed sample and customer-approved labels."),
        "frame_document_detection_rate_targets": _metric(usable_targets_correct, usable_targets_total,
            population_expected_leaves=target_total, annotation_coverage=usable_targets_total / target_total if target_total else None,
            interpretation="Target forms only: the primary frame must make the required fields readable. Readable non-target form headings cannot inflate this measure."),
        "candidate_coverage": _metric(detected, len(expected), method="One-to-one temporal coverage by primary chosen frames; this does not establish legibility."),
        "detection": _metric(detected, len(expected), method="Primary candidate frame lies in the expected leaf interval. This measures candidate coverage, not verified field legibility."),
        "type_accuracy_all_expected": _metric(typed, len(expected)),
        "type_accuracy_targets": _metric(target_typed, target_total),
        "field_accuracy": dict(field_stats),
        # Keep the historical key explicit about its conservative denominator.
        "e2e": conservative,
        "e2e_all_targets_conservative": conservative.copy(),
        "e2e_on_fully_labeled": _metric(success, fully_labeled_total, interpretation="Only fully labeled target leaf intervals; missing detections and wrong types remain failures."),
        "annotation_coverage": {"fully_labeled_targets": fully_labeled_total, "target_total": target_total,
                                "unlabeled_targets": target_total - fully_labeled_total},
        "unknown_rejection": _metric(unknown_correct, unknown_total),
        "abstention_checks": _metric(abstention["correct"], abstention["total"],
            undetected_rows=abstention["undetected"], semantic_pending_rows=abstention["semantic_pending"],
            by_field={name: {**counts, "rate": counts["correct"] / counts["total"] if counts["total"] else None}
                      for name, counts in sorted(abstention["by_field"].items())},
            failures=abstention["failures"],
            interpretation="Rows a human marked not visible on this leaf; a pass means the system claimed no value or blank for that field on the matched prediction. Wrong-type predictions pass unless they still claim the field."),
        "unmatched_predictions": len(actual) - len(used),
        "processing_failures": {"candidate_leaves": sum(bool(_processing_failures(page.get('result', {}))) for page in actual),
                                "reasons": dict(processing_failures)},
        "selected_frame_quality": _selected_frame_quality(actual),
        "failure_reasons": dict(errors), "details": details,
        "labeled_fields": _metric(sum(counts["correct"] for counts in field_stats.values()),
                                  sum(counts["total"] for counts in field_stats.values())),
        "normalization_method": "NFKC/whitespace, case/ё, canonical dates/numbers; explicit legal-form prefixes and organization/address formatting punctuation. Words, meaningful symbols and numeric address tokens are retained. Character-script aliases require per-field accepted_variants.",
        "limitations": ["Provisional labels and the visible-leaf/repeat convention require customer review.",
                        "Unlabeled required fields have no measured end-to-end outcome; the all-target score is a conservative lower bound.",
                        "Repeat showings are separate leaf intervals, not unique document identities; one prediction cannot cover two intervals.",
                        "Unmatched candidate leaves are reported separately; detection is recall-style coverage, not precision or verified usable-frame accuracy.",
                        "Confidence intervals are Wilson 95% intervals over rows, ignoring clustering by sheet and by repeat showing; they understate the true uncertainty."],
    }
    report["confidence_intervals"] = _intervals(report)
    return report


def _truth_documents(truth: dict) -> "OrderedDict[str, dict]":
    """Join labelled target sheets into documents by their explicit ``document_id``."""
    from collections import OrderedDict
    from .normalize import comparison

    documents: "OrderedDict[str, dict]" = OrderedDict()
    pages = {item.get("page_id"): item for item in truth.get("pages", [])}
    for item in truth.get("pages", []):
        kind = item["document_type"]
        if kind == "unknown":
            continue
        identity = item.get("document_id")
        if not identity:
            raise ValueError(f"Target sheet {item.get('page_id')} has no document_id")
        record = documents.setdefault(identity, {"document_id": identity, "document_type": kind, "page_ids": []})
        if record["document_type"] != kind:
            raise ValueError(f"Document {identity} mixes forms {record['document_type']} and {kind}")
        record["page_ids"].append(item.get("page_id"))
    for record in documents.values():
        kind = record["document_type"]
        fields = {}
        for name in FIELDS_BY_TYPE[kind]:
            labelled, visibility = [], []
            for page_id in record["page_ids"]:
                target = pages[page_id].get("fields", {}).get(name)
                if target is None:
                    continue
                if isinstance(target, dict) and target.get("labeled", True) is False:
                    visibility.append(target.get("manual_visibility"))
                    continue
                value = target.get("value") if isinstance(target, dict) else target
                status = target.get("status", "recognized" if value is not None else "not_present") if isinstance(target, dict) else (
                    "recognized" if value is not None else "not_present")
                labelled.append({"status": status, "value": value, "target": target, "page_id": page_id})
            if labelled:
                distinct = {(entry["status"], comparison(name, entry["value"]) if entry["value"] else None) for entry in labelled}
                fields[name] = ({"kind": "labelled", **labelled[0], "page_ids": [entry["page_id"] for entry in labelled]}
                                if len(distinct) == 1 else {"kind": "label_conflict", "labels": labelled})
            elif visibility and all(value == NOT_VISIBLE for value in visibility):
                fields[name] = {"kind": "abstention"}
            else:
                fields[name] = {"kind": "pending"}
        record["fields"] = fields
    return documents


def evaluate_documents(result: dict, truth: dict, *, matching: str = "bracket") -> dict:
    """Score the per-document table: one truth document (form + own number) against one output document.

    Output documents are built by ``sever.documents.assemble_documents`` from the
    sheet results. A truth document is matched to the output document that holds
    most of its sheets (sheets are matched one-to-one by the sheet evaluator);
    each output document matches at most one truth document. Unmatched output
    documents are reported as extra (duplicates, splits or non-target forms).
    """
    from .documents import assemble_documents

    sheet_report = evaluate(result, truth, matching=matching)
    truth_docs = _truth_documents(truth)
    assembly = assemble_documents(result)
    output_docs = {record["document_id"]: record for record in assembly["documents"]}
    truth_of_sheet = {page_id: identity for identity, record in truth_docs.items() for page_id in record["page_ids"]}
    kinds = {item.get("page_id"): item["document_type"] for item in truth.get("pages", [])}
    truth_sheet_of_prediction = {detail["prediction_page_id"]: detail["truth_page_id"]
                                 for detail in sheet_report["details"] if detail["prediction_page_id"]}
    overlap = Counter()
    non_target_sheets = Counter()
    for identity, record in output_docs.items():
        for sheet in record["sheets"]:
            truth_page = truth_sheet_of_prediction.get(sheet["page_id"])
            if truth_page in truth_of_sheet:
                overlap[(identity, truth_of_sheet[truth_page])] += 1
            elif truth_page is not None and kinds.get(truth_page) == "unknown":
                non_target_sheets[identity] += 1
    order = list(output_docs)
    pairs = sorted(overlap.items(), key=lambda item: (-item[1], order.index(item[0][0])))
    matched_output, matched_truth = {}, {}
    for (output_id, truth_id), _ in pairs:
        if output_id in matched_output or truth_id in matched_truth:
            continue
        matched_output[output_id], matched_truth[truth_id] = truth_id, output_id
    touched = defaultdict(set)
    for (output_id, truth_id), _ in overlap.items():
        touched[truth_id].add(output_id)

    found = typed = 0
    field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    abstention = {"correct": 0, "total": 0, "undetected": 0, "failures": []}
    shown = {"correct": 0, "total": 0}
    e2e = fully = pending = label_conflicts = 0
    details = []
    for truth_id, record in truth_docs.items():
        kind = record["document_type"]
        output_id = matched_truth.get(truth_id)
        output = output_docs.get(output_id) if output_id else None
        found += int(output is not None)
        type_ok = output is not None and output["document_type"] == kind
        typed += int(type_ok)
        checks, abstentions = {}, {}
        complete = True
        for name, label in record["fields"].items():
            got = (output or {}).get("fields", {}).get(name) or {} if type_ok else {}
            if label["kind"] == "pending":
                pending += 1
                complete = False
                continue
            if label["kind"] == "label_conflict":
                label_conflicts += 1
                complete = False
                continue
            if label["kind"] == "abstention":
                if output is None:
                    abstention["undetected"] += 1
                    continue
                claimed = got.get("status") in {"recognized", "not_present"}
                abstentions[name] = not claimed
                abstention["total"] += 1
                abstention["correct"] += int(not claimed)
                if claimed:
                    abstention["failures"].append({"truth_document": truth_id, "field": name,
                                                   "status": got.get("status"), "raw_value": got.get("raw_value")})
                continue
            ok = (type_ok and got.get("status") == label["status"]
                  and _value_matches(name, got.get("normalized_value") if got.get("status") == "recognized" else None,
                                     label["target"], label["value"]))
            checks[name] = ok
            field_stats[f"{kind}.{name}"]["total"] += 1
            field_stats[f"{kind}.{name}"]["correct"] += int(ok)
            if got.get("status") == "recognized":
                shown["total"] += 1
                shown["correct"] += int(ok)
        fully += int(complete)
        success = complete and type_ok and all(checks.values()) and all(abstentions.values())
        e2e += int(success) if complete else 0
        details.append({"truth_document": truth_id, "document_type": kind, "truth_sheets": record["page_ids"],
                        "output_document": output_id, "output_type": output["document_type"] if output else None,
                        "output_number": output["number"] if output else None,
                        "output_sheets": [sheet["page_id"] for sheet in output["sheets"]] if output else [],
                        "split_into_output_documents": sorted(touched.get(truth_id, set())),
                        "type_correct": type_ok, "field_checks": checks, "abstention_checks": abstentions,
                        "fully_labelled": complete, "e2e_success": success if complete else None})
    extra = []
    for identity, record in output_docs.items():
        if identity in matched_output:
            continue
        shares = [truth_id for (output_id, truth_id) in overlap if output_id == identity]
        duplicate_of = next((other for other, truth_id in matched_output.items()
                             if output_docs[other]["document_type"] == record["document_type"]
                             and output_docs[other]["number"] and output_docs[other]["number"] == record["number"]), None)
        reason = ("split_from_matched_document" if shares else
                  "non_target_sheets_typed_as_target" if non_target_sheets[identity] else
                  "sheets_not_matched_to_any_labelled_sheet")
        extra.append({"output_document": identity, "document_type": record["document_type"], "number": record["number"],
                      "sheets": [sheet["page_id"] for sheet in record["sheets"]], "reason": reason,
                      "shares_sheets_with": shares, "same_number_as": duplicate_of})
    labelled = _metric(sum(value["correct"] for value in field_stats.values()), sum(value["total"] for value in field_stats.values()))
    report = {
        "unit": "document = contract form + its own number; sheets (copies, reverse, continuation, repeat showings) joined",
        "assembly_version": assembly["version"], "matching": matching,
        "truth_documents": len(truth_docs), "output_documents": len(output_docs),
        "documents_found": _metric(found, len(truth_docs)),
        "document_type_accuracy": _metric(typed, len(truth_docs)),
        "labeled_fields": labelled,
        "field_accuracy": {name: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else None}
                           for name, value in sorted(field_stats.items())},
        "abstention_checks": _metric(abstention["correct"], abstention["total"], undetected=abstention["undetected"],
                                     failures=abstention["failures"]),
        "shown_values_correct": _metric(shown["correct"], shown["total"],
                                        interpretation="Of the values printed in the document table for labelled fields, how many are right."),
        "e2e_on_fully_labeled": _metric(e2e, fully),
        "extra_output_documents": extra,
        "unscored_fields": {"pending_customer_decision": pending, "label_conflicts": label_conflicts},
        "details": details,
    }
    report["confidence_intervals"] = {name: wilson_interval(report[name]["correct"], report[name]["total"])
                                      for name in ("documents_found", "document_type_accuracy", "labeled_fields",
                                                   "abstention_checks", "shown_values_correct", "e2e_on_fully_labeled")}
    return report
