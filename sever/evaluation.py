"""Provisional one-to-one visible-leaf evaluation, with annotation coverage explicit."""
from collections import Counter, defaultdict
from .normalize import comparison
from .schema import FIELDS_BY_TYPE


def _metric(correct: int, total: int, **extra) -> dict:
    return {"correct": correct, "total": total, "rate": correct / total if total else None, **extra}


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


def evaluate(result: dict, truth: dict) -> dict:
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
    target_total = unknown_total = unknown_correct = 0
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
        # Labeled fields retain misses and wrong types in their denominators.
        for name, target in item.get("fields", {}).items():
            if isinstance(target, dict) and target.get("labeled", True) is False:
                continue
            value = target.get("value") if isinstance(target, dict) else target
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
    return {
        "ground_truth_provenance": truth.get("provenance", "independent manual provisional"),
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
        "unmatched_predictions": len(actual) - len(used),
        "processing_failures": {"candidate_leaves": sum(bool(_processing_failures(page.get('result', {}))) for page in actual),
                                "reasons": dict(processing_failures)},
        "selected_frame_quality": _selected_frame_quality(actual),
        "failure_reasons": dict(errors), "details": details,
        "normalization_method": "NFKC/whitespace, case/ё, canonical dates/numbers; explicit legal-form prefixes and organization/address formatting punctuation. Words, meaningful symbols and numeric address tokens are retained. Character-script aliases require per-field accepted_variants.",
        "limitations": ["Provisional labels and the visible-leaf/repeat convention require customer review.",
                        "Unlabeled required fields have no measured end-to-end outcome; the all-target score is a conservative lower bound.",
                        "Repeat showings are separate leaf intervals, not unique document identities; one prediction cannot cover two intervals.",
                        "Unmatched candidate leaves are reported separately; detection is recall-style coverage, not precision or verified usable-frame accuracy."],
    }
