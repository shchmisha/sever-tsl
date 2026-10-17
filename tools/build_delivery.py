#!/usr/bin/env python3
"""Build a read-only delivery snapshot from saved files; never call a provider.

Run from any directory: python tools/build_delivery.py [--root WORKSPACE]
    [--runs-dir runs/closing] [--truth analysis/ground-truth-expanded-provisional.json].
Only the catalog, summary HTML/JSON and trace-derived expense ledger are written.
Re-run after final results and metrics have been saved. Denominators are added,
not percentages averaged. Video leaf intervals and PDF reference pages stay apart.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


TARGETS = ("upd", "torg12", "ttn", "trn")
METRICS = (
    "candidate_coverage", "frame_document_detection_rate", "frame_document_detection_rate_targets",
    "type_accuracy_all_expected", "type_accuracy_targets", "unknown_rejection",
    "e2e_on_fully_labeled", "e2e_all_targets_conservative",
)
LABELS = {
    "candidate_coverage": "Покрытие интервалов выбранным кадром",
    "frame_document_detection_rate": "Пригодный кадр: независимо проверенные интервалы",
    "frame_document_detection_rate_targets": "Целевые формы: кадр с читаемыми обязательными полями",
    "type_accuracy_all_expected": "Тип: все размеченные интервалы",
    "type_accuracy_targets": "Тип: целевые интервалы",
    "unknown_rejection": "Отклонение нецелевых листов",
    "e2e_on_fully_labeled": "End-to-end: полностью размеченные целевые листы",
    "e2e_all_targets_conservative": "End-to-end: нижняя граница по всем целевым листам",
}
USAGE_KEYS = (
    "attempted_requests", "requests", "input_tokens", "output_tokens", "tool_calls",
    "actual_cost_rub", "estimated_cost_rub", "costed_requests",
)


def read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None, "expected JSON object"
        return value, None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, type(exc).__name__


def digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def trace_rows(path: Path) -> tuple[list[dict], int]:
    rows, malformed = [], 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        malformed += 1
                except ValueError:
                    malformed += 1
    except FileNotFoundError:
        pass
    return rows, malformed


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def event_timestamp(value) -> float | None:
    if finite_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def execution_snapshot(rows: list[dict], *, recovery_valid: bool = False) -> dict:
    """Describe the latest marked execution, without treating a replay as cold."""
    starts = [i for i, row in enumerate(rows) if row.get("event") == "run_started"]
    if not starts:
        return {"status": "unavailable", "mode": "not_recorded", "cold_runtime_seconds": None,
                "reason": "No run_started marker; current execution cannot be separated from history."}
    segment = rows[starts[-1]:]
    cache_names = {"cache_hit", "classifier_cache_hit", "classifier_cache_revalidated",
                   "vision_cache_hit", "vision_cache_revalidated", "vision_paid_parse_recovered"}
    cache = Counter(r.get("event") for r in segment if r.get("event") in cache_names)
    attempts = sum(r.get("event") == "model_request_started" for r in segment)
    responses = {}
    for row in segment:
        if row.get("event") not in {"model_response_received", "model_response"}:
            continue
        response_id = row.get("response_id") or row.get("provider_response_id")
        identity = {k: row.get(k) for k in ("time", "source_id", "page_id", "input_tokens", "output_tokens", "billing")}
        identity["content_sha256"] = digest(row.get("response_parts"))
        key = digest({"provider_response_id": response_id}) if response_id else digest(identity)
        responses[key] = row
    actual, estimated, unknown = 0.0, 0.0, 0
    for row in responses.values():
        billing = row.get("billing") or {}
        value = billing.get("value")
        known = billing.get("currency") == "RUB" and finite_number(value) and value >= 0
        if known and billing.get("kind") == "provider_reported":
            actual += value
        elif known and billing.get("kind") == "catalogue_estimate":
            estimated += value
        else:
            unknown += 1
    completed = [r for r in segment if r.get("event") == "run_completed"]
    recovered = [r for r in segment if r.get("event") == "exports_recovered" and r.get("source_complete_proven") is True]
    completion = recovered[-1] if recovered else completed[-1] if completed else None
    start_time = event_timestamp(segment[0].get("time"))
    end_time = event_timestamp((completion or segment[-1]).get("time"))
    elapsed = completed[-1].get("elapsed_seconds") if completed else None
    if not finite_number(elapsed) and completed and start_time is not None and end_time is not None:
        elapsed = max(end_time - start_time, 0)
    mode = ("mixed_cache_and_paid" if cache and attempts else "cached_replay_recorded" if cache else
            "no_model_cache_hits_recorded" if attempts else "no_model_activity_recorded")
    return {"status": "complete_recovered" if recovered and recovery_valid else "pending" if recovered else "complete" if completed else "pending", "mode": mode,
            "started_at": segment[0].get("time"), "completed_at": completion.get("time") if completion else None,
            "recovery": ({k: recovered[-1].get(k) for k in ("reason", "recovery_provenance_path", "results_sha256", "page_count", "replay_provider_calls", "recovery_elapsed_seconds", "replay_duration_seconds", "parent_elapsed_seconds")} if recovered else None),
            "recorded_elapsed_seconds": elapsed, "cold_runtime_seconds": None,
            "cache_events": dict(cache), "attempted_requests": attempts, "recorded_responses": len(responses),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in responses.values()),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in responses.values()),
            "actual_cost_rub": actual, "estimated_cost_rub": estimated,
            "unknown_billing_responses": unknown, "attempts_without_recorded_response": max(attempts - len(responses), 0),
            "scope": "Latest run_started trace segment only; historical expense is separate and overlaps.",
            "runtime_limit": "Recorded elapsed time may reuse model, OCR and frame caches. No controlled empty-cache cold runtime was measured."}


def expense_ledger(root: Path, generated_at: str) -> dict:
    """Deduplicate provider responses, retaining failed paid responses as expense.

    IDs are preferred when recorded. Otherwise exact event time/source/page,
    token counts, billing and a content hash identify an event copied into traces.
    Response content and error messages are never included in the output ledger.
    Request starts without response records have unknown billing, not zero cost.
    """
    responses, attempts, warnings = {}, {}, []
    rank = {"unresolved": 0, "validated": 1, "validated_with_errors": 2, "failed": 3}
    for path in sorted((root / "runs").rglob("*.jsonl")):
        rows, malformed = trace_rows(path)
        location = relative(path, root)
        if malformed:
            warnings.append({"trace": location, "unparsed_lines": malformed})
        pending: dict[tuple, list[dict]] = {}
        for index, row in enumerate(rows, 1):
            event = row.get("event")
            group = (row.get("source_id"), row.get("page_id"))
            safe = {k: row.get(k) for k in ("time", "source_id", "page_id")}
            if event == "model_request_started":
                key = digest({**safe, "page_request": row.get("page_request"),
                              "batch_attempted_requests": row.get("batch_attempted_requests")})
                entry = attempts.setdefault(key, {**safe, "request_key": key, "response_keys": [],
                                                   "outcome": "unresolved", "locations": []})
                entry["locations"].append({"trace": location, "line": index})
                pending.setdefault(group, []).append(entry)
            elif event in {"model_response_received", "model_response"}:
                response_id = row.get("response_id") or row.get("provider_response_id")
                identity = {**safe, "input_tokens": row.get("input_tokens"),
                            "output_tokens": row.get("output_tokens"), "billing": row.get("billing"),
                            "content_sha256": digest(row.get("response_parts"))}
                key = digest({"provider_response_id": response_id}) if response_id else digest(identity)
                entry = responses.setdefault(key, {**safe, "response_key": key,
                    "input_tokens": row.get("input_tokens", 0), "output_tokens": row.get("output_tokens", 0),
                    "billing": row.get("billing"), "outcome": "unresolved", "locations": []})
                entry["locations"].append({"trace": location, "line": index})
                current = pending.get(group, [])
                if current:
                    if key not in current[-1]["response_keys"]:
                        current[-1]["response_keys"].append(key)
                else:
                    # Some historical hooks saved a response without a start.
                    pending.setdefault(group, []).append({"response_keys": [key], "outcome": "unresolved"})
            elif event in {"classifier_error", "vision_error", "analysis_error",
                           "classifier_validation", "vision_reading", "validation"}:
                failure = event.endswith("_error")
                results = list((row.get("results") or {}).values()) if isinstance(row.get("results"), dict) else []
                results.append(row.get("result", {}))
                validation_errors = any(isinstance(result, dict) and bool(result.get("errors")) for result in results)
                outcome = "failed" if failure else ("validated_with_errors" if validation_errors else "validated")
                for attempt in pending.pop(group, []):
                    if rank[outcome] > rank[attempt["outcome"]]:
                        attempt["outcome"] = outcome
                    for key in attempt["response_keys"]:
                        entry = responses[key]
                        if rank[outcome] > rank[entry["outcome"]]:
                            entry["outcome"] = outcome
                        if failure:
                            entry["error_type"] = row.get("error_type")
                            entry["http_status"] = row.get("http_status")
                    if failure:
                        attempt["error_type"] = row.get("error_type")
                        attempt["http_status"] = row.get("http_status")
    rows = sorted(responses.values(), key=lambda row: str(row.get("time")))
    request_rows = sorted(attempts.values(), key=lambda row: str(row.get("time")))
    actual, estimated, unknown = 0.0, 0.0, 0
    for row in rows:
        billing = row.get("billing") or {}
        value = billing.get("value")
        known = billing.get("currency") == "RUB" and finite_number(value) and value >= 0
        if known and billing.get("kind") == "provider_reported":
            actual += value
        elif known and billing.get("kind") == "catalogue_estimate":
            estimated += value
        else:
            unknown += 1
    no_response = [r for r in request_rows if not r["response_keys"]]
    return {
        "schema_version": 2, "generated_at": generated_at,
        "scope": "All recorded development, probe, failed and final provider-response events in runs/**/*.jsonl; not a provider wallet reconciliation.",
        "deduplication": "Provider response ID where saved, otherwise exact timestamp/source/page/token/billing/content hash; trace copies are merged.",
        "currency": "RUB", "responses": rows, "requests": request_rows,
        "response_count": len(rows), "recorded_attempt_count": len(request_rows),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "provider_reported_cost_rub": actual, "catalogue_estimated_cost_rub": estimated,
        "unknown_billing_responses": unknown,
        "failed_responses": sum(r["outcome"] == "failed" for r in rows),
        "validated_responses_with_result_errors": sum(r["outcome"] == "validated_with_errors" for r in rows),
        "unresolved_response_outcomes": sum(r["outcome"] == "unresolved" for r in rows),
        "requests_without_response": len(no_response),
        "failed_requests_without_response": sum(r["outcome"] == "failed" for r in no_response),
        "unresolved_requests_without_response": sum(r["outcome"] == "unresolved" for r in no_response),
        "unknown_billing_requests_without_response": len(no_response),
        "trace_parse_warnings": warnings,
        "limitations": [
            "Responses rejected by output validation remain in paid expense totals.",
            "No response record does not prove that a provider charged zero; billing remains unknown.",
            "Unlogged requests, manual account charges and provider adjustments cannot be reconstructed here.",
            "Final run usage overlaps this historical ledger; the two totals must not be added.",
        ],
    }


def usage_snapshot(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in USAGE_KEYS} | {"currency": "RUB"}


def total_usage(entries: list[dict]) -> dict:
    totals = {key: 0 for key in USAGE_KEYS}
    present = 0
    for entry in entries:
        value = entry.get("final_usage")
        if value is None:
            continue
        present += 1
        for key in USAGE_KEYS:
            if finite_number(value.get(key)):
                totals[key] += value[key]
    return {**totals, "currency": "RUB", "runs_with_usage": present, "runs_without_usage": len(entries) - present,
            "responses_without_reported_provider_cost": max(totals["requests"] - totals["costed_requests"], 0),
            "attempts_without_recorded_response": max(totals["attempted_requests"] - totals["requests"], 0),
            "scope": "Per-run results.usage only; cumulative batch_usage is excluded; cache reuse does not erase historical expense."}


def metric_snapshot(metrics: dict) -> dict:
    selected = {key: metrics.get(key) for key in METRICS}
    if selected["candidate_coverage"] is None and isinstance(metrics.get("detection"), dict):
        selected["candidate_coverage"] = metrics["detection"] | {"legacy_name": "detection"}
    for key in ("field_accuracy", "annotation_coverage", "unmatched_predictions", "processing_failures",
                "selected_frame_quality", "failure_reasons", "normalization_method", "limitations", "unit"):
        selected[key] = metrics.get(key)
    return selected


def truth_counts(truth: dict) -> dict:
    pages = truth.get("pages", [])
    fields = truth.get("field_contract", {})
    types = Counter(page.get("document_type", "unknown") for page in pages)
    fully_labeled = 0
    for page in pages:
        names = fields.get(page.get("document_type"), [])
        # Contract supplied by caller for videos, explicit fields for PDF truth.
        if not names and page.get("document_type") in TARGETS:
            names = list(page.get("fields", {}))
        if names and all(page.get("fields", {}).get(name, {}).get("labeled", False) for name in names):
            fully_labeled += 1
    return {"leaf_intervals": len(pages), "types": dict(sorted(types.items())),
            "target_intervals": sum(types[t] for t in TARGETS),
            "fully_labeled_target_intervals": fully_labeled,
            "labeled_fields": sum(bool(field.get("labeled")) for page in pages for field in page.get("fields", {}).values()),
            "independently_reviewed_selected_frames": len(truth.get("selected_frame_usability", {}))}


def expected_field_denominators(truth: dict) -> dict[str, int]:
    counts = Counter()
    for page in truth.get("pages", []):
        kind = page.get("document_type", "unknown")
        for name, field in page.get("fields", {}).items():
            if isinstance(field, dict) and field.get("labeled", True) is False:
                continue
            counts[f"{kind}.{name}"] += 1
    return dict(counts)


def processing_observations(result: dict | None) -> dict:
    """Saved behavior only; an enabled switch is not proof of a successful lookup."""
    pages = (result or {}).get("pages", [])
    comparisons = [p["neighbor_comparison"] for p in pages if isinstance(p.get("neighbor_comparison"), dict)]
    attempts = [a for comparison in comparisons for a in comparison.get("attempts", [])]
    registry_statuses, binding_statuses, printed_statuses = Counter(), Counter(), Counter()
    for page in pages:
        for role in (page.get("inn_verification", {}).get("roles") or {}).values():
            if isinstance(role, dict):
                registry_statuses[(role.get("registry") or {}).get("status", "unavailable")] += 1
                binding_statuses[(role.get("company_name_binding") or {}).get("status", "not_recorded")] += 1
                printed_statuses[role.get("printed_status", "not_recorded")] += 1
    return {
        "quality_gate_leaf_statuses": dict(Counter((p.get("quality_gate") or {}).get("status", "not_recorded") for p in pages)),
        "same_leaf_comparison_statuses": dict(Counter(c.get("status", "not_recorded") for c in comparisons)),
        "same_leaf_comparison_attempts": len(attempts),
        "same_leaf_attempt_statuses": dict(Counter(a.get("status", "not_recorded") for a in attempts)),
        "fusion_recovered_field_observations": sum(len((a.get("fusion") or {}).get("recovered_fields", [])) for a in attempts),
        "fusion_agreement_field_observations": sum(len((a.get("fusion") or {}).get("agreement_fields", [])) for a in attempts),
        "fusion_conflict_field_observations": sum(len((a.get("fusion") or {}).get("conflicting_fields", [])) for a in attempts),
        "registry_role_statuses": dict(registry_statuses),
        "inn_company_name_binding_statuses": dict(binding_statuses),
        "printed_inn_role_statuses": dict(printed_statuses),
        "registry_usage": (result or {}).get("registry_usage"),
    }


def load_run(root: Path, folder: Path, truth: dict, unit: str) -> dict:
    result, result_error = read_json(folder / "results.json")
    metrics, metrics_error = read_json(folder / "metrics.json")
    reasons = []
    if not truth.get("pages"):
        reasons.append("manual ground truth missing; not eligible for an accuracy aggregate")
    if result_error:
        reasons.append(f"results.json: {result_error}")
    if metrics_error:
        reasons.append(f"metrics.json: {metrics_error}")
    trace, malformed = trace_rows(folder / "trace.jsonl")
    starts = [i for i, row in enumerate(trace) if row.get("event") == "run_started"]
    completions = [i for i, row in enumerate(trace) if row.get("event") == "run_completed"]
    recovery_events = [(i, row) for i, row in enumerate(trace) if row.get("event") == "exports_recovered" and row.get("source_complete_proven") is True]
    recovery = recovery_events[-1][1] if recovery_events else None
    provenance_path = root / recovery["recovery_provenance_path"] if recovery and isinstance(recovery.get("recovery_provenance_path"), str) else None
    recovery_valid = bool(recovery and result and recovery.get("results_sha256") == file_hash(folder / "results.json")
                          and recovery.get("page_count") == len(result.get("pages", []))
                          and provenance_path and provenance_path.is_file()
                          and (not starts or recovery_events[-1][0] > starts[-1]))
    if unit == "video" and starts and (not completions or starts[-1] > completions[-1]) and not recovery_valid:
        reasons.append("latest run_started has no run_completed; prior artifacts are not final")
    if recovery and not recovery_valid:
        reasons.append("recovered export completion lacks matching result SHA/page count/provenance")
    if result:
        saved_at = (folder / "results.json").stat().st_mtime
        activity = [event_timestamp(row.get("time")) for row in trace if row.get("event") in {"model_request_started", "model_response_received", "model_response"}]
        if any(time is not None and time > saved_at for time in activity):
            reasons.append("provider activity is newer than results.json; latest processing has not saved final results")
    if result and metrics and (folder / "metrics.json").stat().st_mtime_ns < (folder / "results.json").stat().st_mtime_ns:
        reasons.append("metrics.json predates results.json; regenerate evaluation")
    if metrics and truth.get("pages") and (metrics.get("type_accuracy_all_expected") or {}).get("total") != len(truth["pages"]):
        reasons.append("metrics denominator differs from current manual leaf inventory")
    if metrics and truth.get("pages"):
        saved_denominators = {name: value.get("total", 0) for name, value in (metrics.get("field_accuracy") or {}).items() if value.get("total", 0)}
        if saved_denominators != expected_field_denominators(truth):
            reasons.append("field metrics denominators differ from selected ground truth; regenerate evaluation")
    if result and truth.get("source_sha256") and result.get("source", {}).get("sha256") != truth["source_sha256"]:
        reasons.append("source hash differs from ground truth")
    if malformed:
        reasons.append(f"trace contains {malformed} incomplete/unparsed lines")
    artifacts = {}
    for name in ("report.html", "results.json", "metrics.json", "fields.csv", "trace.jsonl", "classifications.json", "partial-results.json"):
        path = folder / name
        artifacts[name] = {"path": relative(path, root), "exists": path.is_file()}
    pages = (result or {}).get("pages", [])
    types = Counter(page.get("result", {}).get("document_type", "unknown") for page in pages)
    statuses = Counter(field.get("status", "unknown") for page in pages for field in page.get("result", {}).get("fields", {}).values())
    return {
        "run_id": folder.name, "unit": unit,
        "source": (result or {}).get("source") or {"path": truth.get("source"), "sha256": truth.get("source_sha256")},
        "partition": truth.get("partition", "reference"),
        "status": "pending" if reasons else "complete_provisional",
        "pending_reasons": reasons, "artifacts": artifacts,
        "missing_delivery_artifacts": [name for name in ("report.html", "results.json", "metrics.json", "fields.csv", "trace.jsonl") if not artifacts[name]["exists"]],
        "truth": truth_counts(truth), "output_leaf_count": len(pages) if result else None,
        "predicted_types": dict(sorted(types.items())) if result else None,
        "field_status_counts": dict(sorted(statuses.items())) if result else None,
        "leaves_with_result_errors": sum(bool(page.get("result", {}).get("errors")) for page in pages) if result else None,
        "configuration": (result or {}).get("configuration"),
        "saved_processing_observations": processing_observations(result),
        "elapsed_seconds": (result or {}).get("elapsed_seconds"),
        "execution": execution_snapshot(trace, recovery_valid=recovery_valid),
        "recovery": (result or {}).get("recovery"),
        "final_usage": usage_snapshot((result or {}).get("usage")),
        "metrics": metric_snapshot(metrics) if metrics and not reasons else None,
        "labeled_field_accuracy": add_ratio(list((metrics.get("field_accuracy") or {}).values())) if metrics and not reasons else None,
        "input_hashes": {name: file_hash(folder / name) for name in ("results.json", "metrics.json")},
        "limitations": (result or {}).get("limitations", []),
    }


def add_ratio(items: list[dict]) -> dict:
    correct = sum(item.get("correct", 0) for item in items)
    total = sum(item.get("total", 0) for item in items)
    return {"correct": correct, "total": total, "rate": correct / total if total else None,
            "contributing_runs": len(items)}


def aggregate(entries: list[dict]) -> dict:
    completed = [entry for entry in entries if entry["metrics"] is not None]
    result = {"included_runs": [e["run_id"] for e in completed],
              "excluded_runs": [e["run_id"] for e in entries if e["metrics"] is None],
              "completed_runs": len(completed), "expected_runs": len(entries),
              "expected_truth_leaf_intervals": sum(e["truth"]["leaf_intervals"] for e in entries),
              "completed_truth_leaf_intervals": sum(e["truth"]["leaf_intervals"] for e in completed),
              "metrics": {}, "field_accuracy": {},
              "scope": "Weighted counts from complete saved results+metrics pairs only; pending runs contribute no measured outcome."}
    for key in METRICS:
        items = [e["metrics"][key] for e in completed if isinstance(e["metrics"].get(key), dict)]
        result["metrics"][key] = add_ratio(items)
        if key in {"frame_document_detection_rate", "frame_document_detection_rate_targets"}:
            metric = result["metrics"][key]
            population = sum(item.get("population_expected_leaves", 0) for item in items)
            metric["whole_population_rate"] = metric["rate"] if population and metric["total"] == population else None
            metric["reviewed_subset_coverage"] = metric["total"] / population if population else None
            metric["coverage_denominator"] = population
            if key == "frame_document_detection_rate":
                metric["uncertain"] = sum(item.get("uncertain", 0) for item in items)
                metric["confirmed_unusable"] = sum(item.get("confirmed_unusable", 0) for item in items)
    names = sorted({name for e in completed for name in (e["metrics"].get("field_accuracy") or {})})
    for name in names:
        result["field_accuracy"][name] = add_ratio([e["metrics"]["field_accuracy"][name] for e in completed if name in (e["metrics"].get("field_accuracy") or {})])
    result["labeled_field_accuracy"] = add_ratio(list(result["field_accuracy"].values()))
    result["annotation_coverage"] = {key: sum((e["metrics"].get("annotation_coverage") or {}).get(key, 0) for e in completed)
                                     for key in ("fully_labeled_targets", "target_total", "unlabeled_targets")}
    result["unmatched_predictions"] = sum(e["metrics"].get("unmatched_predictions", 0) or 0 for e in completed)
    return result


def comparison_snapshot(root: Path, path: Path, truth_path: Path, videos: list[dict]) -> dict:
    """Expose saved before/after counts only while all input hashes still match."""
    value, error = read_json(path)
    snapshot = {"path": relative(path, root), "exists": path.is_file(), "sha256": file_hash(path),
                "status": "unavailable" if error == "missing" else "pending", "pending_reasons": [], "metrics": None}
    if error:
        snapshot["pending_reasons"] = [f"baseline-comparison.json: {error}"]
        return snapshot
    reasons = []
    if value.get("truth_sha256") != file_hash(truth_path):
        reasons.append("comparison ground truth SHA differs from selected reference")
    expected_fields = sum(e["truth"]["labeled_fields"] for e in videos)
    expected_full = sum(e["truth"]["fully_labeled_target_intervals"] for e in videos)
    for version in ("baseline", "final"):
        counts = value.get(version) or {}
        if (counts.get("labeled_fields") or {}).get("total") != expected_fields:
            reasons.append(f"{version} labelled-field denominator differs from selected reference")
        if (counts.get("e2e_on_fully_labeled") or {}).get("total") != expected_full:
            reasons.append(f"{version} fully-labelled leaf denominator differs from selected reference")
    by_source = value.get("by_source") or []
    expected_sources = {e["source"].get("sha256"): e for e in videos}
    if len(by_source) != len(videos) or set((r.get("source") or {}).get("sha256") for r in by_source) != set(expected_sources):
        reasons.append("comparison source inventory differs from selected video runs")
    for row in by_source:
        entry = expected_sources.get((row.get("source") or {}).get("sha256"))
        for version in ("baseline", "final"):
            recorded = (row.get("result_inputs") or {}).get(version) or {}
            saved_path = recorded.get("path")
            input_path = root / saved_path if isinstance(saved_path, str) else None
            if input_path is None or not recorded.get("sha256") or file_hash(input_path) != recorded["sha256"]:
                reasons.append(f"{version} results SHA is missing or stale")
            if version == "final" and entry and input_path and input_path.resolve() != (root / entry["artifacts"]["results.json"]["path"]).resolve():
                reasons.append("comparison final result path differs from selected run directory")
    if any(e["status"] != "complete_provisional" for e in videos):
        reasons.append("current video runs are pending; before/after is not final")
    else:
        current = aggregate(videos)
        for name in ("candidate_coverage", "type_accuracy_targets", "unknown_rejection", "e2e_on_fully_labeled", "labeled_fields"):
            expected = current["labeled_field_accuracy"] if name == "labeled_fields" else current["metrics"][name]
            saved = (value.get("final") or {}).get(name) or {}
            if any(saved.get(k) != expected.get(k) for k in ("correct", "total")):
                reasons.append(f"comparison final {name} differs from current saved evaluation")
    snapshot.update(status="pending" if reasons else "verified_provisional",
                    pending_reasons=sorted(set(reasons)), expected_labeled_fields=expected_fields,
                    expected_fully_labeled_targets=expected_full)
    if not reasons:
        snapshot["metrics"] = {k: value.get(k) for k in ("baseline", "final", "multiview", "quality_dispositions", "inn_registry_statuses", "comparison_rule", "by_source")}
    return snapshot


def update_catalog(root: Path, gt: dict, generated_at: str, *, runs_dir: Path | None = None, truth_path: Path | None = None) -> dict:
    runs_dir = runs_dir or root / "runs/mvp"
    truth_path = truth_path or root / "analysis/provisional-ground-truth.json"
    old, _ = read_json(root / "analysis/catalog.json")
    existing = {s.get("sample_id"): s for s in (old or {}).get("samples", [])}
    samples = []
    for video in sorted(gt.get("videos", []), key=lambda v: v.get("source_id", "")):
        sample_id = video["source_id"]
        sample = dict(existing.get(sample_id, {}))
        metadata, _ = read_json(root / "analysis" / sample_id / "metadata.json")
        sampling, _ = read_json(root / "analysis" / sample_id / "sampling.json")
        stream = next((s for s in (metadata or {}).get("streams", []) if s.get("codec_type") == "video"), {})
        format_info = (metadata or {}).get("format", {})
        rotation = next((s.get("rotation") for s in stream.get("side_data_list", []) if "rotation" in s), sample.get("display_rotation_degrees"))
        width, height = stream.get("width"), stream.get("height")
        run, _ = read_json(runs_dir / Path(video["source"]).stem / "results.json")
        source_hash = video.get("source_sha256") or sample.get("source_sha256") or (run or {}).get("source", {}).get("sha256")
        sample.update({"sample_id": sample_id, "source_path": video["source"], "source_sha256": source_hash,
                       "partition": video.get("partition"), "status": "provisional",
                       "truth": truth_counts(video | {"field_contract": gt.get("field_contract", {})}),
                       "ground_truth": relative(truth_path, root),
                       "sampling_manifest": str(root / "analysis" / sample_id / "sampling.json"),
                       "manual_observations": str(root / "analysis" / sample_id / "manual-observations.json"),
                       "mvp_run": relative(runs_dir / Path(video['source']).stem, root),
                       "saved_run_configuration": (run or {}).get("configuration"),
                       "saved_processing_observations": processing_observations(run),
                       "visually_reviewed_page_intervals": len(video.get("pages", [])),
                       "sampled_frame_count": len((sampling or {}).get("samples", [])) or video.get("sample_count"),
                       "native_ocr_batch": None})
        if stream:
            sample.update({"source_bytes": int(format_info.get("size", 0)),
                           "video_duration_s": float(stream.get("duration", video.get("video_duration_s", 0))),
                           "video_frame_count": int(stream.get("nb_frames", 0)),
                           "encoded_dimensions": [width, height],
                           "upright_frame_dimensions": [height, width] if rotation is not None and abs(rotation) % 180 == 90 else [width, height],
                           "codec": stream.get("codec_name"), "display_rotation_degrees": rotation,
                           "capture_metadata": format_info.get("tags", {})})
        samples.append(sample)
    return {
        "schema_version": 2, "review_date": generated_at[:10], "generated_at": generated_at,
        "status": "provisional", "agreed_document_types": [t.upper() for t in TARGETS],
        "scope_mapping_status": "The original four document types and 19 role-specific fields are implemented; video evidence covers UPD/TTN, and four rendered original PDF pages exercise all types separately.",
        "field_contract": gt.get("field_contract", {}),
        "automated_operations": [
            "Video decode with source PTS/display rotation and automatic candidate leaf segmentation",
            "Within-leaf quality rejection/ranking, initial retained views and conditional adaptive same-leaf frame comparison/fusion",
            "Local native OCR with content-orientation selection, pixel crops and coordinate transforms",
            "Batched ML classification of every candidate leaf; UPD/TORG12/TTN/TRN/unknown literal scope",
            "Structured role-focused field extraction from selected image pixels, evidence validation and normalization",
            "Own-role literal INN/checksum reading and optional bounded official FNS identity comparison; unavailable results remain explicit",
            "JSON/CSV/HTML output, CLI and local API, response caches, traces and shared request/token budgets",
            "Evaluation against provisional independent manual ground truth; separate field, type, coverage and E2E denominators",
        ],
        "manual_operations": [
            "Independent visible-leaf interval inventory and type labels, including repeated showings",
            f"Independent video fields: {sum(truth_counts(v | {'field_contract': gt.get('field_contract', {})})['labeled_fields'] for v in gt.get('videos', []))} labelled instances; {sum(truth_counts(v | {'field_contract': gt.get('field_contract', {})})['fully_labeled_target_intervals'] for v in gt.get('videos', []))} fully labelled target intervals; PDF labels separate",
            "Independent heading/field-legibility review of all current selected primary video frames, with source hashes and preserved interval boundaries",
        ],
        "customer_approved_ground_truth": False, "contractual_accuracy_metrics": None,
        "ground_truth_provenance": gt.get("provenance"),
        "ground_truth_sha256": file_hash(truth_path),
        "confirmed_literal_target_classes_in_provided_clips": gt.get("observed_target_types", []),
        "target_classes_not_observed_in_videos": gt.get("missing_target_types", []),
        "samples": samples,
        "pdf_reference": {"ground_truth": "analysis/reference-pdf/ground-truth.json", "ground_truth_sha256": file_hash(root / "analysis/reference-pdf/ground-truth.json"),
                          "run": "runs/reference-final", "unit": "original rendered PDF page; not video or document detection", "customer_approved": False},
        "validation_summary": relative(runs_dir / "validation-summary.json", root),
        "implementation_tools": {"pipeline": "sever/pipeline.py", "local_ocr": "sever/native/vision_ocr.swift",
                                 "cli": "sever/cli.py", "api": "sever/api.py", "delivery_builder": "tools/build_delivery.py"},
        "limitations": [
            "Manual labels and interval boundaries are provisional and not customer-approved.",
            "Quality indicators and model confidence are not calibrated accuracy probabilities.",
            "All current primary frames have independent provisional reviews; interval availability misses and uncertain judgments remain explicit. Customer-approved contractual legibility is pending.",
            "Same-leaf comparison and registry modes are recorded per run; an enabled switch does not prove field improvement or an available official registry match.",
            "Projective rectification is not implemented; quality rejection/selection cannot reconstruct hidden text.",
            "Repeated leaf showings are evaluated as intervals; unique-document identity/grouping is not validated.",
        ],
    }


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def ratio(item: dict | None) -> str:
    if not item or not item.get("total"):
        return "не измерено (0 размеченных)"
    return f"{item['correct']}/{item['total']} · {item['correct'] / item['total']:.1%}"


def report_html(summary: dict, root: Path, *, report_dir: Path | None = None) -> str:
    report_dir = report_dir or root / summary.get("paths", {}).get("runs_dir", "runs/mvp")
    def link(path: str, title: str) -> str:
        return f'<a href="{quote(relative(root / path, report_dir), safe="/.")}">{esc(title)}</a>'
    def metric_table(metrics: dict, pdf: bool = False) -> str:
        rows = []
        for key in METRICS:
            if pdf and key in {"candidate_coverage", "frame_document_detection_rate", "frame_document_detection_rate_targets"}:
                continue
            rows.append(f"<tr><td>{esc(LABELS[key])}</td><td>{esc(ratio(metrics.get(key)))}</td></tr>")
        return "<table><thead><tr><th>Показатель</th><th>Верно / знаменатель</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    def run_rows(entries: list[dict]) -> str:
        rows = []
        for entry in entries:
            links = " · ".join(link(v["path"], k) for k, v in entry["artifacts"].items() if v["exists"])
            status = "готов, предварительная оценка" if entry["status"] == "complete_provisional" else "ожидает завершения"
            reasons = "<br>".join(esc(r) for r in entry["pending_reasons"])
            count = entry["output_leaf_count"]
            types = ", ".join(f"{k}: {v}" for k, v in (entry["predicted_types"] or {}).items())
            missing = ", ".join(entry["missing_delivery_artifacts"])
            rows.append(f"<tr><td>{esc(entry['run_id'])}<br><small>{esc(entry['partition'])}</small></td><td>{esc(status)}<br><small>{reasons}</small></td><td>{esc(count if count is not None else '—')} / {entry['truth']['leaf_intervals']}<br><small>{esc(types)}<br>Листов с result.errors: {esc(entry['leaves_with_result_errors'] if entry['leaves_with_result_errors'] is not None else '—')}</small></td><td>{links}<br><small>{'Ожидаемые файлы: ' + esc(missing) if missing else ''}</small></td></tr>")
        return "<table><thead><tr><th>Источник</th><th>Состояние</th><th>Кандидаты / ручные интервалы</th><th>Файлы</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    agg = summary["video_aggregate"]
    ledger = summary["historical_expense"]
    annotation = agg["annotation_coverage"]
    usability = agg["metrics"].get("frame_document_detection_rate") or {}
    coverage = agg["metrics"].get("candidate_coverage") or {}
    missing_intervals = coverage.get("total", 0) - coverage.get("correct", 0)
    fields = "".join(f"<tr><td>{esc(k)}</td><td>{esc(ratio(v))}</td></tr>" for k, v in agg["field_accuracy"].items())
    comparison = summary["baseline_comparison"]
    before_after = comparison.get("metrics")
    if before_after:
        comparison_rows = ''.join(f"<tr><td>{esc(LABELS.get(k, 'Размеченные поля'))}</td><td>{esc(ratio(before_after['baseline'].get(k)))}</td><td>{esc(ratio(before_after['final'].get(k)))}</td></tr>" for k in ("candidate_coverage", "type_accuracy_targets", "unknown_rejection", "labeled_fields", "e2e_on_fully_labeled"))
        comparison_html = f"<p>Обе версии проверены по одним {comparison['expected_labeled_fields']} размеченным полям и {comparison['expected_fully_labeled_targets']} полностью размеченным листам. Проверены SHA эталона, исходных results.json и совпадение итоговых числителей со сводкой.</p><table><thead><tr><th>Показатель</th><th>Исходная версия</th><th>Закрывающая версия</th></tr></thead><tbody>{comparison_rows}</tbody></table>"
    else:
        comparison_html = '<p>Сравнение до/после ожидает действующих сохранённых данных: ' + esc('; '.join(comparison['pending_reasons'])) + '.</p>'
    if comparison['exists']:
        comparison_html += '<p>' + link(comparison['path'], 'Сохранённое сравнение и SHA входов') + '.</p>'
    execution_rows = []
    for entry in summary['videos']:
        execution = entry['execution']
        recovery = entry.get('recovery') or execution.get('recovery')
        recovery_link = (link(relative(Path(recovery['recovery_provenance_path']), root), 'Происхождение восстановления')
                         if recovery and isinstance(recovery.get('recovery_provenance_path'), str) else '')
        recovery_note = (f"<br>Экспорт восстановлен: {esc(recovery.get('prefix_page_count'))} листов из checkpoint и {esc(recovery.get('suffix_page_count'))} из проверенного архива. "
                         f"Исходный полный прогон: {esc(recovery.get('parent_elapsed_seconds'))} с; записанный повтор: {esc(recovery.get('replay_duration_seconds'))} с; экспорт: {esc(recovery.get('recovery_elapsed_seconds'))} с. "
                         + recovery_link) if recovery else ''
        execution_rows.append(f"<tr><td>{esc(entry['run_id'])}</td><td>{esc(execution['status'])}: {esc(execution['mode'])}<br>{esc(execution.get('cache_events', {}))}{recovery_note}</td><td>{esc(execution.get('recorded_elapsed_seconds'))}</td><td>{esc(execution.get('attempted_requests'))} / {esc(execution.get('recorded_responses'))}</td><td>{esc(execution.get('actual_cost_rub'))} / {esc(execution.get('estimated_cost_rub'))}<br>Без ответа: {esc(execution.get('attempts_without_recorded_response'))}; стоимость ответа неизвестна: {esc(execution.get('unknown_billing_responses'))}</td></tr>")
    execution_table = '<table><thead><tr><th>Видео</th><th>Trace: режим и cache events</th><th>Сохранённое время, с</th><th>Новые попытки / ответы</th><th>Новый расход провайдера / оценка, ₽</th></tr></thead><tbody>' + ''.join(execution_rows) + '</tbody></table>'
    behavior = []
    for entry in summary["videos"]:
        if entry["status"] != "complete_provisional":
            continue
        config = entry.get("configuration") or {}
        obs = entry.get("saved_processing_observations") or {}
        limit = config.get("per_leaf_frame_limit", "not_recorded")
        frames = "без лимита числа кадров одной стороны" if "per_leaf_frame_limit" in config and limit is None else f"лимит: {limit}"
        behavior.append(f"<tr><td>{esc(entry['run_id'])}</td><td>neighbor_fallback={esc(config.get('neighbor_fallback', 'not_recorded'))}; {esc(frames)}<br>verify_inn={esc(config.get('verify_inn', 'not_recorded'))}</td><td>{esc(obs.get('quality_gate_leaf_statuses', {}))}</td><td>Попыток: {esc(obs.get('same_leaf_comparison_attempts', 0))}; статусы: {esc(obs.get('same_leaf_attempt_statuses', {}))}<br>Наблюдений восстановленных полей: {esc(obs.get('fusion_recovered_field_observations', 0))}; согласований: {esc(obs.get('fusion_agreement_field_observations', 0))}; конфликтов: {esc(obs.get('fusion_conflict_field_observations', 0))}</td><td>Реестр: {esc(obs.get('registry_role_statuses', {}))}<br>Связь ИНН/названия: {esc(obs.get('inn_company_name_binding_statuses', {}))}<br>Напечатанный ИНН: {esc(obs.get('printed_inn_role_statuses', {}))}</td></tr>")
    behavior_table = '<table><thead><tr><th>Видео</th><th>Сохранённая конфигурация</th><th>Отбор/отбраковка</th><th>Дополнительные кадры этой стороны</th><th>ФНС: статусы ролей</th></tr></thead><tbody>' + ''.join(behavior) + '</tbody></table>' if behavior else '<p>Сохранённые завершённые видео-прогоны с подтверждением этих режимов пока отсутствуют.</p>'
    failures = Counter()
    for entry in summary["videos"]:
        if entry["metrics"]:
            failures.update(entry["metrics"].get("failure_reasons") or {})
    failure_rows = ''.join(f'<tr><td>{esc(name)}</td><td>{count}</td></tr>' for name, count in failures.most_common())
    failure_table = '<table><thead><tr><th>Причина из сохранённой оценки видео</th><th>Количество</th></tr></thead><tbody>' + failure_rows + '</tbody></table>' if failure_rows else '<p>В завершённых сохранённых оценках пока нет записанных причин ошибок.</p>'
    costs = []
    for kind, entries in (("Видео", summary["videos"]), ("PDF", [summary["pdf_reference"]])):
        for entry in entries:
            usage = entry.get("final_usage")
            if usage:
                cost = usage.get("actual_cost_rub")
                costs.append(f"<tr><td>{kind}: {esc(entry['run_id'])}</td><td>{esc(entry['status'])}</td><td>{usage.get('attempted_requests')} / {usage.get('requests')}</td><td>{usage.get('input_tokens')} / {usage.get('output_tokens')}</td><td>{esc(f'{cost:.6f}' if finite_number(cost) else 'неизвестно')}</td><td>{esc(usage.get('estimated_cost_rub'))}</td></tr>")
    final_total = summary["final_run_usage"]["complete_runs"]
    reference = summary["pdf_reference"]
    pdf_metrics = metric_table(reference["metrics"], pdf=True) if reference["metrics"] else "<p>Итоговая PDF-оценка ожидается; прошлые диагностические прогоны сюда не подставляются.</p>"
    if reference["metrics"]:
        pdf_fields = "".join(f"<tr><td>{esc(k)}</td><td>{esc(ratio(v))}</td></tr>" for k, v in (reference["metrics"].get("field_accuracy") or {}).items())
        pdf_metrics += f'<p>Общая точность размеченных полей PDF: {esc(ratio(reference["labeled_field_accuracy"]))}.</p>'
        pdf_metrics += '<table><thead><tr><th>PDF: тип / поле</th><th>Верно / размечено</th></tr></thead><tbody>' + pdf_fields + '</tbody></table>'
    demonstration = summary.get("demonstration")
    demo_html = ""
    if demonstration:
        video_url = quote(relative(root / demonstration["video_path"], report_dir), safe="/.")
        demo_html = f'<h2>Аннотированный фрагмент</h2><video controls preload="metadata" style="width:100%;max-width:960px" src="{video_url}"></video><p>Реквизиты из сохранённого результата распознавания; рамки привязаны к исходному кадру. Фрагмент увеличен и замедлен ×4. {link(demonstration["provenance_path"], "Происхождение значений и рамок")}.</p>'
    return f'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Север ТСЛ — сводка сохранённых проверок</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#172033;background:#fafbfd}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%;margin:16px 0;background:white}}th,td{{border:1px solid #dce2ea;padding:10px;text-align:left;vertical-align:top}}th{{background:#edf1f7}}small{{color:#586476}}a{{color:#165ea0}}.notice{{background:#fff3cf;padding:16px;border-left:4px solid #c28b10}}code{{background:#e9edf3;padding:2px 5px}}footer{{margin:32px 0;color:#586476}}</style>
<h1>Север ТСЛ: сохранённые проверки MVP</h1>
<p class="notice">Снимок {esc(summary['generated_at'])}. Завершено видео {agg['completed_runs']}/{agg['expected_runs']}; статус всей поставки: {esc(summary['status'])}. Разметка предварительная, заказчиком не утверждена. Это проверка прототипа на данных источниках, а не подтверждение договорной или промышленной точности.</p>
<p>Оценка видео считает показы листов, включая повторные показы. Покрытие интервала выбранным кадром не подтверждает читаемость полей. PDF считается отдельно как четыре оригинальные страницы. Сводка только читает сохранённые файлы и не вызывает модели.</p>
{demo_html}
<h2>Видео и состояние файлов</h2>{run_rows(summary['videos'])}
<p>В суммарную оценку включены {agg['completed_truth_leaf_intervals']}/{agg['expected_truth_leaf_intervals']} ручных интервалов из завершённых прогонов. Проценты получены сложением числителей и знаменателей.</p>
{metric_table(agg['metrics'])}
<p>Пригодность связывается с независимо просмотренным конкретным кадром. В знаменателе оценки {usability.get('total', 0)} из {usability.get('coverage_denominator', 0)} ручных интервалов; пропущено {missing_intervals}, неопределённых оценок {usability.get('uncertain', 0)}. Подтверждённая пригодность {esc(ratio(usability))} — предварительная нижняя граница. Для целевых форм отдельно проверяется читаемость обязательных полей.</p>
<p>Полностью размечены {annotation['fully_labeled_targets']}/{annotation['target_total']} целевых интервалов в включённых прогонах; {annotation['unlabeled_targets']} без полной разметки. Их нельзя трактовать как измеренные ошибки извлечения. Несопоставленных кандидатов: {agg['unmatched_predictions']}.</p>
<h2>Поля видео</h2><p>Общая точность размеченных полей: {esc(ratio(agg['labeled_field_accuracy']))}. Каждый знаменатель включает только ручные метки данного поля.</p>
<table><thead><tr><th>Тип / поле</th><th>Верно / размечено</th></tr></thead><tbody>{fields or '<tr><td colspan="2">Пока нет завершённой оценки размеченных полей.</td></tr>'}</tbody></table>
<h2>До и после на одном эталоне</h2>{comparison_html}
<h2>Отбраковка, дополнительные кадры и проверка ИНН</h2>{behavior_table}
<p>Количество восстановлений/согласований — наблюдения работы алгоритма, а не дополнительная метрика точности. Дополнительные кадры читаются только после проверки того же листа; конфликт сохраняется явно. Включённая проверка ИНН сама по себе не доказывает доступный ответ реестра; статус unavailable не означает подтверждение организации. Справочник не заменяет напечатанный адрес доставки.</p>
<h2>Типовые ошибки и их сохранённые причины</h2>{failure_table}
<h2>Оригинальный PDF: отдельная проверка</h2>{run_rows([reference])}{pdf_metrics}
<p>PDF не измеряет смену страниц, выбор кадров или пригодность видеосъёмки. Синтетические интервалы PDF служат индексами страниц.</p>
<h2>Расход</h2>
<p>Последняя отмеченная trace-обработка каждого видео отделена от истории. cache/replay означает повторную проверку сохранённых ответов; новый расход отражает только новые записанные ответы. Отсутствие model-cache events не доказывает холодный запуск: могли использоваться кадры или OCR. Контролируемое время с пустыми кэшами не измерялось.</p>{execution_table}
<p>Ниже — расход, записанный в <code>results.usage</code> каждого сохранённого итогового файла. Строки pending могут относиться к прежнему прогону; в сумму завершённых входят только готовые пары результатов и метрик. Накопительный <code>batch_usage</code> исключён.</p>
<table><thead><tr><th>Прогон</th><th>Статус</th><th>Попытки / вызовы по usage</th><th>Токены вход / выход</th><th>Провайдер, ₽</th><th>Оценка, ₽</th></tr></thead><tbody>{''.join(costs)}</tbody></table>
<p>Завершённые итоговые прогоны: {final_total['actual_cost_rub']:.6f} ₽ по провайдеру, {final_total['estimated_cost_rub']:.6f} ₽ оценочно; попыток {final_total['attempted_requests']}, вызовов по сохранённому usage {final_total['requests']}. Стоимость {final_total['responses_without_reported_provider_cost']} вызовов не подтверждена сохранённым ответом провайдера.</p>
<p>История всех записанных ответов, включая разработку и ошибки: {ledger['provider_reported_cost_rub']:.6f} ₽ по провайдеру и {ledger['catalogue_estimated_cost_rub']:.6f} ₽ оценочно, {ledger['response_count']} ответов; {ledger['failed_responses']} завершились ошибкой обработки, {ledger['unknown_billing_responses']} с неизвестной стоимостью. Ещё {ledger['requests_without_response']} попыток без записи ответа, их стоимость неизвестна. {link('runs/expense-ledger.json', 'Реестр расходов')}.</p>
<p>Итоговый расход пересекается с историческим: эти суммы не складываются. Реестр восстанавливает записанные trace-события и не является сверкой баланса аккаунта; незаписанные вызовы и корректировки провайдера не измерены.</p>
<h2>Пределы проверки</h2><ul>{''.join('<li>' + esc(s) + '</li>' for s in summary['limitations'])}</ul>
<h2>Рекомендации по съёмке</h2><ul>{''.join('<li>' + esc(s) + '</li>' for s in summary['filming_recommendations'])}</ul>
<h2>Применимость прототипа и следующий этап</h2><p>{esc(summary['applicability'])}</p><ul>{''.join('<li>' + esc(s) + '</li>' for s in summary['next_stage'])}</ul>
<p>{link(summary['paths']['summary_path'],'Машиночитаемая сводка')} · {link('analysis/catalog.json','Каталог четырёх источников')} · {link(summary['ground_truth']['video_path'],'Предварительная ручная разметка видео')} · {link('analysis/reference-pdf/ground-truth.json','Ручная разметка оригинального PDF')} · {link('analysis/ground-truth-expanded-review-portable.html','Переносимая таблица проверки с исходными изображениями')} · {link('analysis/ground-truth-expanded-review.html','Локальная таблица проверки')} · {link('analysis/ground-truth-expanded-review.csv','CSV для исправлений')} · {link('analysis/ground-truth-role-address-audit.json','Независимый аудит адресных ролей')}</p>
<footer>Обновление после завершения прогонов: <code>{esc(summary['rebuild_command'])}</code>. HTML содержит только локальные ссылки, без скриптов, форм и сетевых запросов.</footer></html>'''


def build(root: Path, *, runs_dir: Path | None = None, truth_path: Path | None = None) -> dict:
    root = root.resolve()
    def workspace_path(value: Path | None, default: str) -> Path:
        path = Path(value) if value is not None else Path(default)
        return (path if path.is_absolute() else root / path).resolve()
    runs_dir = workspace_path(runs_dir, "runs/mvp")
    truth_path = workspace_path(truth_path, "analysis/provisional-ground-truth.json")
    generated_at = datetime.now(timezone.utc).isoformat()
    gt, error = read_json(truth_path)
    if not gt:
        raise RuntimeError(f"Cannot read manual video ground truth: {error}")
    videos = []
    for truth in sorted(gt.get("videos", []), key=lambda v: v.get("source_id", "")):
        truth = truth | {"field_contract": gt.get("field_contract", {})}
        videos.append(load_run(root, runs_dir / Path(truth["source"]).stem, truth, "video"))
    expected_folders = {e["run_id"] for e in videos}
    for path in sorted(runs_dir.glob("*/results.json")):
        if path.parent.name not in expected_folders:
            videos.append(load_run(root, path.parent, {}, "video"))
    pdf_truth, _ = read_json(root / "analysis/reference-pdf/ground-truth.json")
    pdf = load_run(root, root / "runs/reference-final", pdf_truth or {}, "pdf_reference_page")
    ledger = expense_ledger(root, generated_at)
    comparison = comparison_snapshot(root, runs_dir / "baseline-comparison.json", truth_path, videos)
    complete = [e for e in videos + [pdf] if e["status"] == "complete_provisional"]
    demo_video, demo_provenance = runs_dir / "annotated-demo.mp4", runs_dir / "annotated-demo.provenance.json"
    demo, _ = read_json(demo_provenance)
    demonstration = None
    if (demo_video.is_file() and demo and demo.get("manual_ground_truth_read") is False
            and demo.get("all_recognized_fields_grounded_on_reference") is True
            and demo.get("results_sha256") and demo.get("fields")
            and demo.get("results_sha256") == file_hash(Path(demo.get("annotation_values_source", "")))):
        demonstration = {"video_path": relative(demo_video, root), "video_sha256": file_hash(demo_video),
                         "provenance_path": relative(demo_provenance, root), "provenance_sha256": file_hash(demo_provenance),
                         "results_sha256": demo["results_sha256"], "field_count": len(demo.get("fields", {}))}
    summary = {
        "schema_version": 2, "generated_at": generated_at, "status": "provisional_pending" if len(complete) != len(videos) + 1 or any(e["missing_delivery_artifacts"] for e in videos + [pdf]) or comparison["exists"] and comparison["metrics"] is None else "complete_provisional",
        "customer_approved": False, "contractual_acceptance": False,
        "demonstration": demonstration,
        "paths": {"runs_dir": relative(runs_dir, root), "summary_path": relative(runs_dir / "validation-summary.json", root),
                  "report_path": relative(runs_dir / "report.html", root)},
        "rebuild_command": f".venv/bin/python tools/build_delivery.py --runs-dir {relative(runs_dir, root)} --truth {relative(truth_path, root)}",
        "ground_truth": {"video_path": relative(truth_path, root), "video_sha256": file_hash(truth_path),
                         "pdf_path": "analysis/reference-pdf/ground-truth.json", "pdf_sha256": file_hash(root / "analysis/reference-pdf/ground-truth.json"),
                         "provenance": gt.get("provenance"), "customer_approved": False},
        "truth_annotation_counts": {key: sum(e["truth"][key] for e in videos) for key in ("leaf_intervals", "target_intervals", "fully_labeled_target_intervals", "labeled_fields")},
        "videos": videos, "pdf_reference": pdf, "video_aggregate": aggregate(videos),
        "baseline_comparison": comparison,
        "review_artifacts": {"portable_html": {"path": "analysis/ground-truth-expanded-review-portable.html", "sha256": file_hash(root / "analysis/ground-truth-expanded-review-portable.html")},
                             "csv": {"path": "analysis/ground-truth-expanded-review.csv", "sha256": file_hash(root / "analysis/ground-truth-expanded-review.csv")},
                             "role_address_audit": {"path": "analysis/ground-truth-role-address-audit.json", "sha256": file_hash(root / "analysis/ground-truth-role-address-audit.json")}},
        "final_run_usage": {"complete_runs": total_usage(complete), "all_available_result_files": total_usage(videos + [pdf]),
                            "warning": "Pending rows may contain previous results. Final usage and the historical trace ledger overlap; do not add."},
        "historical_expense": {k: v for k, v in ledger.items() if k not in {"responses", "requests"}} | {"path": "runs/expense-ledger.json"},
        "limitations": [
            "Данные и ручная разметка предварительные; приёмка заказчиком не проведена.",
            "В видео вручную подтверждены УПД и ТТН; ТОРГ-12 и ТрН проверяются только на отдельных оригинальных PDF-страницах.",
            "Повторные показы считаются интервалами листов; число уникальных документов и их группировка не проверены.",
            f"В выбранном эталоне полностью размечены {sum(e['truth']['fully_labeled_target_intervals'] for e in videos)} из {sum(e['truth']['target_intervals'] for e in videos)} целевых видео-интервалов, всего {sum(e['truth']['labeled_fields'] for e in videos)} полевых меток. Неразмеченные поля исключены из знаменателей точности значений.",
            "Пригодность оценивается по сохранённому независимому просмотру основных кадров; метрика учитывает привязку к неизменным интервалам, пропуски и неопределённые оценки. Приёмка заказчиком не проведена.",
            "Границы листов приблизительны; очень краткие или полностью закрытые показы могли отсутствовать в ручной выборке.",
            "Показатели резкости, движения, перспективы, бликов и перекрытия — эвристики; confidence модели не калиброван.",
            "Условное сравнение/объединение кадров одной стороны и проверка ИНН отражаются в сохранённой конфигурации и статусах данного прогона; улучшение или доступность реестра не предполагаются заранее.",
            "Отсутствие лимита числа кадров одной стороны не отменяет общий бюджет запросов/токенов, проверку тождества листа и исчерпание исходных кадров.",
            "Несколько юридических названий в строке своей роли создают неоднозначную связь ИНН/названия. Контрольная сумма ИНН не подтверждает регистрацию; справочник не заменяет грузовой адрес.",
            "Выборка содержит четыре фактических видео и четыре оригинальные PDF-страницы; результаты этих разных единиц не объединяются в общую точность.",
            "Закрывающая обработка может повторно использовать кэши. Сохранённое время выполнения и новый расход не являются измерением холодного запуска; история расходов показана отдельно.",
            "Исправление перспективы не реализовано; скрытый рукой или другим листом текст нельзя восстановить уверенным предположением.",
            "Нижняя граница E2E по всем целевым листам включает неразмеченные листы; она не заменяет точность на полностью размеченных данных.",
        ],
        "filming_recommendations": [
            "Остановить перелистывание и удерживать каждую сторону неподвижно несколько секунд; оставить читаемой собственную шапку с номером/датой и строками ролей.",
            "Снимать ближе к перпендикуляру; включать в кадр весь лист с небольшим запасом по краям, без перекрытия рукой, загнутой соседней страницей или другой накладной.",
            "Обеспечить равномерный рассеянный свет без бликов; дождаться фокусировки, чтобы мелкие адреса и продолжения строк оставались читаемыми.",
            "Лицевую сторону, оборот и продолжения показывать последовательно; заранее согласовать, объединяются ли они в один документ и как учитываются повторные показы.",
        ],
        "applicability": "Сводка строится только по завершённым сохранённым результатам и оценкам. Прототип применим как исследовательская обработка предоставленных читаемых листов с проверяемыми доказательствами и явными отказами/конфликтами. Подтверждение всей видеосъёмки для четырёх типов и промышленной точности требует отдельной согласованной выборки; положительные видео этой выборки содержат УПД и ТТН. Если есть pending, итог исследования по ним ещё не сформирован.",
        "next_stage": [
            "Сотруднику «Север ТСЛ» проверить и исправить предварительный эталон, согласовать единицу документа, роли на оборотах и семантику номера/даты на продолжениях УПД.",
            "Добавить реальные положительные видео ТОРГ-12 и ТрН, а также трудные отрицательные формы; отделить будущую контрольную выборку от разработки.",
            "По согласованным целям точности разобрать сохранённые ошибки, проверить улучшение от дополнительных кадров и определить допустимые отказы/ручную проверку.",
            "Решение об эксплуатации, интеграциях и дальнейшей доработке принимать отдельно по измеренным результатам и ограничениям исследования; текущая поставка не утверждает промышленный SLA.",
        ],
    }
    write_json(root / "runs/expense-ledger.json", ledger)
    write_json(root / "analysis/catalog.json", update_catalog(root, gt, generated_at, runs_dir=runs_dir, truth_path=truth_path))
    write_json(runs_dir / "validation-summary.json", summary)
    report = runs_dir / "report.html"
    report.write_text(report_html(summary, root, report_dir=runs_dir), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-dir", type=Path, help="Video runs and summary output directory, relative to --root unless absolute")
    parser.add_argument("--truth", type=Path, help="Selected manual video ground truth, relative to --root unless absolute")
    args = parser.parse_args()
    summary = build(args.root.resolve(), runs_dir=args.runs_dir, truth_path=args.truth)
    print(json.dumps({"status": summary["status"], "completed_video_runs": summary["video_aggregate"]["completed_runs"],
                      "expected_video_runs": summary["video_aggregate"]["expected_runs"], "pdf_status": summary["pdf_reference"]["status"],
                      "historical_responses": summary["historical_expense"]["response_count"],
                      "provider_reported_cost_rub": summary["historical_expense"]["provider_reported_cost_rub"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
