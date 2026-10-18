"""Score completed server results after inference against explicit private truth.

This process must run separately from the recognition API. It never modifies
recognition results, ground truth, cache contents, or the batch driver index.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from sever.evaluation import evaluate
from sever.schema import FIELDS_BY_TYPE


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric(correct, total):
    return {"correct": correct, "total": total, "rate": correct / total if total else None}


def aggregate(reports):
    """Pool numerators and denominators; never average per-video percentages."""
    names = ("candidate_coverage", "type_accuracy_all_expected", "type_accuracy_targets",
             "unknown_rejection", "e2e_on_fully_labeled", "e2e_all_targets_conservative")
    result = {name: metric(sum(report[name]["correct"] for report in reports),
                           sum(report[name]["total"] for report in reports)) for name in names}
    fields = defaultdict(lambda: {"correct": 0, "total": 0})
    failures = Counter()
    annotation = Counter()
    for report in reports:
        for name, values in report.get("field_accuracy", {}).items():
            fields[name]["correct"] += values["correct"]
            fields[name]["total"] += values["total"]
        failures.update(report.get("failure_reasons", {}))
        annotation.update(report.get("annotation_coverage", {}))
    result["field_accuracy"] = {name: {**values,
        "accuracy": values["correct"] / values["total"] if values["total"] else None}
        for name, values in sorted(fields.items())}
    result["labeled_fields"] = metric(sum(value["correct"] for value in fields.values()),
                                       sum(value["total"] for value in fields.values()))
    result["annotation_coverage"] = dict(annotation)
    result["failure_reasons"] = dict(failures)
    result["unmatched_predictions"] = sum(report.get("unmatched_predictions", 0) for report in reports)
    result["scored_video_count"] = len(reports)
    result["frame_document_detection_rate"] = {"correct": None, "total": 0, "rate": None,
        "status": "not_measured", "reason": "Historical frame usability is not a review of freshly selected Linux frames"}
    result["unit"] = "visible leaf interval, including repeat showings; unchanged one-to-one primary-frame timestamp matching"
    return result


def truth_counts(videos):
    counts = Counter(leaf_intervals=0, target_intervals=0, fully_labeled_target_intervals=0,
                     labeled_fields=0, required_target_fields=0)
    kinds = Counter()
    for video in videos:
        for page in video["pages"]:
            kind = page["document_type"]
            kinds[kind] += 1
            counts["leaf_intervals"] += 1
            if kind != "unknown":
                counts["target_intervals"] += 1
                counts["required_target_fields"] += len(FIELDS_BY_TYPE[kind])
                fields = page.get("fields", {})
                full = set(fields) == set(FIELDS_BY_TYPE[kind]) and all(
                    not isinstance(value, dict) or value.get("labeled", True) for value in fields.values())
                counts["fully_labeled_target_intervals"] += int(full)
            counts["labeled_fields"] += sum(not isinstance(value, dict) or value.get("labeled", True)
                                           for value in page.get("fields", {}).values())
    return {**dict(counts), "types": dict(kinds),
            "unlabeled_required_fields": counts["required_target_fields"] - counts["labeled_fields"]}


def validated_result(job, run_dir):
    """Read only this API job's result and verify source/artifact identity."""
    identifier = str(uuid.UUID(job["job_id"]))
    directory = (run_dir / identifier).resolve()
    if directory.parent != run_dir:
        raise ValueError("Job directory is outside the configured run directory")
    path = directory / "results.json"
    if not path.is_file() or path.resolve().parent != directory:
        raise ValueError("Completed job results are missing or outside their own directory")
    value = json.loads(path.read_text())
    if value.get("source", {}).get("sha256") != job["source"]["sha256"]:
        raise ValueError("Results source hash differs from batch input")
    sha = digest(path)
    recorded = job.get("proof", {}).get("artifacts", {}).get("results.json", {}).get("sha256")
    if recorded and recorded != sha:
        raise ValueError("Results file changed after the batch provenance snapshot")
    return value, path, sha


def run(args):
    manifest_path = args.manifest.resolve()
    truth_path = args.truth.resolve()
    output = args.out.resolve()
    if output in {manifest_path, truth_path}:
        raise ValueError("Evaluation output cannot overwrite an input")
    manifest = json.loads(manifest_path.read_text())
    run_dir = Path(manifest["settings"]["run_dir"]).resolve()
    if output.suffix != ".json":
        raise ValueError("Evaluation summary output must be a JSON file")
    try:
        first = output.relative_to(run_dir).parts[0]
        uuid.UUID(first)
    except (ValueError, IndexError):
        pass
    else:
        raise ValueError("Evaluation output cannot be inside an API job artifact directory")
    if any(job["status"] not in {"completed", "completed_with_errors", "failed"}
           for job in manifest.get("jobs", [])):
        raise ValueError("Recognition jobs are still active or uncertain; evaluate after inference finishes")
    truth = json.loads(truth_path.read_text())
    # Later labelling rounds are scored only when asked for; by default the
    # original four independently labelled videos keep their fixed denominators.
    rounds = set(getattr(args, "include_round", None) or [])
    videos = [video for video in truth["videos"] if video.get("labelling_round") is None or video.get("labelling_round") in rounds]
    by_hash = {}
    for video in videos:
        sha = video.get("source_sha256")
        if not sha or sha in by_hash:
            raise ValueError("Every truth video needs a unique full-source SHA-256")
        by_hash[sha] = video
    if sum(video.get("labelling_round") is None for video in videos) != 4:
        raise ValueError("This evaluation requires the four original independently labelled videos")
    jobs_by_hash = defaultdict(list)
    operational_jobs, unlabelled = [], []
    for job in manifest.get("jobs", []):
        sha = job["source"]["sha256"]
        jobs_by_hash[sha].append(job)
        operational = {"name": job["source"]["name"], "sha256": sha, "job_id": job.get("job_id"),
                       "status": job["status"], "api_error_type": job.get("api_state", {}).get("error_type"),
                       "pages_with_errors": job.get("summary", {}).get("pages_with_errors"),
                       "errors": job.get("summary", {}).get("errors", []),
                       "fresh_run_verified": job.get("proof", {}).get("fresh_run_verified"),
                       "verification_issues": job.get("proof", {}).get("verification_issues", []),
                       "usage": job.get("usage", {}), "has_independent_truth": sha in by_hash}
        operational_jobs.append(operational)
    for source in manifest["sources"]:
        if source["sha256"] not in by_hash:
            matching = jobs_by_hash.get(source["sha256"], [])
            unlabelled.append({"name": source["name"], "sha256": source["sha256"],
                              "accuracy_status": "not_measured", "reason": "No independent ground truth for this video",
                              "job_statuses": [job["status"] for job in matching] or ["not_submitted"],
                              "treated_as_recognition_error": False})
    measured, all_required, originals, missing = [], [], [], []
    metrics_dir = output.parent / (output.stem + "-original-metrics")
    selected_hashes = {source["sha256"] for source in manifest["sources"]}
    for sha, original in by_hash.items():
        expected = deepcopy(original)
        # The old judgement concerns specific macOS primary frames. It must not
        # become an asserted review of new Linux frames merely through equal IDs.
        expected.pop("selected_frame_usability", None)
        expected.update(customer_approved=truth.get("customer_approved", False), provenance=truth.get("provenance"))
        candidates = jobs_by_hash.get(sha, [])
        completed = [job for job in candidates if job["status"] in {"completed", "completed_with_errors"}]
        entry = {"original_name": Path(original["source"]).name, "source_sha256": sha,
                 "selected_in_server_batch": sha in selected_hashes,
                 "recorded_job_statuses": [job["status"] for job in candidates],
                 "customer_approved": bool(truth.get("customer_approved", False)),
                 "historical_frame_usability_reused": False}
        report = None
        if len(completed) == 1:
            job = completed[0]
            try:
                predicted, path, result_sha = validated_result(job, run_dir)
                report = evaluate(predicted, expected)
            except (ValueError, KeyError, OSError) as exc:
                entry.update(status="result_not_comparable", validation_error_type=type(exc).__name__)
            else:
                entry.update(status="measured_provisional", job_id=job["job_id"],
                             results_path=str(path), results_sha256=result_sha,
                             fresh_run_verified=job.get("proof", {}).get("fresh_run_verified"),
                             verification_issues=job.get("proof", {}).get("verification_issues", []))
        elif len(completed) > 1:
            entry.update(status="ambiguous_completed_jobs", validation_error_type="MultipleCompletedJobsForSameSource")
        else:
            entry["status"] = "not_selected" if sha not in selected_hashes else "not_processed_to_completion"
        absent = report is None
        if absent:
            missing.append({"original_name": entry["original_name"], "source_sha256": sha, "status": entry["status"]})
            # This is explicitly a missing-output projection, not a fabricated
            # recognition result. It preserves all original denominators.
            projected = evaluate({"source": {"sha256": sha}, "pages": []}, expected)
            entry["missing_output_projection"] = {"interpretation": "Unprocessed or incomparable original counted as a miss; no prediction was created", "metrics": projected}
            all_required.append(projected)
        else:
            entry["metrics"] = report
            measured.append(report)
            all_required.append(report)
        metrics_path = metrics_dir / (Path(entry["original_name"]).stem + ".json")
        save(metrics_path, entry)
        originals.append({**{key: value for key, value in entry.items() if key not in {"metrics", "missing_output_projection"}},
                          "metrics_path": str(metrics_path)})
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_provisional_original_subset" if not missing else "partial_provisional_original_subset",
        "customer_approved": bool(truth.get("customer_approved", False)), "contractual_acceptance": False,
        "inputs": {"manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
                   "ground_truth_path": str(truth_path), "ground_truth_sha256": digest(truth_path),
                   "ground_truth_loaded_only_after_inference": True},
        "ground_truth_counts": truth_counts(videos), "originals": originals,
        "missing_or_unprocessed_originals": missing,
        "accuracy": {"completed_comparable_originals": aggregate(measured),
                     "all_four_originals_unprocessed_as_misses": {**aggregate(all_required),
                         "interpretation": "Conservative full-original denominator; missing/incomparable outputs are explicitly projected as misses"}},
        "unlabelled_server_videos": unlabelled,
        "operational": {"batch_status": manifest["status"], "selected_videos": len(manifest["sources"]),
                        "submitted_jobs": len(manifest.get("jobs", [])),
                        "status_counts": dict(Counter(job["status"] for job in manifest.get("jobs", []))),
                        "jobs": operational_jobs, "shared_batch_usage": manifest.get("usage", {}),
                        "configured_limits": {key: manifest["settings"].get(key) for key in ("max_requests", "max_tokens")},
                        "usage_scope": "All selected server jobs, including new unlabelled videos and failed attempts; separate from accuracy denominators"},
        "limitations": ["Customer approval is pending; these are provisional measurements, not contractual acceptance.",
                        "Only exact full-source hashes with independent labels are scored; new unlabelled videos are not treated as value errors.",
                        "Visible intervals include repeated showings; temporal boundaries and value labels are unchanged.",
                        "Historical selected-frame usability was removed from an evaluation-only copy; fresh Linux frame legibility was not independently reviewed.",
                        "Unlabelled required fields are excluded from value accuracy; the all-target E2E view is a conservative lower bound.",
                        "Operational completion, model confidence and field status recognized do not establish correct values."]}
    save(output, summary)
    print(json.dumps({"summary": str(output), "status": summary["status"],
                      "measured_originals": len(measured), "unlabelled_videos": len(unlabelled),
                      "labeled_fields": summary["accuracy"]["completed_comparable_originals"]["labeled_fields"]}, ensure_ascii=False), flush=True)
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True, help="Explicit private ground truth, read only by this post-inference process")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-round", action="append", default=[],
                        help="Also score videos from this labelling round (e.g. 2026-10-18); repeatable")
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
