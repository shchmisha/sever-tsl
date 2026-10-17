"""Wait for inference to stop, then evaluate one immutable batch snapshot once.

This root-side controller never calls a model or the recognition API. Private
truth is opened only by the separate evaluator after all idle checks pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace
import uuid

from sever.inference_gate import (GATE_PROTOCOL, InferenceGateBusy,
                                  evaluation_access, gate_path)


JOB_TERMINAL = {"completed", "completed_with_errors", "failed"}
BATCH_TERMINAL = {"completed", "completed_with_errors", "budget_exhausted",
                  "provider_failure", "provider_verification_failed", "driver_error", "failed"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def driver_state(unit):
    answer = subprocess.run(["systemctl", "show", "--property=LoadState", "--property=ActiveState", "--", unit],
                            check=True, capture_output=True, text=True, timeout=10)
    fields = dict(line.split("=", 1) for line in answer.stdout.splitlines() if "=" in line)
    if fields.get("LoadState") != "loaded" or not fields.get("ActiveState"):
        raise RuntimeError("The requested driver unit is not a loaded service")
    return {"unit": unit, "load_state": fields["LoadState"], "active_state": fields["ActiveState"]}


def _api_states(run_dir, jobs):
    states, uncertain = {}, []
    if not run_dir.is_dir():
        raise ValueError("The manifest run directory is unavailable")
    for directory in sorted(run_dir.iterdir()):
        try:
            identifier = str(uuid.UUID(directory.name))
        except ValueError:
            continue
        path = directory / "job.json"
        if directory.resolve().parent != run_dir or path.resolve().parent != directory.resolve():
            raise ValueError("An API job state is outside its job directory")
        if not path.is_file():
            uncertain.append({"job_id": identifier, "status": "missing_api_state"})
            continue
        data = json.loads(path.read_bytes())
        if data.get("job_id") != identifier:
            raise ValueError("An API job state has a different job identity")
        states[identifier] = data.get("status", "unknown")
    for job in jobs:
        identifier = job.get("job_id")
        if not identifier:
            uncertain.append({"job_id": None, "status": "missing_job_identity"})
            continue
        if identifier and identifier not in states:
            uncertain.append({"job_id": identifier, "status": "missing_api_state"})
        elif job.get("status") in JOB_TERMINAL and states[identifier] != job["status"]:
            uncertain.append({"job_id": identifier, "status": "manifest_api_state_mismatch"})
    uncertain.extend({"job_id": identifier, "status": status}
                     for identifier, status in states.items() if status not in JOB_TERMINAL)
    return states, uncertain


def _snapshot(path, service):
    if service.get("load_state") != "loaded":
        raise ValueError("The requested driver service is unavailable")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    sources, jobs = manifest["sources"], manifest.get("jobs", [])
    if not isinstance(sources, list) or not sources or not isinstance(jobs, list):
        raise ValueError("The manifest must have a nonempty source inventory and a job list")
    by_source = {(source["name"], source["sha256"]): source for source in sources}
    if len(by_source) != len(sources):
        raise ValueError("The manifest has duplicate source identities")
    matched = {key: [] for key in by_source}
    for job in jobs:
        key = (job["source"]["name"], job["source"]["sha256"])
        if key not in by_source or job["source"] != by_source[key]:
            raise ValueError("A job source differs from the selected inventory")
        matched[key].append(job)
    run_dir = Path(manifest["settings"]["run_dir"]).resolve()
    states, uncertain = _api_states(run_dir, jobs)
    reasons = []
    if service["active_state"] not in {"inactive", "failed"}:
        reasons.append("driver_not_inactive")
    terminal = manifest.get("status") in BATCH_TERMINAL
    driver_failed = service["active_state"] == "failed"
    if not terminal and not driver_failed:
        reasons.append("batch_not_terminal")
    if any(job.get("status") not in JOB_TERMINAL for job in jobs):
        reasons.append("manifest_has_active_or_uncertain_jobs")
    if uncertain:
        reasons.append("api_has_active_or_uncertain_jobs")
    completed = [source for key, source in by_source.items()
                 if any(job.get("status") in {"completed", "completed_with_errors"} for job in matched[key])]
    missing = [{"source": source, "job_statuses": [job.get("status") for job in matched[key]] or ["not_submitted"]}
               for key, source in by_source.items() if source not in completed]
    summary = {"manifest_path": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "batch_status": manifest.get("status"), "driver": service,
        "terminal_reason": "driver_error" if driver_failed and not terminal else manifest.get("status"),
        "source_count": len(sources), "job_count": len(jobs),
        "job_status_counts": dict(Counter(job.get("status", "unknown") for job in jobs)),
        "completed_job_count": sum(job.get("status") in {"completed", "completed_with_errors"} for job in jobs),
        "completed_sources": completed, "missing_sources": missing,
        "api_job_status_counts": dict(Counter(states.values())), "active_or_uncertain_jobs": uncertain,
        "evaluation_ready": not reasons, "blocking_reasons": reasons}
    return raw, manifest, summary


def _outside_jobs(path, run_dir):
    try:
        first = path.relative_to(run_dir).parts[0]
        uuid.UUID(first)
    except (ValueError, IndexError):
        return
    raise ValueError("Controller outputs cannot be inside API job artifact directories")


def _evaluate(args):
    # Load the sibling evaluator from this isolated deployment, without relying
    # on the active API's checkout or importing any recognition/provider code.
    spec = importlib.util.spec_from_file_location("sever_post_inference_evaluator",
                                                Path(__file__).with_name("evaluate_server_results.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(args)


def run(args, *, evaluator=None, service_state=None, sleep=time.sleep, monotonic=time.monotonic):
    poll, timeout = float(args.poll_seconds), float(args.wait_timeout)
    if not math.isfinite(poll) or not 0 < poll <= 60 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Poll must be between 0 and 60 seconds and wait timeout must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*", args.driver_unit):
        raise ValueError("Provide one explicit driver service unit")
    # Resolving a pathname does not open or inspect private truth contents.
    manifest_path, truth_path, output = (Path(value).expanduser().resolve()
                                        for value in (args.manifest, args.truth, args.out))
    evaluation_output = output.with_name(output.stem + "-evaluation.json")
    frozen_manifest = output.with_name(output.stem + "-manifest.json")
    lock_path = output.with_name(output.stem + ".lock")
    outputs = (output, evaluation_output, frozen_manifest, lock_path)
    if output.suffix != ".json" or any(path in {manifest_path, truth_path} for path in outputs):
        raise ValueError("Controller outputs must be separate JSON artifacts and cannot overwrite inputs")
    initial = json.loads(manifest_path.read_bytes())
    run_dir = Path(initial["settings"]["run_dir"]).resolve()
    inference_gate_path = gate_path(getattr(args, "inference_gate", None), run_dir=run_dir)
    if inference_gate_path in {*outputs, manifest_path, truth_path}:
        raise ValueError("Inference gate must be separate from controller inputs and outputs")
    for path in (*outputs, inference_gate_path):
        _outside_jobs(path, run_dir)
    configuration = {"manifest_path": str(manifest_path), "truth_path": str(truth_path),
                     "driver_unit": args.driver_unit, "evaluation_output": str(evaluation_output),
                     "inference_gate_path": str(inference_gate_path), "inference_gate_protocol": GATE_PROTOCOL}
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, ExitStack() as held_gate:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller owns this output") from None
        if output.exists():
            previous = json.loads(output.read_bytes())
            if previous.get("configuration") != configuration:
                raise ValueError("Existing controller inputs differ; use a new output path")
            if previous.get("status") == "evaluated":
                if (hashlib.sha256(manifest_path.read_bytes()).hexdigest() != previous["snapshot"]["manifest_sha256"]
                        or hashlib.sha256(evaluation_output.read_bytes()).hexdigest() != previous["evaluation"]["sha256"]):
                    raise ValueError("A completed controller input/output changed; use a new output path")
                return previous
            if previous.get("evaluation_attempted"):
                raise RuntimeError("Evaluation was already attempted; inspect its outcome before using a new output path")
        elif evaluation_output.exists() or frozen_manifest.exists():
            raise ValueError("Existing evaluation artifacts require a new controller output path")
        state = {"schema_version": 2, "status": "waiting", "started_at": _now(), "configuration": configuration,
                 "evaluation_attempted": False, "truth_loaded_before_idle": False,
                 "model_or_api_calls": 0, "customer_approved": False, "contractual_acceptance": False,
                 "inference_gate": {"path": str(inference_gate_path), "protocol": GATE_PROTOCOL,
                                    "held_during_evaluation": False}}
        started = monotonic()
        service_state = service_state or driver_state
        evaluator = evaluator or _evaluate
        stage = "waiting_for_inference"
        try:
            while True:
                raw, manifest, snapshot = _snapshot(manifest_path, service_state(args.driver_unit))
                if Path(manifest["settings"]["run_dir"]).resolve() != run_dir:
                    raise ValueError("The batch run directory changed during waiting")
                state.update(snapshot=snapshot, updated_at=_now(), waited_seconds=monotonic() - started)
                _save(output, state)
                if snapshot["evaluation_ready"]:
                    try:
                        with ExitStack() as candidate_gate:
                            candidate_gate.enter_context(evaluation_access(inference_gate_path))
                            # Admission and workers cannot enter while this
                            # repeated idle snapshot and private evaluation run.
                            repeated, _, verified = _snapshot(manifest_path, service_state(args.driver_unit))
                            if verified["evaluation_ready"] and repeated == raw:
                                state["snapshot"] = verified
                                state["inference_gate"].update(held_during_evaluation=True,
                                                               acquired_at=_now())
                                held_gate.callback(candidate_gate.pop_all().close)
                                break
                            if repeated != raw:
                                verified["evaluation_ready"] = False
                                verified["blocking_reasons"].append("manifest_changed_at_evaluation_boundary")
                            state.update(snapshot=verified, updated_at=_now())
                            _save(output, state)
                    except InferenceGateBusy:
                        snapshot["evaluation_ready"] = False
                        snapshot["blocking_reasons"].append("inference_gate_busy")
                        state.update(snapshot=snapshot, updated_at=_now())
                        _save(output, state)
                remaining = timeout - (monotonic() - started)
                if remaining <= 0:
                    state.update(status="timed_out", updated_at=_now())
                    _save(output, state)
                    return state
                sleep(min(poll, remaining))
            stage = "evaluation"
            temporary = frozen_manifest.with_name(f".{frozen_manifest.name}-{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(raw)
                temporary.replace(frozen_manifest)
            finally:
                temporary.unlink(missing_ok=True)
            state.update(status="evaluating", evaluation_attempted=True,
                         manifest_snapshot_path=str(frozen_manifest), updated_at=_now())
            _save(output, state)
            summary = evaluator(SimpleNamespace(manifest=frozen_manifest, truth=truth_path, out=evaluation_output))
            if summary.get("inputs", {}).get("manifest_sha256") != state["snapshot"]["manifest_sha256"]:
                raise ValueError("Evaluation did not use the exact gated manifest snapshot")
            if summary.get("accuracy", {}).get("all_four_originals_unprocessed_as_misses", {}).get("scored_video_count") != 4:
                raise ValueError("Evaluation did not retain the four original video denominators")
            state.update(status="evaluated", updated_at=_now(), evaluation={"path": str(evaluation_output),
                "sha256": hashlib.sha256(evaluation_output.read_bytes()).hexdigest(), "status": summary["status"]},
                all_four_original_denominators_retained=True)
            _save(output, state)
            return state
        except BaseException as exc:
            state.update(status="failed", updated_at=_now(), failure={"stage": stage, "error_type": type(exc).__name__})
            _save(output, state)
            raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Controller JSON; evaluation and frozen manifest are sibling artifacts")
    parser.add_argument("--driver-unit", required=True)
    parser.add_argument("--inference-gate", type=Path,
                        help="Shared persistent flock file; defaults to SEVER_INFERENCE_GATE_PATH or the manifest run directory")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--wait-timeout", type=float, default=172800)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(arguments())
    print(json.dumps({"status": result["status"], "evaluation": result.get("evaluation"),
                      "snapshot": result.get("snapshot")}, ensure_ascii=False))
    if result["status"] != "evaluated":
        raise SystemExit(1)
