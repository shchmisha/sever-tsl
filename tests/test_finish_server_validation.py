"""Offline post-inference gating; no server, model or recognition API calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest


spec = importlib.util.spec_from_file_location(
    "finish_server_validation", Path(__file__).parents[1] / "deploy/finish_server_validation.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def case(tmp_path, *, status="completed", job_status="completed", source_count=1):
    runs = tmp_path / "runs"
    runs.mkdir()
    sources = [{"name": f"original-{i}.mp4", "path": str(tmp_path / f"original-{i}.mp4"),
                "bytes": 10 + i, "sha256": hashlib.sha256(f"source-{i}".encode()).hexdigest()}
               for i in range(source_count)]
    job_id = str(uuid.uuid4())
    manifest = {"status": status, "settings": {"run_dir": str(runs)}, "sources": sources,
                "jobs": [{"source": sources[0], "job_id": job_id, "status": job_status}]}
    manifest_path = runs / "batches" / "offline" / "batch.json"
    save(manifest_path, manifest)
    save(runs / job_id / "job.json", {"job_id": job_id, "status": job_status})
    truth = tmp_path / "private" / "truth.json"
    save(truth, {"fixture": "must remain unread until idle"})
    truth.chmod(0o600)
    args = SimpleNamespace(manifest=manifest_path, truth=truth, out=tmp_path / "evaluation" / "finish.json",
                           driver_unit="offline-driver.service", poll_seconds=.1, wait_timeout=.2)
    return args, manifest


def service(status="inactive"):
    return lambda unit: {"unit": unit, "load_state": "loaded", "active_state": status}


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def run(args, **kwargs):
    clock = Clock()
    return controller.run(args, sleep=clock.sleep, monotonic=clock.monotonic,
                          service_state=kwargs.pop("service_state", service()), **kwargs)


def forbid_truth_reads(monkeypatch, truth):
    original_open = Path.open
    reads = []

    def guarded(path, mode="r", *args, **kwargs):
        if path.resolve() == truth.resolve() and ("r" in mode or "+" in mode):
            reads.append(mode)
            raise AssertionError("Private truth read before all inference idle checks")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    return reads


def evaluator_stub(calls):
    def evaluate(args):
        calls.append(args)
        args.truth.read_text()
        summary = {"status": "partial_provisional_original_subset", "inputs": {
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()},
            "accuracy": {"all_four_originals_unprocessed_as_misses": {"scored_video_count": 4}}}
        save(args.out, summary)
        return summary
    return evaluate


@pytest.mark.parametrize("condition", ["pending_batch", "active_driver", "active_api",
                                      "uncertain_job", "missing_state", "missing_identity",
                                      "unrelated_active_api", "unrelated_unknown_directory",
                                      "terminal_state_mismatch"])
def test_pending_or_uncertain_inference_never_opens_truth(tmp_path, monkeypatch, condition):
    args, manifest = case(tmp_path)
    active = service()
    job_id = manifest["jobs"][0]["job_id"]
    job_path = Path(manifest["settings"]["run_dir"]) / job_id / "job.json"
    if condition == "pending_batch":
        manifest["status"] = "processing"
    elif condition == "active_driver":
        active = service("active")
    elif condition == "active_api":
        save(job_path, {"job_id": job_id, "status": "processing"})
    elif condition == "uncertain_job":
        manifest["jobs"][0]["status"] = "submitting"
    elif condition == "missing_state":
        job_path.unlink()
    elif condition == "missing_identity":
        del manifest["jobs"][0]["job_id"]
    elif condition == "unrelated_active_api":
        other = str(uuid.uuid4())
        save(job_path.parent.parent / other / "job.json", {"job_id": other, "status": "queued"})
    elif condition == "unrelated_unknown_directory":
        (job_path.parent.parent / str(uuid.uuid4())).mkdir()
    elif condition == "terminal_state_mismatch":
        save(job_path, {"job_id": job_id, "status": "failed"})
    save(args.manifest, manifest)
    reads = forbid_truth_reads(monkeypatch, args.truth)
    calls = []
    result = run(args, service_state=active, evaluator=evaluator_stub(calls))
    assert result["status"] == "timed_out"
    assert result["evaluation_attempted"] is False
    assert result["snapshot"]["blocking_reasons"]
    assert result["snapshot"]["source_count"] == result["snapshot"]["job_count"] == 1
    assert result["snapshot"]["manifest_sha256"] == hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    assert reads == calls == []
    assert not args.out.with_name("finish-evaluation.json").exists()
    assert json.loads(args.out.read_text())["status"] == "timed_out"


def test_terminal_idle_snapshot_evaluates_once_and_preserves_exact_bytes(tmp_path, monkeypatch):
    args, manifest = case(tmp_path, source_count=4)
    before = args.manifest.read_bytes()
    calls = []
    result = run(args, evaluator=evaluator_stub(calls))
    assert result["status"] == "evaluated" and result["evaluation_attempted"] is True
    assert len(calls) == 1
    assert calls[0].manifest != args.manifest and calls[0].manifest.read_bytes() == before
    assert args.manifest.read_bytes() == before
    assert calls[0].out == args.out.with_name("finish-evaluation.json")
    assert result["snapshot"]["source_count"] == 4
    assert result["snapshot"]["completed_job_count"] == 1
    assert result["snapshot"]["completed_sources"] == manifest["sources"][:1]
    assert len(result["snapshot"]["missing_sources"]) == 3
    assert all(item["job_statuses"] == ["not_submitted"] for item in result["snapshot"]["missing_sources"])
    assert result["all_four_original_denominators_retained"] is True
    assert result["model_or_api_calls"] == 0 and result["contractual_acceptance"] is False
    reads = forbid_truth_reads(monkeypatch, args.truth)
    assert run(args, evaluator=evaluator_stub(calls)) == result
    assert len(calls) == 1 and reads == []


def test_final_boundary_rechecks_driver_before_truth_is_allowed(tmp_path, monkeypatch):
    args, _ = case(tmp_path)
    sequence = iter(["inactive", "active", "active", "active"])
    reads = forbid_truth_reads(monkeypatch, args.truth)
    calls = []
    result = run(args, service_state=lambda unit: service(next(sequence))(unit), evaluator=evaluator_stub(calls))
    assert result["status"] == "timed_out" and calls == reads == []


@pytest.mark.parametrize("failure_status", ["budget_exhausted", "provider_failure", "driver_error"])
def test_terminal_failure_can_evaluate_missing_sources_after_idle(tmp_path, failure_status):
    args, _ = case(tmp_path, status=failure_status, job_status="failed", source_count=4)
    calls = []
    result = run(args, evaluator=evaluator_stub(calls))
    assert result["status"] == "evaluated" and len(calls) == 1
    assert result["snapshot"]["terminal_reason"] == failure_status
    assert result["snapshot"]["completed_job_count"] == 0
    assert len(result["snapshot"]["missing_sources"]) == 4


def test_failed_driver_with_no_live_jobs_is_recorded_as_terminal_driver_error(tmp_path):
    args, _ = case(tmp_path, status="processing", job_status="failed", source_count=4)
    calls = []
    result = run(args, service_state=service("failed"), evaluator=evaluator_stub(calls))
    assert result["status"] == "evaluated" and len(calls) == 1
    assert result["snapshot"]["batch_status"] == "processing"
    assert result["snapshot"]["terminal_reason"] == "driver_error"
    assert len(result["snapshot"]["missing_sources"]) == 4


def test_actual_evaluator_keeps_four_original_denominators_when_only_one_completed(tmp_path):
    args, manifest = case(tmp_path, status="budget_exhausted", source_count=4)
    videos = [{"source": source["path"], "source_sha256": source["sha256"], "pages": [
        {"start_s": 0, "end_s": 2, "document_type": "ttn", "fields": {"document_number": "0001"}}]}
        for source in manifest["sources"]]
    save(args.truth, {"customer_approved": False, "videos": videos})
    job = manifest["jobs"][0]
    predicted = {"source": {"sha256": job["source"]["sha256"]}, "pages": [{
        "page_id": "one", "selected_frame_id": "primary", "frames": [{"frame_id": "primary", "timestamp_s": 1}],
        "result": {"document_type": "ttn", "errors": [], "fields": {
            "document_number": {"status": "recognized", "normalized_value": "0001"}}}}]}
    result_path = Path(manifest["settings"]["run_dir"]) / job["job_id"] / "results.json"
    save(result_path, predicted)
    before = {path: path.read_bytes() for path in (args.manifest, args.truth, result_path)}
    result = run(args)
    summary = json.loads(Path(result["evaluation"]["path"]).read_text())
    conservative = summary["accuracy"]["all_four_originals_unprocessed_as_misses"]
    assert conservative["scored_video_count"] == 4
    assert conservative["candidate_coverage"] == {"correct": 1, "total": 4, "rate": .25}
    assert conservative["labeled_fields"] == {"correct": 1, "total": 4, "rate": .25}
    assert summary["ground_truth_counts"]["required_target_fields"] == 16
    assert len(summary["missing_or_unprocessed_originals"]) == 3
    assert summary["status"] == "partial_provisional_original_subset"
    assert summary["customer_approved"] is summary["contractual_acceptance"] is False
    assert summary["inputs"]["manifest_sha256"] == result["snapshot"]["manifest_sha256"]
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert all(not Path(entry["metrics_path"]).is_relative_to(result_path.parent) for entry in summary["originals"])


def test_evaluator_failure_records_attempt_and_cannot_silently_repeat(tmp_path):
    args, _ = case(tmp_path)
    calls = []

    def failing(evaluation):
        calls.append(evaluation)
        evaluation.truth.read_text()
        raise RuntimeError("controlled scoring failure")

    with pytest.raises(RuntimeError, match="controlled scoring failure"):
        run(args, evaluator=failing)
    checkpoint = json.loads(args.out.read_text())
    assert checkpoint["status"] == "failed" and checkpoint["evaluation_attempted"] is True
    assert checkpoint["failure"] == {"stage": "evaluation", "error_type": "RuntimeError"}
    assert "evaluation" not in checkpoint
    with pytest.raises(RuntimeError, match="already attempted"):
        run(args, evaluator=failing)
    assert len(calls) == 1


def test_driver_state_failure_records_checkpoint_without_opening_truth(tmp_path, monkeypatch):
    args, _ = case(tmp_path)
    reads = forbid_truth_reads(monkeypatch, args.truth)

    def failure(unit):
        raise OSError("controlled service read failure")

    with pytest.raises(OSError):
        run(args, service_state=failure)
    state = json.loads(args.out.read_text())
    assert state["status"] == "failed" and state["evaluation_attempted"] is False
    assert state["failure"]["stage"] == "waiting_for_inference"
    assert reads == []


@pytest.mark.parametrize("bad_proof", ["different_manifest", "three_originals"])
def test_invalid_evaluator_proof_never_claims_success(tmp_path, bad_proof):
    args, _ = case(tmp_path)
    calls = []
    base = evaluator_stub(calls)

    def invalid(evaluation):
        summary = base(evaluation)
        if bad_proof == "different_manifest":
            summary["inputs"]["manifest_sha256"] = "0" * 64
        else:
            summary["accuracy"]["all_four_originals_unprocessed_as_misses"]["scored_video_count"] = 3
        save(evaluation.out, summary)
        return summary

    with pytest.raises(ValueError):
        run(args, evaluator=invalid)
    state = json.loads(args.out.read_text())
    assert state["status"] == "failed" and "evaluation" not in state
    assert len(calls) == 1


def test_controller_cannot_write_inside_api_job_directories(tmp_path, monkeypatch):
    args, manifest = case(tmp_path)
    args.out = Path(manifest["settings"]["run_dir"]) / manifest["jobs"][0]["job_id"] / "score.json"
    reads = forbid_truth_reads(monkeypatch, args.truth)
    with pytest.raises(ValueError, match="API job"):
        run(args)
    assert not args.out.exists() and reads == []


@pytest.mark.parametrize("input_name", ["manifest", "truth"])
def test_controller_cannot_overwrite_inputs(tmp_path, input_name):
    args, _ = case(tmp_path)
    args.out = getattr(args, input_name)
    before = args.out.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        run(args)
    assert args.out.read_bytes() == before


@pytest.mark.parametrize("poll,timeout", [(0, 1), (61, 1), (float("nan"), 1), (.1, 0), (.1, float("inf"))])
def test_poll_and_timeout_are_finite_and_bounded(tmp_path, poll, timeout):
    args, _ = case(tmp_path)
    args.poll_seconds, args.wait_timeout = poll, timeout
    with pytest.raises(ValueError, match="Poll"):
        run(args)


def test_driver_service_read_is_local_read_only_systemctl(monkeypatch):
    calls = []

    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="LoadState=loaded\nActiveState=inactive\n")

    monkeypatch.setattr(controller.subprocess, "run", fake)
    assert controller.driver_state("test.service")["active_state"] == "inactive"
    command, options = calls[0]
    assert command == ["systemctl", "show", "--property=LoadState", "--property=ActiveState", "--", "test.service"]
    assert options["timeout"] == 10 and options["check"] is True
