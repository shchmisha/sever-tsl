"""Fixtures only: controller must hold the gate before opening evaluator inputs."""
from contextlib import contextmanager
import json
from pathlib import Path
import uuid

import pytest

from sever.inference_gate import InferenceGateBusy, evaluation_access, gate_path, inference_access
from test_finish_server_validation import (case, controller, evaluator_stub,
                                          forbid_truth_reads, run, save, service)


def path_for(args):
    manifest = json.loads(args.manifest.read_bytes())
    return gate_path(getattr(args, "inference_gate", None), run_dir=manifest["settings"]["run_dir"])


@pytest.mark.parametrize("held_by", [inference_access, evaluation_access])
def test_busy_gate_never_opens_truth_even_with_idle_saved_jobs(tmp_path, monkeypatch, held_by):
    args, _ = case(tmp_path)
    reads = forbid_truth_reads(monkeypatch, args.truth)
    calls = []
    with held_by(path_for(args)):
        result = run(args, evaluator=evaluator_stub(calls))
    assert result["status"] == "timed_out"
    assert result["evaluation_attempted"] is False
    assert result["inference_gate"]["held_during_evaluation"] is False
    assert result["snapshot"]["blocking_reasons"] == ["inference_gate_busy"]
    assert not result["snapshot"]["evaluation_ready"]
    assert calls == reads == []


def test_gate_held_before_repeated_idle_snapshot_and_through_evaluation(tmp_path):
    args, _ = case(tmp_path, source_count=4)
    path = path_for(args)
    calls, checks = [], []
    base = evaluator_stub(calls)
    def current_service(unit):
        checks.append(unit)
        if len(checks) == 2:
            with pytest.raises(InferenceGateBusy):
                with inference_access(path):
                    pytest.fail("New admission entered during the repeated idle snapshot")
        return service()(unit)
    def evaluate(value):
        with pytest.raises(InferenceGateBusy):
            with inference_access(path):
                pytest.fail("New inference entered while fixture evaluator reads its inputs")
        saved = json.loads(args.out.read_text())
        assert saved["status"] == "evaluating"
        assert saved["inference_gate"]["held_during_evaluation"] is True
        return base(value)
    result = run(args, evaluator=evaluate, service_state=current_service)
    assert result["status"] == "evaluated" and len(calls) == 1
    assert len(checks) == 2
    assert result["configuration"]["inference_gate_path"] == str(path)
    with inference_access(path):
        pass


def test_job_admitted_between_first_snapshot_and_gate_is_caught_by_recheck(tmp_path, monkeypatch):
    args, manifest = case(tmp_path)
    reads = forbid_truth_reads(monkeypatch, args.truth)
    path, original = path_for(args), controller.evaluation_access
    other = str(uuid.uuid4())
    @contextmanager
    def raced_gate(selected):
        # This represents an admission that completed just before exclusive
        # acquisition; its queued job must prevent evaluation after recheck.
        with inference_access(path):
            save(Path(manifest["settings"]["run_dir"]) / other / "job.json",
                 {"job_id": other, "status": "queued"})
        with original(selected) as held:
            yield held
    monkeypatch.setattr(controller, "evaluation_access", raced_gate)
    calls = []
    result = run(args, evaluator=evaluator_stub(calls))
    assert result["status"] == "timed_out" and calls == reads == []
    assert "api_has_active_or_uncertain_jobs" in result["snapshot"]["blocking_reasons"]
    with inference_access(path):
        pass


def test_evaluator_failure_releases_gate_and_still_cannot_repeat(tmp_path):
    args, _ = case(tmp_path)
    path, calls = path_for(args), []
    def failure(value):
        calls.append(value)
        with pytest.raises(InferenceGateBusy):
            with inference_access(path):
                pass
        raise RuntimeError("fixture evaluation failure")
    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        run(args, evaluator=failure)
    with inference_access(path):
        pass
    state = json.loads(args.out.read_text())
    assert state["evaluation_attempted"] and state["status"] == "failed"
    with pytest.raises(RuntimeError, match="already attempted"):
        run(args, evaluator=failure)
    assert len(calls) == 1


def test_explicit_gate_is_pinned_and_cannot_be_changed_on_resume(tmp_path):
    args, _ = case(tmp_path)
    args.inference_gate = tmp_path / "common.lock"
    calls = []
    result = run(args, evaluator=evaluator_stub(calls))
    assert result["configuration"]["inference_gate_path"] == str(args.inference_gate)
    args.inference_gate = tmp_path / "different.lock"
    with pytest.raises(ValueError, match="inputs differ"):
        run(args, evaluator=evaluator_stub(calls))
    assert len(calls) == 1


@pytest.mark.parametrize("input_name", ["truth", "manifest", "out"])
def test_gate_cannot_be_an_input_or_output_and_never_opens_truth(tmp_path, monkeypatch, input_name):
    args, _ = case(tmp_path)
    args.inference_gate = getattr(args, input_name)
    reads = forbid_truth_reads(monkeypatch, args.truth)
    with pytest.raises(ValueError, match="separate"):
        run(args)
    assert reads == []


def test_cli_accepts_explicit_common_gate_path(tmp_path):
    args = controller.arguments(["--manifest", str(tmp_path / "batch.json"), "--truth", str(tmp_path / "truth.json"),
        "--out", str(tmp_path / "finish.json"), "--driver-unit", "fixture.service",
        "--inference-gate", str(tmp_path / "common.lock")])
    assert args.inference_gate == tmp_path / "common.lock"
