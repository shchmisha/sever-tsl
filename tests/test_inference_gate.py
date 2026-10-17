"""Real local flock checks against fixtures, with no models or ground truth."""
import json
import os
import subprocess
import sys
import threading

import pytest

from sever.inference_gate import (GATE_FILENAME, InferenceGateBusy,
                                  evaluation_access, gate_path, inference_access)


def test_resolution_is_common_and_explicit_path_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("SEVER_INFERENCE_GATE_PATH", raising=False)
    monkeypatch.setenv("SEVER_RUNS_DIR", str(tmp_path / "api-runs"))
    assert gate_path() == tmp_path / "api-runs" / GATE_FILENAME
    assert gate_path(run_dir=tmp_path / "manifest-runs") == tmp_path / "manifest-runs" / GATE_FILENAME
    monkeypatch.setenv("SEVER_INFERENCE_GATE_PATH", str(tmp_path / "common.lock"))
    assert gate_path(run_dir=tmp_path / "ignored") == tmp_path / "common.lock"
    assert gate_path(tmp_path / "explicit.lock") == tmp_path / "explicit.lock"
    monkeypatch.setenv("SEVER_INFERENCE_GATE_PATH", "")
    with pytest.raises(ValueError, match="nonempty"):
        gate_path()


def test_shared_admission_and_execution_can_overlap_but_exclude_evaluation(tmp_path):
    path = tmp_path / "gate.lock"
    with inference_access(path), inference_access(path):
        with pytest.raises(InferenceGateBusy):
            with evaluation_access(path):
                pytest.fail("Evaluation overlapped inference")
    inode = path.stat().st_ino
    with evaluation_access(path):
        for access in (inference_access, evaluation_access):
            with pytest.raises(InferenceGateBusy):
                with access(path):
                    pytest.fail("Exclusive evaluation gate was bypassed")
    assert path.exists() and path.stat().st_ino == inode
    with inference_access(path):
        pass


@pytest.mark.parametrize("access", [inference_access, evaluation_access])
def test_body_failure_releases_lock_without_deleting_gate(tmp_path, access):
    path = tmp_path / "gate.lock"
    with pytest.raises(RuntimeError, match="fixture failure"):
        with access(path):
            raise RuntimeError("fixture failure")
    with evaluation_access(path):
        assert path.is_file()


def test_separate_process_obeys_exclusive_gate(tmp_path):
    path = tmp_path / "gate.lock"
    code = """
import sys
from sever.inference_gate import InferenceGateBusy, inference_access
try:
    with inference_access(sys.argv[1]):
        print('admitted')
except InferenceGateBusy:
    print('blocked')
"""
    with evaluation_access(path):
        answer = subprocess.run([sys.executable, "-c", code, str(path)], check=True,
                                capture_output=True, text=True, timeout=5)
        assert answer.stdout.strip() == "blocked"
    answer = subprocess.run([sys.executable, "-c", code, str(path)], check=True,
                            capture_output=True, text=True, timeout=5)
    assert answer.stdout.strip() == "admitted"


def test_accepted_worker_can_wait_without_crossing_the_exclusive_gate(tmp_path):
    path = tmp_path / "gate.lock"
    attempted, entered, failures = threading.Event(), threading.Event(), []
    def accepted_worker():
        attempted.set()
        try:
            with inference_access(path, blocking=True):
                entered.set()
        except BaseException as exc:
            failures.append(exc)
    with evaluation_access(path):
        worker = threading.Thread(target=accepted_worker, daemon=True)
        worker.start()
        assert attempted.wait(2)
        assert not entered.wait(.05)
        with pytest.raises(InferenceGateBusy):
            with inference_access(path):
                pytest.fail("Admission stopped being nonblocking")
    worker.join(2)
    assert not worker.is_alive() and entered.is_set() and failures == []


def test_storage_error_is_not_misreported_as_busy(tmp_path, monkeypatch):
    from sever import inference_gate
    def failure(*args, **kwargs):
        raise PermissionError("fixture permissions")
    monkeypatch.setattr(inference_gate.os, "open", failure)
    with pytest.raises(PermissionError):
        with inference_access(tmp_path / "gate.lock"):
            pass


def test_nonregular_gate_fails_without_waiting_for_a_fifo_peer(tmp_path):
    path = tmp_path / "gate-fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        with evaluation_access(path):
            pytest.fail("FIFO admitted as a process-shared gate")


def test_helper_import_does_not_load_recognition_or_provider_modules():
    code = """
import json, sys
import sever.inference_gate
prefixes=('sever.agents', 'sever.image_', 'sever.pipeline', 'pydantic_ai', 'openai')
print(json.dumps(sorted(name for name in sys.modules if name.startswith(prefixes))))
"""
    answer = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True, timeout=5)
    assert json.loads(answer.stdout) == []
