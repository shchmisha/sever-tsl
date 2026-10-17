"""Job ownership and interruption reporting; no inference or source replay."""
from concurrent.futures import Future
from datetime import datetime
import json
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import pytest

from sever import api
from sever.inference_gate import evaluation_access, gate_path, InferenceGateBusy


@pytest.fixture
def isolated_api(monkeypatch, tmp_path):
    root = tmp_path / "runs"
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "_ACTIVE_FUTURES", set())
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setattr(api, "agent_config", lambda *args: SimpleNamespace(refresh_cache=False))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"Offline admission fixture, never decoded")
    submitted = []

    class Worker:
        def submit(self, *args):
            submitted.append(args)

    monkeypatch.setattr(api, "WORKER", Worker())
    monkeypatch.setattr(api, "process_video", lambda *args, **kwargs: pytest.fail("Inference is forbidden"))
    return root, source, submitted


def save_job(root, status, **extra):
    job_id = str(uuid.uuid4())
    path = root / job_id / "job.json"
    path.parent.mkdir(parents=True)
    saved = {"job_id": job_id, "status": status, "source_path": "/source/original.mp4", **extra}
    path.write_text(json.dumps(saved))
    return job_id, path, saved


def test_startup_marks_only_interrupted_jobs_and_preserves_every_artifact(isolated_api):
    root, _, submitted = isolated_api
    pending = [save_job(root, status, enqueued_at="2026-10-10T10:00:00+00:00",
                        native_implementation_fingerprint="a" * 64, progress="Historical progress")
               for status in ("queued", "processing")]
    terminal = [save_job(root, status, error_type="OldFailure" if status == "failed" else None)
                for status in ("completed", "completed_with_errors", "failed")]
    artifact_bytes = {}
    for _, path, _ in pending + terminal:
        for name in ("partial-results.json", "results.json", "report.html", "trace.jsonl", "raw.bin"):
            artifact = path.parent / name
            artifact.write_bytes(f"Preserve {artifact.parent.name}/{name} exactly".encode())
            artifact_bytes[artifact] = artifact.read_bytes()
    terminal_bytes = {path: path.read_bytes() for _, path, _ in terminal}
    with TestClient(api.app) as client:
        for job_id, _, saved in pending:
            status = client.get(f"/v1/jobs/{job_id}").json()
            assert status["status"] == "failed"
            assert status["error_type"] == "InterruptedByServiceRestart"
            assert status["interrupted_status"] == saved["status"]
            assert status["source_path"] == saved["source_path"]
            assert status["enqueued_at"] == saved["enqueued_at"]
            assert status["native_implementation_fingerprint"] == "a" * 64
            assert status["progress"] == "Historical progress"
            assert datetime.fromisoformat(status["finished_at"]).tzinfo is not None
        assert submitted == []
    assert all(path.read_bytes() == before for path, before in artifact_bytes.items())
    assert all(path.read_bytes() == before for path, before in terminal_bytes.items())
    # A second startup does not rewrite an already terminal interruption.
    interrupted_bytes = {path: path.read_bytes() for _, path, _ in pending}
    with TestClient(api.app):
        pass
    assert all(path.read_bytes() == before for path, before in interrupted_bytes.items())


def test_second_api_instance_refused_before_reconciling_current_jobs(isolated_api):
    root, source, submitted = isolated_api
    with TestClient(api.app) as first:
        created = first.post("/v1/jobs", json={"source_path": str(source)}).json()
        before = (root / created["job_id"] / "job.json").read_bytes()
        with pytest.raises(RuntimeError, match="Another API process"):
            with TestClient(api.app):
                pytest.fail("A second API instance must not enter admission")
        assert (root / created["job_id"] / "job.json").read_bytes() == before
        assert first.get(f"/v1/jobs/{created['job_id']}").json()["status"] == "queued"
        assert len(submitted) == 1


def test_startup_will_not_reconcile_during_isolated_evaluation(isolated_api):
    root, _, submitted = isolated_api
    _, path, _ = save_job(root, "processing")
    before = path.read_bytes()
    with evaluation_access(gate_path(run_dir=root)):
        with pytest.raises(InferenceGateBusy):
            with TestClient(api.app):
                pytest.fail("Evaluation must retain exclusive ownership")
    assert path.read_bytes() == before and submitted == []


@pytest.mark.parametrize("failed", [False, True])
def test_job_keeps_source_and_lifecycle_times_after_progress_and_finish(monkeypatch, isolated_api, failed):
    _, source, submitted = isolated_api
    observed = []

    async def process(source, out, *, progress, **kwargs):
        observed.append(api.get_job(out.name))
        progress("Offline lifecycle progress")
        observed.append(api.get_job(out.name))
        if failed:
            raise RuntimeError("Controlled failure")
        return {"pages": [], "usage": {"requests": 0}}

    monkeypatch.setattr(api, "process_video", process)
    with TestClient(api.app) as client:
        created = client.post("/v1/jobs", json={"source_path": str(source)}).json()
        queued = client.get(f"/v1/jobs/{created['job_id']}").json()
        function, *args = submitted[0]
        function(*args)
        final = client.get(f"/v1/jobs/{created['job_id']}").json()
        assert final["status"] == ("failed" if failed else "completed")
        assert final["source_path"] == str(source.resolve())
        assert final["enqueued_at"] == queued["enqueued_at"]
        assert "started_at" not in queued and "finished_at" not in queued
        assert observed[0]["started_at"] == observed[1]["started_at"] == final["started_at"]
        times = [datetime.fromisoformat(final[name]) for name in ("enqueued_at", "started_at", "finished_at")]
        assert all(moment.tzinfo is not None for moment in times)
        assert times == sorted(times)


def test_queue_submission_failure_is_terminal_and_visible(monkeypatch, isolated_api):
    root, source, _ = isolated_api

    class FailedWorker:
        def submit(self, *args):
            raise RuntimeError("Worker has stopped")

    monkeypatch.setattr(api, "WORKER", FailedWorker())
    with TestClient(api.app) as client:
        response = client.post("/v1/jobs", json={"source_path": str(source)})
        assert response.status_code == 503
        paths = list(root.glob("*/job.json"))
        assert len(paths) == 1
        saved = json.loads(paths[0].read_text())
        assert saved["status"] == "failed" and saved["error_type"] == "WorkerUnavailable"
        assert saved["source_path"] == str(source.resolve())
        assert saved["finished_at"] and "started_at" not in saved


def test_graceful_shutdown_keeps_instance_lock_until_accepted_worker_finishes(monkeypatch, isolated_api):
    from threading import Event, Thread

    root, source, _ = isolated_api
    future = Future()
    draining = Event()
    closed = Event()
    real_wait = api.wait

    class PendingWorker:
        def submit(self, *args):
            return future

    def observed_wait(futures):
        draining.set()
        return real_wait(futures)

    monkeypatch.setattr(api, "WORKER", PendingWorker())
    monkeypatch.setattr(api, "wait", observed_wait)
    client = TestClient(api.app)
    client.__enter__()
    response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 202

    def close():
        client.__exit__(None, None, None)
        closed.set()

    thread = Thread(target=close)
    thread.start()
    try:
        assert draining.wait(timeout=2)
        assert not closed.is_set()
        with pytest.raises(RuntimeError, match="Another API process"):
            with api.api_instance_access():
                pytest.fail("Lock released while worker is still writing")
        future.set_result(None)
        assert closed.wait(timeout=2)
    finally:
        if not future.done():
            future.set_result(None)
        thread.join(timeout=2)
    with api.api_instance_access():
        pass
    assert api._ACTIVE_FUTURES == set()


def test_malformed_job_state_blocks_startup_without_overwriting_evidence(isolated_api):
    root, _, submitted = isolated_api
    _, path, _ = save_job(root, "processing")
    path.write_text("truncated job state")
    before = path.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        with TestClient(api.app):
            pytest.fail("Unknown persisted state cannot be reconciled safely")
    assert path.read_bytes() == before and submitted == []
