"""Offline service-switch recovery; no server, model or customer artifacts."""
from contextlib import contextmanager
import fcntl
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import uuid

import pytest


spec = importlib.util.spec_from_file_location(
    "switch_release_verified_fields", Path(__file__).parents[1] / "deploy/switch_release_verified_fields.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def case(tmp_path):
    releases = tmp_path / "releases"
    old, new = releases / "old", releases / "new"
    for release in (old, new):
        (release / ".venv/bin").mkdir(parents=True)
        for name in ("python", "sever"):
            (release / ".venv/bin" / name).write_text("offline fixture; never executed")
    current = tmp_path / "current"
    current.symlink_to(old, target_is_directory=True)
    runs = tmp_path / "runs"
    runs.mkdir()
    gate = runs / ".inference-evaluation.lock"
    gate.touch()
    db = tmp_path / "portal.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE jobs (id TEXT, status TEXT)")
    args = SimpleNamespace(release=new, current=current, releases=releases, runs=runs,
                           portal_db=db, inference_gate=gate, dry_run=False,
                           expected_fingerprint="b" * 64, health_attempts=1, health_interval=.01)
    return args, old, new


def portal_job(args, status):
    with sqlite3.connect(args.portal_db) as connection:
        connection.execute("INSERT INTO jobs VALUES (?,?)", (str(uuid.uuid4()), status))


def api_job(args, status):
    identifier = str(uuid.uuid4())
    folder = args.runs / identifier
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"job_id": identifier, "status": status}))
    return folder / "job.json"


def assert_gate_held(args):
    with args.inference_gate.open("rb") as stream:
        with pytest.raises(BlockingIOError):
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)


def assert_admission_blocked(args):
    with sqlite3.connect(args.portal_db, timeout=.01) as connection:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            connection.execute("BEGIN IMMEDIATE")


class Services:
    def __init__(self, args):
        self.args = args
        self.active = {controller.API_UNIT: True, controller.PORTAL_UNIT: True}
        self.actions = []
        self.on_action = None
        self.wrong_new_fingerprint = False
        self.portal_ok = True

    def service(self, action, unit):
        if action == "is-active":
            return self.active[unit]
        assert unit in {controller.API_UNIT, controller.PORTAL_UNIT}
        assert_gate_held(self.args)
        if action == "stop":
            assert unit == controller.PORTAL_UNIT
            assert_admission_blocked(self.args)
        self.actions.append((action, unit))
        self.active[unit] = action != "stop"
        if self.on_action:
            self.on_action(action, unit)

    def health(self):
        new = self.args.current.resolve() == self.args.release
        fingerprint = "c" * 64 if new and self.wrong_new_fingerprint else ("b" if new else "a") * 64
        return {"status": "ok", "worker_count": 1, "ffmpeg": True, "model_configured": True,
                "model_backend": "local", "document_reader": "native_image",
                "native_implementation": {"fingerprint_sha256": fingerprint}}

    def run(self):
        return controller.run(self.args, service=self.service, health=self.health,
                              portal_check=lambda: self.portal_ok, sleep=lambda seconds: None)


@pytest.mark.parametrize("status", ["uploading", "validating", "submitting", "submission_uncertain", "queued", "processing"])
def test_portal_activity_never_stops_services_or_changes_release(tmp_path, status):
    args, old, _ = case(tmp_path)
    portal_job(args, status)
    services = Services(args)
    result = services.run()
    assert result["status"] == "deferred" and not result["snapshot"]["idle"]
    assert services.actions == [] and args.current.resolve() == old


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_api_activity_never_stops_services_or_changes_artifacts(tmp_path, status):
    args, old, _ = case(tmp_path)
    path = api_job(args, status)
    original = path.read_bytes()
    services = Services(args)
    assert services.run()["status"] == "deferred"
    assert services.actions == [] and args.current.resolve() == old and path.read_bytes() == original


def test_dry_run_is_read_only_even_when_idle(tmp_path):
    args, old, _ = case(tmp_path)
    args.dry_run = True
    services = Services(args)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert services.run()["status"] == "ready"
    assert services.actions == [] and args.current.resolve() == old
    assert all(path.read_bytes() == value for path, value in before.items())


def test_admission_between_initial_snapshot_and_sqlite_lock_is_caught(tmp_path, monkeypatch):
    args, old, _ = case(tmp_path)
    original = controller.admission_block

    @contextmanager
    def raced(path):
        portal_job(args, "uploading")
        with original(path) as connection:
            yield connection

    monkeypatch.setattr(controller, "admission_block", raced)
    services = Services(args)
    assert services.run()["status"] == "deferred"
    assert not services.actions and args.current.resolve() == old


def test_api_admission_before_exclusive_gate_is_caught_before_portal_stop(tmp_path, monkeypatch):
    args, old, _ = case(tmp_path)
    original = controller.exclusive_gate

    @contextmanager
    def raced(path):
        api_job(args, "queued")
        with original(path):
            yield

    monkeypatch.setattr(controller, "exclusive_gate", raced)
    services = Services(args)
    assert services.run()["status"] == "deferred"
    assert not services.actions and args.current.resolve() == old


def test_shared_inference_gate_defers_without_stopping_portal(tmp_path):
    args, old, _ = case(tmp_path)
    services = Services(args)
    with args.inference_gate.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        result = services.run()
    assert result["status"] == "deferred" and result["error"] == "inference_gate_busy"
    assert not services.actions and args.current.resolve() == old


def test_success_holds_both_guards_before_stop_and_restarts_api_only(tmp_path):
    args, _, new = case(tmp_path)
    path = api_job(args, "completed")
    original = path.read_bytes()
    services = Services(args)
    result = services.run()
    assert result["status"] == "switched" and result["fingerprint_sha256"] == "b" * 64
    assert args.current.resolve() == new and path.read_bytes() == original
    assert services.actions == [("stop", controller.PORTAL_UNIT), ("restart", controller.API_UNIT),
                                ("start", controller.PORTAL_UNIT)]
    with args.inference_gate.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with sqlite3.connect(args.portal_db, timeout=.01) as connection:
        connection.execute("BEGIN IMMEDIATE")


def test_new_fingerprint_failure_rolls_back_and_restores_health(tmp_path):
    args, old, _ = case(tmp_path)
    services = Services(args)
    services.wrong_new_fingerprint = True
    result = services.run()
    assert result["status"] == "rolled_back" and result["error"] == "api_fingerprint_mismatch"
    assert args.current.resolve() == old and all(services.active.values())
    assert services.actions == [("stop", controller.PORTAL_UNIT), ("restart", controller.API_UNIT),
                                ("stop", controller.PORTAL_UNIT), ("restart", controller.API_UNIT),
                                ("start", controller.PORTAL_UNIT)]


@pytest.mark.parametrize("failure", ["api_restart", "portal_start", "pointer_after_replace", "interrupt"])
def test_switch_restart_and_portal_failures_all_enter_rollback(tmp_path, monkeypatch, failure):
    args, old, _ = case(tmp_path)
    services = Services(args)
    failed = []

    def on_action(action, unit):
        matches = ((failure == "api_restart" and action == "restart" and unit == controller.API_UNIT)
                   or (failure == "portal_start" and action == "start" and unit == controller.PORTAL_UNIT)
                   or (failure == "interrupt" and action == "restart" and unit == controller.API_UNIT))
        if matches and not failed:
            failed.append(True)
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise controller.SwitchFailure("service_operation_failed")

    services.on_action = on_action
    if failure == "pointer_after_replace":
        original = controller.point_current

        def point(current, release):
            original(current, release)
            if not failed:
                failed.append(True)
                raise OSError("private fixture content must not be printed")

        monkeypatch.setattr(controller, "point_current", point)
    result = services.run()
    assert result["status"] == "rolled_back" and failed
    assert args.current.resolve() == old and all(services.active.values())
    assert "private fixture" not in json.dumps(result)


def test_changed_current_since_preflight_defers_before_stopping_services(tmp_path, monkeypatch):
    args, old, new = case(tmp_path)
    original = controller.exclusive_gate

    @contextmanager
    def raced(path):
        controller.point_current(args.current, new)
        with original(path):
            yield

    monkeypatch.setattr(controller, "exclusive_gate", raced)
    services = Services(args)
    result = services.run()
    assert result["status"] == "deferred" and result["error"] == "current_release_changed"
    assert not services.actions and args.current.resolve() == new


def test_upload_after_portal_start_blocks_rollback_without_interrupting_it(tmp_path):
    args, _, new = case(tmp_path)
    services = Services(args)

    def on_action(action, unit):
        if action == "start" and unit == controller.PORTAL_UNIT:
            portal_job(args, "uploading")
            services.portal_ok = False

    services.on_action = on_action
    result = services.run()
    assert result["status"] == "recovery_blocked" and result["recovery_error"] == "customer_work_active"
    assert args.current.resolve() == new and all(services.active.values())
    assert services.actions.count(("stop", controller.PORTAL_UNIT)) == 1
    with sqlite3.connect(args.portal_db) as connection:
        assert connection.execute("SELECT status FROM jobs").fetchone() == ("uploading",)


@pytest.mark.parametrize("failure", ["missing_state", "corrupt_state", "unknown_status", "symlink_state", "missing_gate"])
def test_uncertain_state_and_missing_common_gate_fail_closed(tmp_path, failure):
    args, old, _ = case(tmp_path)
    path = api_job(args, "completed")
    if failure == "missing_state":
        path.unlink()
    elif failure == "corrupt_state":
        path.write_text("private invalid JSON")
    elif failure == "unknown_status":
        saved = json.loads(path.read_text())
        path.write_text(json.dumps({**saved, "status": "unknown"}))
    elif failure == "symlink_state":
        destination = tmp_path / "outside.json"
        path.replace(destination)
        path.symlink_to(destination)
    else:
        args.inference_gate.unlink()
    services = Services(args)
    with pytest.raises(controller.SwitchFailure):
        services.run()
    assert not services.actions and args.current.resolve() == old


def test_cli_exposes_explicit_release_dry_run_and_fingerprint(tmp_path):
    args = controller.arguments(["--release", str(tmp_path / "new"), "--dry-run",
                                 "--expected-fingerprint", "a" * 64])
    assert args.release == tmp_path / "new" and args.dry_run
    assert args.expected_fingerprint == "a" * 64
