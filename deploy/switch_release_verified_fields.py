#!/usr/bin/env python3
"""Switch an already staged release only when customer work is idle.

This controller does not stage code, run recognition, restart the model, or
schedule a later switch. Run --dry-run first; a busy server is left untouched.
The release must already contain its own .venv/bin/python and .venv/bin/sever.
Build/install the wheel in that isolated release, never in the current release.

Admission is blocked with the same SQLite write reservation used by
portal.reserve(), before the final idle checks. The persistent inference gate
then excludes API admission and CLI inference through switch and health checks.
Only an idle portal is stopped. An unsuccessful switch restores the old release
inside the gate; if fresh portal activity makes recovery unsafe, recovery is
refused rather than interrupting an upload or accepted job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import uuid


API_UNIT = "sever-tsl.service"
PORTAL_UNIT = "sever-portal.service"
TERMINAL = {"completed", "completed_with_errors", "failed"}
PORTAL_STATUSES = TERMINAL | {"uploading", "validating", "submitting", "submission_uncertain", "queued", "processing"}
API_STATUSES = TERMINAL | {"queued", "processing"}


class SwitchFailure(RuntimeError):
    """The code is safe to print; exception causes may contain private context."""


def systemctl(action, unit):
    if action not in {"is-active", "start", "stop", "restart"} or unit not in {API_UNIT, PORTAL_UNIT}:
        raise SwitchFailure("invalid_service_operation")
    try:
        result = subprocess.run(["systemctl", action, "--", unit], capture_output=True,
                                text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise SwitchFailure("service_operation_failed") from None
    if action == "is-active":
        return result.returncode == 0 and result.stdout.strip() == "active"
    if result.returncode:
        raise SwitchFailure("service_operation_failed")


def api_health():
    # Local service checks must never inherit proxy settings or credentials.
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request("http://127.0.0.1:8080/health"), timeout=5) as response:
        if response.status != 200:
            raise SwitchFailure("api_health_failed")
        return json.loads(response.read(2 * 1024 * 1024))


def portal_health():
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request("http://127.0.0.1:8082/"), timeout=5) as response:
            return response.status == 401
    except HTTPError as exc:
        # The unauthenticated loopback request must reach the private portal.
        return exc.code == 401


def checked_health(health, expected_fingerprint=None):
    try:
        value = health()
        if (not isinstance(value, dict) or value.get("status") != "ok"
                or value.get("worker_count") != 1 or value.get("ffmpeg") is not True
                or value.get("model_configured") is not True
                or value.get("model_backend") != "local"
                or value.get("document_reader") != "native_image"):
            raise SwitchFailure("api_health_failed")
        fingerprint = value.get("native_implementation", {}).get("fingerprint_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            raise SwitchFailure("api_fingerprint_unavailable")
        if expected_fingerprint and fingerprint != expected_fingerprint:
            raise SwitchFailure("api_fingerprint_mismatch")
        return fingerprint
    except SwitchFailure:
        raise
    except Exception:
        raise SwitchFailure("api_health_failed") from None


def portal_counts(connection):
    try:
        counts = Counter({row[0]: row[1] for row in connection.execute(
            "SELECT status,COUNT(*) FROM jobs GROUP BY status")})
    except sqlite3.Error:
        raise SwitchFailure("portal_state_unavailable") from None
    if any(status not in PORTAL_STATUSES for status in counts):
        raise SwitchFailure("portal_state_unknown")
    return dict(counts)


def api_counts(runs):
    if not runs.is_dir() or runs.is_symlink():
        raise SwitchFailure("api_state_unavailable")
    counts = Counter()
    try:
        for folder in runs.iterdir():
            try:
                identifier = str(uuid.UUID(folder.name))
            except ValueError:
                continue
            path = folder / "job.json"
            if (identifier != folder.name or folder.is_symlink() or path.is_symlink()
                    or not folder.is_dir() or not path.is_file()):
                raise SwitchFailure("api_state_uncertain")
            saved = json.loads(path.read_bytes())
            if (not isinstance(saved, dict) or saved.get("job_id") != identifier
                    or saved.get("status") not in API_STATUSES):
                raise SwitchFailure("api_state_uncertain")
            counts[saved["status"]] += 1
    except SwitchFailure:
        raise
    except (OSError, ValueError, TypeError):
        raise SwitchFailure("api_state_unavailable") from None
    return dict(counts)


def connect_portal(path, *, admission=False):
    if not path.is_file() or path.is_symlink():
        raise SwitchFailure("portal_state_unavailable")
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + ("?mode=rw" if admission else "?mode=ro"),
                                     uri=True, timeout=2)
        if admission:
            connection.execute("BEGIN IMMEDIATE")
        return connection
    except sqlite3.Error:
        if "connection" in locals():
            connection.close()
        raise SwitchFailure("portal_admission_busy") from None


@contextmanager
def admission_block(path):
    connection = connect_portal(path, admission=True)
    try:
        yield connection
    finally:
        # No SQL changes are made. Keep reserve() blocked until the portal has
        # stopped; a request still waiting to reserve has not begun an upload.
        connection.rollback()
        connection.close()


@contextmanager
def exclusive_gate(path):
    # The running API has already created this inode. Never replace/create a
    # different lock file: every inference participant must use the same one.
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError:
        raise SwitchFailure("inference_gate_unavailable") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SwitchFailure("inference_gate_invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SwitchFailure("inference_gate_busy") from None
        yield
    finally:
        os.close(descriptor)


def snapshot(args, connection=None):
    if connection is None:
        connection = connect_portal(args.portal_db)
        try:
            portal = portal_counts(connection)
        finally:
            connection.close()
    else:
        portal = portal_counts(connection)
    api = api_counts(args.runs)
    return {"portal_status_counts": portal, "api_status_counts": api,
            "idle": not any(status not in TERMINAL and count for counts in (portal, api)
                            for status, count in counts.items())}


def release_paths(args):
    try:
        releases = args.releases.resolve(strict=True)
        release = args.release.resolve(strict=True)
        if (not args.current.is_symlink() or release.parent != releases or not release.is_dir()
                or not (release / ".venv/bin/python").is_file()
                or not (release / ".venv/bin/sever").is_file()):
            raise SwitchFailure("release_layout_invalid")
        old = args.current.resolve(strict=True)
        if old.parent != releases or not old.is_dir():
            raise SwitchFailure("current_release_invalid")
        return release, old
    except OSError:
        raise SwitchFailure("release_layout_invalid") from None


def point_current(current, target):
    temporary = current.with_name(".current-switch-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)


def wait_health(args, health, portal_check, service, sleep, expected, *, portal=False):
    last = SwitchFailure("api_health_failed")
    for attempt in range(args.health_attempts):
        try:
            fingerprint = checked_health(health, expected)
            if not service("is-active", API_UNIT):
                raise SwitchFailure("api_service_inactive")
            if portal and (not service("is-active", PORTAL_UNIT) or not portal_check()):
                raise SwitchFailure("portal_health_failed")
            return fingerprint
        except SwitchFailure as exc:
            last = exc
        except Exception:
            last = SwitchFailure("portal_health_failed" if portal else "api_health_failed")
        if attempt + 1 < args.health_attempts:
            sleep(args.health_interval)
    raise last


def run(args, *, service=systemctl, health=api_health, portal_check=portal_health, sleep=time.sleep):
    release, old = release_paths(args)
    before = snapshot(args)
    result = {"status": "deferred" if not before["idle"] else "ready", "snapshot": before,
              "release": release.name, "previous_release": old.name, "dry_run": args.dry_run}
    if not before["idle"] or args.dry_run:
        return result
    if release == old:
        return {**result, "status": "already_current"}
    if not service("is-active", API_UNIT) or not service("is-active", PORTAL_UNIT):
        raise SwitchFailure("current_service_inactive")
    old_fingerprint = checked_health(health)
    if not portal_check():
        raise SwitchFailure("current_portal_unhealthy")

    portal_stopped = False
    switched = False
    try:
        with admission_block(args.portal_db) as connection:
            before = snapshot(args, connection)
            if not before["idle"]:
                return {**result, "status": "deferred", "snapshot": before}
            with exclusive_gate(args.inference_gate):
                # API jobs admitted immediately before the exclusive lock are
                # visible here, even if their worker has not started yet.
                final = snapshot(args, connection)
                if not final["idle"]:
                    return {**result, "status": "deferred", "snapshot": final}
                if args.current.resolve(strict=True) != old:
                    return {**result, "status": "deferred", "error": "current_release_changed"}
                portal_stopped = True
                service("stop", PORTAL_UNIT)
                # Keep the inference lock after releasing the DB reservation.
                # The portal must be fully stopped before this transaction ends.
                if service("is-active", PORTAL_UNIT):
                    raise SwitchFailure("portal_stop_failed")
                connection.rollback()
                try:
                    # Rollback also covers a failure immediately after replacing
                    # current; do not set this flag only after a successful call.
                    switched = True
                    point_current(args.current, release)
                    service("restart", API_UNIT)
                    fingerprint = wait_health(args, health, portal_check, service, sleep,
                                              args.expected_fingerprint)
                    service("start", PORTAL_UNIT)
                    portal_stopped = False
                    fingerprint = wait_health(args, health, portal_check, service, sleep,
                                              args.expected_fingerprint, portal=True)
                    return {**result, "status": "switched", "snapshot": final,
                            "fingerprint_sha256": fingerprint}
                except BaseException as exc:
                    code = (str(exc) if isinstance(exc, SwitchFailure) else
                            "switch_interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "switch_failed")
                    try:
                        # A portal that started just before a health failure can
                        # have a new upload. Never stop it until this idle check
                        # and the SQLite admission reservation both succeed.
                        with admission_block(args.portal_db) as recovery:
                            recovery_state = snapshot(args, recovery)
                            if not recovery_state["idle"]:
                                return {**result, "status": "recovery_blocked", "error": code,
                                        "recovery_error": "customer_work_active",
                                        "snapshot": recovery_state}
                            portal_stopped = True
                            service("stop", PORTAL_UNIT)
                            recovery.rollback()
                            point_current(args.current, old)
                            service("restart", API_UNIT)
                            wait_health(args, health, portal_check, service, sleep, old_fingerprint)
                            service("start", PORTAL_UNIT)
                            portal_stopped = False
                            wait_health(args, health, portal_check, service, sleep, old_fingerprint, portal=True)
                        return {**result, "status": "rolled_back", "error": code}
                    except BaseException as recovery_exc:
                        recovery_code = (str(recovery_exc) if isinstance(recovery_exc, SwitchFailure)
                                         else "recovery_failed")
                        return {**result, "status": "recovery_failed", "error": code,
                                "recovery_error": recovery_code}
    except BaseException as exc:
        failure = exc if isinstance(exc, SwitchFailure) else SwitchFailure(
            "switch_interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "switch_failed")
        # No release was replaced on these paths. Restore portal availability
        # after a stop command that failed partway; the original API stays live.
        if portal_stopped and not switched:
            try:
                service("start", PORTAL_UNIT)
                portal_stopped = False
            except BaseException:
                return {**result, "status": "recovery_failed", "error": str(failure),
                        "recovery_error": "portal_restore_failed"}
        if not switched and str(failure) in {"inference_gate_busy", "portal_admission_busy"}:
            return {**result, "status": "deferred", "error": str(failure)}
        raise failure from None


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--expected-fingerprint", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--current", type=Path, default=Path("/opt/sever-tsl/current"))
    parser.add_argument("--releases", type=Path, default=Path("/opt/sever-tsl/releases"))
    parser.add_argument("--runs", type=Path, default=Path("/srv/sever/runs"))
    parser.add_argument("--portal-db", type=Path, default=Path("/srv/sever/customer/portal.sqlite3"))
    parser.add_argument("--inference-gate", type=Path,
                        default=Path("/srv/sever/runs/.inference-evaluation.lock"))
    parser.add_argument("--health-attempts", type=int, default=12)
    parser.add_argument("--health-interval", type=float, default=2)
    args = parser.parse_args(argv)
    if args.expected_fingerprint and not re.fullmatch(r"[a-f0-9]{64}", args.expected_fingerprint):
        parser.error("--expected-fingerprint must be a lowercase SHA-256")
    if not 1 <= args.health_attempts <= 60 or not 0 < args.health_interval <= 10:
        parser.error("health attempts must be 1..60 and interval must be 0..10 seconds")
    if not args.dry_run and os.geteuid() != 0:
        parser.error("a real switch must run as root")
    return args


def main(argv=None):
    # Ordinary termination must unwind the same guarded recovery as a failed
    # health check. SIGKILL/power loss cannot be recovered by any user process.
    def interrupted(signum, frame):
        raise SwitchFailure("switch_interrupted")

    handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGHUP)}
    try:
        try:
            result = run(arguments(argv))
        except SwitchFailure as exc:
            result = {"status": "blocked", "error": str(exc)}
        except KeyboardInterrupt:
            result = {"status": "blocked", "error": "switch_interrupted"}
        except Exception:
            result = {"status": "blocked", "error": "preflight_failed"}
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (0 if result["status"] in {"ready", "switched", "already_current"}
            else 3 if result["status"] == "deferred" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
