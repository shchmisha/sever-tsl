"""One process-shared gate between recognition and private evaluation.

Inference admission and execution take shared locks. Evaluation takes an
exclusive lock before checking that jobs are idle, and holds it while scoring.
Every participant must use the same persistent file; never replace that file.
This module imports no recognition, model or provider code.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import stat


GATE_PROTOCOL = "flock-shared-inference-exclusive-evaluation-v1"
GATE_FILENAME = ".inference-evaluation.lock"


class InferenceGateBusy(RuntimeError):
    """An incompatible inference/evaluation operation owns the gate."""


def gate_path(path: str | Path | None = None, *, run_dir: str | Path | None = None) -> Path:
    """Resolve the explicit/common gate without opening any input artifacts."""
    selected = path if path is not None else os.getenv("SEVER_INFERENCE_GATE_PATH")
    if selected is None:
        root = run_dir if run_dir is not None else os.getenv("SEVER_RUNS_DIR", "runs")
        selected = Path(root) / GATE_FILENAME
    if not str(selected).strip():
        raise ValueError("Inference gate path must be nonempty")
    return Path(selected).expanduser().resolve()


@contextmanager
def _access(path: str | Path | None, *, exclusive: bool, blocking: bool = False):
    selected = gate_path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(selected, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Inference gate must be a persistent regular file")
        try:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(descriptor, operation if blocking else operation | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EAGAIN, errno.EACCES}:
                raise
            raise InferenceGateBusy("Inference/evaluation gate is held by incompatible activity") from None
        yield selected
    finally:
        # Closing releases this open-file-description's lock, including on
        # errors. Do not unlink: other processes must lock the same inode.
        os.close(descriptor)


def inference_access(path: str | Path | None = None, *, blocking: bool = False):
    """Shared lock held through admission or execution.

    Admission defaults to immediate refusal during evaluation. An already
    accepted worker may set blocking=True to wait for the evaluator's idle
    recheck to release the gate, without failing an otherwise valid queued job.
    """
    return _access(path, exclusive=False, blocking=blocking)


def evaluation_access(path: str | Path | None = None):
    """Nonblocking exclusive lock, held through idle recheck and evaluation."""
    return _access(path, exclusive=True)
