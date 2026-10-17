"""Append trace boundaries safely after interrupted writes; preserve raw history."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4


def append_trace_row(path: Path, row: dict) -> None:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        end = stream.seek(0, 2)
        if end:
            stream.seek(end - 1); terminated = stream.read(1) == b"\n"
            cursor = end - int(terminated)
            tail = b""
            while cursor:
                beginning = max(0, cursor - 65536)
                stream.seek(beginning)
                tail = stream.read(cursor - beginning) + tail
                split = tail.rfind(b"\n")
                if split >= 0:
                    offset = beginning + split + 1
                    tail = tail[split + 1:]
                    break
                cursor = beginning
            else:
                offset = 0
            valid = not tail.strip()
            if tail.strip():
                try:
                    valid = isinstance(json.loads(tail), dict)
                except (ValueError, UnicodeDecodeError):
                    valid = False
            if not valid:
                raw = tail + (b"\n" if terminated else b"")
                archive = path.with_name(f"{path.name}.interrupted-{offset}-{uuid4().hex[:12]}.bin")
                # Archive BEFORE changing the trace; a failed preservation write
                # leaves the original trace bytes untouched for a later retry.
                with archive.open("xb") as backup:
                    backup.write(raw);backup.flush()
                stream.truncate(offset);stream.seek(0, 2)
                recovered = {"time": time.time(), "event": "trace_interrupted_tail_preserved",
                             "raw_fragment_path": str(archive.resolve()), "offset_bytes": offset,
                             "fragment_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                stream.write(json.dumps(recovered, separators=(",", ":")).encode() + b"\n")
            elif not terminated:
                stream.write(b"\n")
        stream.write(payload)
        stream.flush()


def archive_restart_artifacts(out: Path, names: tuple[str, ...], *, reason="restart") -> Path | None:
    """Rename small generated artifacts, leaving source images/caches untouched."""
    out = Path(out)
    existing = [out / name for name in names if (out / name).is_file()]
    if not existing:
        return None
    history = out / "history" / f"{reason}-{time.time_ns()}-{uuid4().hex[:8]}"
    history.mkdir(parents=True)
    for path in existing:
        path.rename(history / path.name)
    return history
