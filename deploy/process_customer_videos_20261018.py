"""Put the customer's original videos into the cabinet as ordinary jobs, one at a time.

Uses the portal's own reservation, validation and submission steps (sever.portal), with the
video copied from /srv/sever/input instead of uploaded. A video is submitted only when no
API job is queued or processing, so testers' own uploads never wait behind the whole batch.
Run as the sever user with the production venv:
  PORTAL_DATA_DIR=/srv/sever/customer SEVER_RUNS_DIR=/srv/sever/runs SEVER_API_URL=http://127.0.0.1:8080 \
    /opt/sever-tsl/current/.venv/bin/python process_customer_videos_20261018.py --owner sever-test NAME...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from sever import portal

INPUT = Path("/srv/sever/input")


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()), message, flush=True)


def api_jobs_active() -> int:
    return sum(json.loads(path.read_text()).get("status") in {"queued", "processing"}
               for path in portal.RUNS.glob("*/job.json") if path.is_file())


async def submit(owner: str, name: str) -> str:
    identifier, source = portal.reserve(owner, name)
    try:
        # The originals are owned by root, so a hard link is refused; a byte copy is the upload.
        shutil.copyfile(INPUT / name, source)
        size = source.stat().st_size
        portal.update(identifier, status="validating", progress="Проверяем видео", size_bytes=size)
        await portal.probe_video(source)
        health = await portal.backend("GET", "/health")
        health.raise_for_status()
        config = health.json()
        payload = {"source_path": str(source.resolve()), "model": config["model_name"], "max_requests": 4000,
                   "max_tokens": 40000000, "max_pages": 300, "sample_interval": .5, "refresh_cache": True}
        fingerprint = config.get("native_implementation", {}).get("fingerprint_sha256")
        if fingerprint:
            payload["native_implementation_fingerprint"] = fingerprint
        portal.update(identifier, status="submitting", progress="Передаём видео на обработку")
        response = await portal.backend("POST", "/v1/jobs", json=payload)
        response.raise_for_status()
        job_id = response.json()["job_id"]
        portal.update(identifier, job_id=job_id, status="queued", progress="Видео в очереди", error=None)
        return identifier
    except BaseException as exc:
        portal.update(identifier, status="failed", progress="Видео не запущено", error=f"Пакетная загрузка не удалась: {type(exc).__name__}")
        raise


async def main(owner: str, names: list[str]) -> None:
    with portal.database() as db:
        existing = {row["name"] for row in db.execute("SELECT name FROM jobs WHERE owner=? AND status IN ('completed','completed_with_errors','queued','processing')", (owner,))}
    for name in names:
        if name in existing:
            log(f"skip {name}: already in the cabinet")
            continue
        if not (INPUT / name).is_file():
            log(f"skip {name}: not found in {INPUT}")
            continue
        while api_jobs_active():
            await asyncio.sleep(30)
        identifier = await submit(owner, name)
        log(f"submitted {name} as portal job {identifier}")
        started = time.monotonic()
        while True:
            await asyncio.sleep(30)
            row = await portal.refresh(dict(portal.owned(identifier, owner)))
            if row["status"] in portal.TERMINAL:
                log(f"{name}: {row['status']} after {time.monotonic() - started:.0f} s")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("names", nargs="+")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.owner, arguments.names))
