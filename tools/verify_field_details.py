"""Bounded source-pixel verification probe; never changes an existing job."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from sever.cli import agent_config
from sever.field_review import checked_detail_field
from sever.field_verification import verify_field_views
from sever.inference_gate import gate_path, inference_access
from sever.pipeline import save_json
from sever.verification_targets import frames_for_verification


async def probe(args):
    run, out = args.run.resolve(), args.out.resolve()
    if out.exists() or out == run or run in out.parents:
        raise ValueError("Use a new separate validation directory")
    result_path = run / "results.json"
    if not result_path.is_file():
        result_path = run / "partial-results.json"
    result = json.loads(result_path.read_text())
    source_id = (result.get("source") or {}).get("sha256") or result.get("source_id")
    pages = [page for page in result["pages"] if page["page_id"] in args.page]
    if len(pages) != len(set(args.page)):
        raise ValueError("Every requested page must exist in the saved snapshot")
    frames = frames_for_verification(pages, args.field, limit=args.frames)
    if len(frames) < 2:
        raise ValueError("Two independent source views must cover the requested field")
    kinds = {page["result"]["document_type"] for page in pages}
    if len(kinds) != 1 or not source_id:
        raise ValueError("Source and own document form must be pinned")
    out.mkdir(parents=True)
    settings = agent_config(os.environ["HARNESS_MODEL_NAME"], args.requests, args.tokens)
    settings.max_output_tokens = min(settings.max_output_tokens, 3000)
    settings.timeout_seconds = min(settings.timeout_seconds, 120)
    started = time.monotonic()
    answer = await verify_field_views(frames, source_id=source_id, page_id="detail-probe",
        document_type=next(iter(kinds)), field_name=args.field, config=settings,
        cache_dir=out / "image-reading-evidence", trace_path=out / "trace.jsonl")
    field = checked_detail_field(answer, args.field, frames)
    output = {"source_id": source_id, "pages": args.page, "field_name": args.field,
        "field": field.model_dump(mode="json"), "elapsed_seconds": round(time.monotonic()-started, 2),
        "usage": settings.usage.model_dump(),
        "answer": {**answer, "fields": {args.field: field.model_dump(mode="json")}}}
    save_json(out / "verification.json", output)
    print(json.dumps({key: output[key] for key in ["field", "elapsed_seconds", "usage"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--page", action="append", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--frames", type=int, choices=[2, 3], default=2)
    parser.add_argument("--requests", type=int, default=9)
    parser.add_argument("--tokens", type=int, default=250000)
    args = parser.parse_args()
    with inference_access(gate_path(run_dir=args.run.parent)):
        asyncio.run(probe(args))


if __name__ == "__main__":
    main()
