"""Fresh local joint form/field review from a strict source-frame manifest, without labels."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sever.agents import AgentConfig, BatchBudgetExceeded
from sever.inference_gate import inference_access
from sever.pipeline import save_json, source_hash, trace
from sever.schema import FIELDS_BY_TYPE, FieldValue, Evidence
from tools.read_reference_images import local_config


class SourceFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    frame_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rotation_clockwise_degrees: int
    timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    source_pts: int | None = None
    source_time_base: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def actual_frame(self):
        if self.rotation_clockwise_degrees not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270")
        if not self.frame_id.strip() or not Path(self.path).is_absolute():
            raise ValueError("Frames require nonempty IDs and absolute source paths")
        return self


class ReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    document_type: Literal["upd", "torg12", "ttn", "trn", "unknown"]
    fields: list[str] = Field(max_length=6)
    frames: list[SourceFrame] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def requested_contract(self):
        if (not self.source_id.strip() or not self.page_id.strip()
                or (self.document_type == "unknown") != (len(self.fields) == 0)
                or len(set(self.fields)) != len(self.fields)
                or any(name not in FIELDS_BY_TYPE[self.document_type] for name in self.fields)):
            raise ValueError("Manifest requires unique requested fields of the supplied form")
        return self


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("HARNESS_MODEL_NAME", ""))
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if not args.model.strip() or args.max_requests < 1 or args.max_tokens < 1000:
        parser.error("Explicit local alias, positive requests and max-tokens >= 1000 are required")
    return args


def _bound_answer(answer, manifest):
    fields = {name: FieldValue.model_validate(value) for name, value in answer["fields"].items()}
    if set(fields) != set(manifest.fields):
        raise ValueError("Joint answer changed the requested field contract")
    known = {frame.frame_id for frame in manifest.frames}
    if manifest.document_type == "unknown":
        kind = answer.get("document_type")
        evidence = [Evidence.model_validate(value) for value in answer.get("classification_evidence", [])]
        cited = {item.frame_id for item in evidence}
        if (kind not in FIELDS_BY_TYPE or not cited <= known
                or (kind != "unknown" and len(cited) < 2)):
            raise ValueError("Joint form answer lacks actual source-bound type evidence")
    if any(e.frame_id not in known for field in fields.values() for e in field.evidence):
        raise ValueError("Joint answer cites a frame outside the manifest")
    for field in fields.values():
        if field.status in {"recognized", "not_present"} and len({e.frame_id for e in field.evidence}) < 2:
            raise ValueError("Resolved joint field requires two actual source-frame citations")
    proofs = answer["recognition"]["pixel_proof"]["sources"]
    expected = {f"frame_{index:02d}": frame for index, frame in enumerate(manifest.frames, 1)}
    if set(proofs) != set(expected):
        raise ValueError("Joint proof does not bind every supplied frame")
    for alias, frame in expected.items():
        proof = proofs[alias]
        if (proof["frame_id"] != frame.frame_id or proof["source_sha256"] != frame.source_sha256
                or Path(proof["source_path"]).resolve() != Path(frame.path).resolve()
                or proof["timestamp_s"] != frame.timestamp_s
                or proof["rotation_clockwise_degrees"] != frame.rotation_clockwise_degrees):
            raise ValueError("Joint proof belongs to different source pixels or timestamps")
    return {**answer, "fields": {name: value.model_dump(mode="json") for name, value in fields.items()}}


async def run(args, *, config: AgentConfig | None = None, reader=None):
    """Hold the shared gate through reading and export; injected reader is offline-only."""
    with inference_access():
        return await _run(args, config=config, reader=reader)


async def _run(args, *, config: AgentConfig | None = None, reader=None):
    config = config or local_config(args)
    if config.provider_kind != "local":
        raise ValueError("Conflict-image probe permits only an explicit local loopback provider")
    config = replace(config, model_name=args.model, vision_model_name=args.model,
        document_reader="native_image", refresh_cache=True, verify_inn=False,
        max_requests=args.max_requests, max_tokens=args.max_tokens)
    input_path, out = args.input.expanduser().resolve(), args.out.expanduser().resolve()
    input_bytes = input_path.read_bytes()
    manifest = ReviewManifest.model_validate_json(input_bytes)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new or empty output directory for a fresh joint review")
    for frame in manifest.frames:
        if source_hash(Path(frame.path)) != frame.source_sha256:
            raise ValueError("Manifest source SHA does not match actual source pixels")
    out.mkdir(parents=True, exist_ok=True)
    if reader is None:
        from sever.image_conflicts import read_leaf_conflicts
        reader = read_leaf_conflicts
    started, before = time.monotonic(), config.usage.model_dump()
    trace_path = out / "trace.jsonl"
    result = {"schema_version": 1, "run_kind": "fresh_joint_source_form_review" if manifest.document_type == "unknown" else "fresh_joint_source_field_review", "status": "processing",
        "input_manifest": manifest.model_dump(mode="json", exclude_none=True),
        "manifest_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "configuration": {"model": config.model_name, "provider_kind": "local", "model_base_url": config.base_url,
            "max_requests": config.max_requests, "max_tokens": config.max_tokens, "refresh_cache": True,
            "ground_truth_read": False, "previous_answers_read": False},
        "acceptance": {"accuracy_evaluated": False, "customer_approved": False,
            "independent_model_corroboration": False},
        "limitations": ["A fresh joint model review is not measured accuracy or independent model agreement.",
            "Only actual source pixels and requested field names are model inputs."]}

    def usage():
        after = config.usage.model_dump()
        result["usage"] = {key: value - before.get(key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            for key, value in after.items()}
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)

    stage = "initialization"
    try:
        save_json(out / "input-provenance.json", result["input_manifest"])
        save_json(out / "partial-results.json", result)
        trace(trace_path, "conflict_probe_started", frame_count=len(manifest.frames),
              requested_fields=manifest.fields, ground_truth_read=False, previous_answers_read=False)
        stage = "joint_reading"
        answer = await reader([frame.model_dump(exclude_none=True) for frame in manifest.frames],
            source_id=manifest.source_id, page_id=manifest.page_id, document_type=manifest.document_type,
            unresolved_fields=manifest.fields, config=config, cache_dir=out / "image-reading-evidence", trace_path=trace_path)
        result["answer"] = _bound_answer(answer, manifest)
        for frame in manifest.frames:
            if source_hash(Path(frame.path)) != frame.source_sha256:
                raise ValueError("A source frame changed while its pixels were being reviewed")
        usage()
        result["status"] = "completed"
        stage = "finalization"
        save_json(out / "partial-results.json", result)
        trace(trace_path, "conflict_probe_completed", adjudication=answer["adjudication"]["status"], usage=result["usage"])
        save_json(out / "results.json", result)
        return result
    except BaseException as exc:
        result["status"] = "budget_exhausted" if isinstance(exc, BatchBudgetExceeded) else "failed"
        result["failure"] = {"error_type": type(exc).__name__, "stage": stage}
        usage()
        # Best-effort failure exports never replace the original fatal/budget error.
        try:
            (out / "results.json").unlink(missing_ok=True)
        except OSError:
            pass
        try:
            save_json(out / "partial-results.json", result)
        except Exception:
            pass
        try:
            trace(trace_path, "conflict_probe_failed", error_type=type(exc).__name__, stage=stage, usage=result["usage"])
        except Exception:
            pass
        raise


def main():
    args = arguments()
    result = asyncio.run(run(args))
    print(json.dumps({"status": result["status"], "results": str(args.out.resolve() / "results.json"),
        "adjudication": result["answer"]["adjudication"]["status"], "usage": result["usage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
