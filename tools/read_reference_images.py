"""Blind local VLM reading of still images, without labels or video consensus.

Each real input image receives a fresh orientation and field reading. The outputs are single-image proposals,
not corroborated video results or measured accuracy. No expected values, OCR,
separate classifier, registry lookup or external model fallback are consulted.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sever.agents import AgentConfig
from sever.cli import agent_config
from sever.pipeline import save_json, trace
from sever.schema import DocumentResult


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("HARNESS_MODEL_NAME", ""),
                        help="Explicit local model alias; defaults to HARNESS_MODEL_NAME")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("Provide --model or HARNESS_MODEL_NAME for the local model")
    if args.max_requests < 1 or args.max_tokens < 1000:
        parser.error("Positive request allowance and max-tokens >= 1000 are required")
    return args


def local_config(args) -> AgentConfig:
    if os.getenv("HARNESS_PROVIDER_KIND", "").strip() != "local":
        raise ValueError("Blind image reading requires HARNESS_PROVIDER_KIND=local; external fallback is disabled")
    config = agent_config(args.model, args.max_requests, args.max_tokens)
    return replace(config, vision_model_name=args.model, document_reader="native_image",
                   refresh_cache=True, allow_tools=False, verify_inn=False)


def image_sources(directory: Path) -> list[dict]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted((path for path in directory.iterdir()
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES),
                   key=lambda path: path.name.casefold())
    if not paths:
        raise ValueError("The input directory contains no supported still images")
    sources = []
    for path in paths:
        source = path.resolve()
        if source.parent != directory:
            raise ValueError("Input images must remain inside the selected input directory")
        with Image.open(source) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Each input must contain exactly one still image")
            image.load()
            size = list(image.size)
        sources.append({"path": str(source), "sha256": source_hash(source),
                        "bytes": source.stat().st_size, "pixel_size": size})
    return sources


def _bound_document(answer: dict, *, source: dict, frame_id: str, page_id: str) -> DocumentResult:
    document = answer["document"]
    if isinstance(document, dict):
        document = DocumentResult.model_validate(document)
    if document.source_id != source["sha256"] or document.page_id != page_id:
        raise ValueError("The image reading belongs to a different source or image")
    citations = [*document.classification_evidence,
                 *(e for field in document.fields.values() for e in field.evidence)]
    if any(e.frame_id != frame_id for e in citations):
        raise ValueError("The image reading cites a different source frame")
    proof = answer["recognition"]["pixel_proof"]
    if (proof["frame_id"] != frame_id or proof["source_sha256"] != source["sha256"]
            or Path(proof["source_path"]).resolve() != Path(source["path"])):
        raise ValueError("The supplied pixel proof belongs to a different source image")
    return document


def _summary(images: list[dict]) -> dict:
    fields = [field for image in images for field in image["result"]["fields"].values()]
    return {"images_read": len(images),
            "types": dict(Counter(image["result"]["document_type"] for image in images)),
            "field_proposals": len(fields),
            "recognized_single_image_proposals": sum(field["status"] == "recognized" for field in fields),
            "field_statuses": dict(Counter(field["status"] for field in fields)),
            "images_with_validation_errors": sum(bool(image["result"]["errors"]) for image in images),
            "errors": [{"page_id": image["page_id"], "errors": image["result"]["errors"]}
                       for image in images if image["result"]["errors"]],
            "independent_video_corroboration": False, "accuracy_evaluated": False}


async def run(args, *, config: AgentConfig | None = None, reader=None) -> dict:
    from sever.inference_gate import inference_access
    with inference_access():
        return await _run(args, config=config, reader=reader)


async def _run(args, *, config: AgentConfig | None = None, reader=None) -> dict:
    """Run direct image readings; an injected reader permits offline verification."""
    config = config or local_config(args)
    if config.provider_kind != "local":
        raise ValueError("Blind image reading permits only an explicit local loopback provider")
    # AgentConfig validates the endpoint. Keep the caller's shared usage object.
    config = replace(config, model_name=args.model, vision_model_name=args.model,
                     document_reader="native_image", refresh_cache=True,
                     allow_tools=False, verify_inn=False,
                     max_requests=args.max_requests, max_tokens=args.max_tokens)
    source_dir, out = args.input_dir.expanduser().resolve(), args.out.expanduser().resolve()
    sources = image_sources(source_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory for a fresh blind reading")
    out.mkdir(parents=True, exist_ok=True)
    if reader is None:
        from sever.zoom_image_reader import read_frame_zoom
        reader = read_frame_zoom
    started = time.monotonic()
    before = config.usage.model_dump()
    trace_path = out / "trace.jsonl"
    result = {"schema_version": 1, "run_kind": "blind_single_image_native_reading",
        "status": "processing", "sources": sources, "images": [],
        "configuration": {"model": config.vision_model_name, "provider_kind": config.provider_kind,
            "model_base_url": config.base_url, "document_reader": "native_image",
            "refresh_cache": True, "max_requests": config.max_requests, "max_tokens": config.max_tokens,
            "output_max_tokens": config.max_output_tokens, "local_ocr_used": False,
            "separate_classifier_used": False, "ground_truth_read": False},
        "unit": "one actual still image; single-image proposal without video-frame corroboration",
        "acceptance": {"customer_approved": False, "accuracy_evaluated": False,
                       "video_detection_evaluated": False, "independent_video_corroboration": False},
        "limitations": ["Recognized fields are uncorroborated single-image proposals, not accepted video results.",
            "No ground truth, expected values or source-filename hints are supplied to the model.",
            "No accuracy or video-detection rate is calculated.",
            "Boxes delimit actual supplied image panels, not measured word positions."]}
    trace(trace_path, "blind_reference_run_started", input_count=len(sources),
          provider_kind="local", document_reader="native_image", refresh_cache=True,
          ground_truth_read=False, separate_classifier_used=False, local_ocr_used=False)
    save_json(out / "input-provenance.json", sources)
    current = None

    def record_usage():
        after = config.usage.model_dump()
        result["usage"] = {key: value - before.get(key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            for key, value in after.items()}
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)

    def record_failure(exc, stage):
        # Final files are success markers. This fresh output directory has no
        # earlier results to preserve, so a failed export must withdraw them.
        for name in ("results.json", "summary.json"):
            (out / name).unlink(missing_ok=True)
        result["status"] = "failed"
        result["failure"] = {**(current or {}), "error_type": type(exc).__name__, "stage": stage,
                             "reason": "The blind image run did not complete all readings and exports"}
        result["summary"] = _summary(result["images"])
        record_usage()
        save_json(out / "partial-results.json", result)
        trace(trace_path, "blind_reference_run_failed", error_type=type(exc).__name__,
              stage=stage, completed_images=len(result["images"]), usage=result["usage"])

    try:
        for number, source in enumerate(sources, 1):
            page_id, frame_id = f"image-{number:04d}", "image-sha256-" + source["sha256"]
            current = {"source": source, "page_id": page_id, "frame_id": frame_id}
            if source_hash(Path(source["path"])) != source["sha256"]:
                raise ValueError("An input image changed after source inventory")
            frame = {"frame_id": frame_id, "path": source["path"], "pixel_size": source["pixel_size"],
                     "source_kind": "actual_still_image"}
            answer = await reader(frame, source_id=source["sha256"], page_id=page_id,
                                  config=config, cache_dir=out / "image-reading-evidence", trace_path=trace_path)
            document = _bound_document(answer, source=source, frame_id=frame_id, page_id=page_id)
            if source_hash(Path(source["path"])) != source["sha256"]:
                raise ValueError("An input image changed while its pixels were being read")
            image = {**current, "result": document.model_dump(mode="json"),
                     "recognition": answer["recognition"], "panels": answer.get("panels", {}),
                     "single_image_proposal": True, "independent_video_corroboration": False}
            result["images"].append(image)
            result["summary"] = _summary(result["images"])
            save_json(out / "partial-results.json", result)
            trace(trace_path, "blind_reference_image_read", page_id=page_id, frame_id=frame_id,
                  source_sha256=source["sha256"], document_type=document.document_type,
                  validation_errors=document.errors, single_image_proposal=True)
    except BaseException as exc:
        record_failure(exc, "image_reading")
        raise
    record_usage()
    result["summary"] = _summary(result["images"])
    result["status"] = "completed_with_errors" if result["summary"]["images_with_validation_errors"] else "completed"
    try:
        save_json(out / "summary.json", {"status": result["status"], **result["summary"], "usage": result["usage"]})
        save_json(out / "partial-results.json", result)
        save_json(out / "results.json", result)
        trace(trace_path, "blind_reference_run_completed", image_count=len(result["images"]),
              status=result["status"], usage=result["usage"])
    except BaseException as exc:
        record_failure(exc, "finalization")
        raise
    return result


def main():
    args = arguments()
    result = asyncio.run(run(args))
    print(json.dumps({"results": str(args.out.expanduser().resolve() / "results.json"),
                      "status": result["status"], "summary": result["summary"], "usage": result["usage"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
