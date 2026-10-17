from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from threading import Lock
from . import __version__
from .schema import DocumentResult
from .tracing import append_trace_row, archive_restart_artifacts

class LocalOCRUnavailable(RuntimeError):
    """No candidate could be read after operational OCR errors; no paid model call."""

def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)

def source_hash(source: Path) -> str:
    digest=hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def trace(path: Path, event: str, **data):
    append_trace_row(path, {"time":time.time(),"event":event,**data})

def export_csv(out: Path, pages: list[dict]):
    names=["source_id","page_id","start_s","end_s","document_type","field","raw_value","normalized_value","status","confidence","frame_ids","errors"]
    with (out/"fields.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        for page in pages:
            result=page["result"]
            base={"source_id":result["source_id"],"page_id":page["page_id"],"start_s":page["start_s"],"end_s":page["end_s"],"document_type":result["document_type"],"errors":"; ".join(result["errors"])}
            for name,field in (result["fields"] or {"":{}}).items():
                writer.writerow({**base,"field":name,**{k:field.get(k) for k in ["raw_value","normalized_value","status","confidence"]},"frame_ids":";".join(e["frame_id"] for e in field.get("evidence",[]))})

async def process_video(source: Path, out: Path, *, config, sample_interval: float=0.5, max_pages: int=100, progress=None, registry=None) -> dict:
    if getattr(config, "document_reader", "ocr_guided") == "native_image":
        from .native_pipeline import process_video_images
        return await process_video_images(source, out, config=config, sample_interval=sample_interval,
                                          max_pages=max_pages, progress=progress,
                                          leaf_concurrency=getattr(config, "leaf_concurrency", 1))
    from .video import analyze_video
    from .ocr import read_image
    from .agents import analyze_page, FatalProviderError, BatchBudgetExceeded
    from .classifier import classify_pages
    from .multiview import compare_page_views
    from .quality import evaluate_page_frames
    source=source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    try:
        ocr_workers = int(os.getenv("SEVER_OCR_WORKERS", "1"))
    except ValueError:
        raise ValueError("SEVER_OCR_WORKERS must be an integer between 1 and 32") from None
    if not 1 <= ocr_workers <= 32:
        raise ValueError("SEVER_OCR_WORKERS must be an integer between 1 and 32")
    started=time.monotonic()
    usage_before=config.usage.model_dump()
    source_id=source_hash(source)
    trace_path=out/"trace.jsonl"
    history = archive_restart_artifacts(out, ("partial-results.json", "results.json", "metrics.json", "fields.csv", "report.html",
                                             "selected-pages.json", "classifications.json"))
    trace(trace_path,"run_started",source=str(source),source_id=source_id,version=__version__,sample_interval=sample_interval,
          local_ocr_workers=ocr_workers,
          previous_artifacts=str(history) if history else None)
    if progress: progress("Selecting stable page frames")
    pages=analyze_video(source,out/"video",sample_interval=sample_interval,max_pages=max_pages)
    # Candidate inventories are preserved even when every frame is rejected.
    # This keeps failed detections in evaluation denominators and avoids sending
    # black/blank/severely blurred source views to a paid model.
    for page in pages:
        page["quality_gate"] = evaluate_page_frames(page)
        if page["quality_gate"].get("selected_frame_id"):
            page["original_selected_frame_id"] = page["selected_frame_id"]
            page["selected_frame_id"] = page["quality_gate"]["selected_frame_id"]
        trace(trace_path, "quality_gate", page_id=page["page_id"], decision=page["quality_gate"])
    save_json(out/"selected-pages.json",pages)
    cache_locks = {}
    cache_locks_guard = Lock()
    def read_local_ocr(frame):
        storage = {"storage_format": "jpeg"} if frame.get("ocr_storage_format") == "jpeg" else {}
        path = Path(frame["path"])
        if ocr_workers > 1:
            # Equal bytes share cache image filenames even across frame ids.
            # Serialize content identities because image saves are non-atomic.
            content_hash = source_hash(path)
            with cache_locks_guard:
                content_lock = cache_locks.setdefault(content_hash, Lock())
            with content_lock:
                ocr = read_image(path, out/"ocr-cache", enhance_header=True, **storage)
        else:
            ocr = read_image(path, out/"ocr-cache", enhance_header=True, **storage)
        ocr.update(frame_id=frame["frame_id"], timestamp_s=frame["timestamp_s"])
        ocr["full_page_visible"] = ocr.get("processing", {}).get("full_page_visible") is True
        return ocr
    def ocr_read_event(frame, page_id, ocr):
        return {"page_id":page_id, "frame_id":frame["frame_id"], "engine":ocr["engine"],
                "region_count":len(ocr.get("regions", [])), "rotation_degrees":ocr.get("rotation_degrees", 0)}
    def local_ocr(frame, page_id):
        # Adaptive neighboring views remain sequential after the initial pool.
        ocr = read_local_ocr(frame)
        trace(trace_path, "ocr_read", **ocr_read_event(frame, page_id, ocr))
        return ocr
    def prepare_local_page(page):
        # Workers read unchanged candidates and return observations/events.
        # The coordinator alone writes shared progress/traces/page artifacts.
        ocr_pages=[]
        attempts=[]
        events=[]
        failures=0
        selected=page["selected_frame_id"]
        ordered_frames=sorted(page["frames"],key=lambda f:(f["frame_id"]!=selected,
                              -f.get("quality",{}).get("relative_selection_score",0)))
        accepted=set(page["quality_gate"].get("accepted_frame_ids",[]))
        for frame in ordered_frames if page["quality_gate"]["status"] == "accepted" else []:
            if frame["frame_id"] not in accepted:
                continue
            try:
                observation=read_local_ocr(frame)
                events.append(("ocr_read", ocr_read_event(frame, page["page_id"], observation)))
                if not observation.get("regions"):
                    attempts.append({"frame_id":frame["frame_id"],"status":"empty"})
                    events.append(("ocr_empty", {"page_id":page["page_id"],"frame_id":frame["frame_id"]}))
                    continue
                ocr_pages.append(observation)
                attempts.append({"frame_id":frame["frame_id"],"status":"read"})
                if frame["frame_id"] != selected:
                    events.append(("ocr_fallback_selected", {"page_id":page["page_id"],
                        "previous_frame_id":selected,"frame_id":frame["frame_id"],"reason":"primary_ocr_failed_or_empty"}))
                    selected=frame["frame_id"]
                break
            except Exception as exc:
                failures+=1
                attempts.append({"frame_id":frame["frame_id"],"status":"failed", "error_type":type(exc).__name__})
                events.append(("ocr_failed", {"page_id":page["page_id"],"frame_id":frame["frame_id"],"error_type":type(exc).__name__}))
        return {"ocr_pages":ocr_pages,"attempts":attempts,"events":events,"failures":failures,"selected_frame_id":selected}
    prepared=[]
    ocr_failures=0
    executor = ThreadPoolExecutor(max_workers=ocr_workers,thread_name_prefix="sever-ocr") if ocr_workers > 1 and len(pages) > 1 else None
    try:
        futures = [executor.submit(prepare_local_page, page) for page in pages] if executor else None
        for position,page in enumerate(pages,1):
            if progress: progress(f"Local OCR {position}/{len(pages)} at {page['start_s']:.1f}s")
            trace(trace_path,"page_selected",**page)
            prepared_page = futures[position-1].result() if futures is not None else prepare_local_page(page)
            page["ocr_attempts"]=prepared_page["attempts"]
            if prepared_page["selected_frame_id"] != page["selected_frame_id"]:
                page["selected_frame_id"]=prepared_page["selected_frame_id"]
                page["quality_gate"]["selected_frame_id"]=page["selected_frame_id"]
            for event, data in prepared_page["events"]:
                trace(trace_path,event,**data)
            ocr_failures+=prepared_page["failures"]
            save_json(out/"ocr"/f"{page['page_id']}.json",prepared_page["ocr_pages"])
            prepared.append({"page_id":page["page_id"],"ocr_pages":prepared_page["ocr_pages"]})
    finally:
        if executor:
            executor.shutdown(wait=True,cancel_futures=True)
    save_json(out/"selected-pages.json",pages)
    if ocr_failures and not any(item["ocr_pages"] for item in prepared):
        trace(trace_path,"batch_stopped",reason="local_ocr_unavailable",ocr_failures=ocr_failures,
              model_requests_attempted=0)
        raise LocalOCRUnavailable("Local OCR failed on every usable candidate; model classification was not requested")
    if progress: progress(f"ML classification of {len(pages)} leaves in batches")
    classifications=await classify_pages(prepared,source_id=source_id,config=config,cache_dir=out.parent/"llm-cache",trace_path=trace_path)
    save_json(out/"classifications.json",{key:value.model_dump() for key,value in classifications.items()})
    if registry is None and getattr(config, "verify_inn", True):
        from .inn import FnsRegistry
        registry = FnsRegistry(out.parent/"inn-cache", max_unique_lookups=10)
    processed=[]
    consecutive_failures=0
    for position,(page,prepared_page) in enumerate(zip(pages,prepared),1):
        ocr_pages=prepared_page["ocr_pages"]
        classification=classifications[page["page_id"]]
        document=classification
        if ocr_pages and classification.document_type!="unknown":
            if progress: progress(f"Fields {position}/{len(pages)}: {classification.document_type} at {page['start_s']:.1f}s")
            try:
                if config.vision_extraction:
                    from .vision import read_document
                    document=await read_document(ocr_pages,source_id=source_id,page_id=page["page_id"],document_type=classification.document_type,config=config,cache_dir=out.parent/"vision-cache",trace_path=trace_path)
                else:
                    document=await analyze_page(ocr_pages,source_id=source_id,page_id=page["page_id"],config=config,cache_dir=out.parent/"llm-cache",trace_path=trace_path)
                document=DocumentResult.model_validate(document) if isinstance(document,dict) else document
                if getattr(config, "neighbor_fallback", True):
                    async def reader(views):
                        if config.vision_extraction:
                            return await read_document(views, source_id=source_id, page_id=page["page_id"],
                                document_type=classification.document_type, config=config,
                                cache_dir=out.parent/"vision-cache", trace_path=trace_path)
                        return await analyze_page(views, source_id=source_id, page_id=page["page_id"],
                            config=config, cache_dir=out.parent/"llm-cache", trace_path=trace_path)
                    def record_comparison(attempt):
                        save_json(out/"ocr"/f"{page['page_id']}.json", ocr_pages)
                        trace(trace_path, "neighbor_comparison", page_id=page["page_id"], **attempt)
                    document = await compare_page_views(source, page, ocr_pages, document,
                        classification.document_type, out=out, local_ocr=local_ocr,
                        reader=reader, record=record_comparison)
                consecutive_failures=0
            except (FatalProviderError, BatchBudgetExceeded) as exc:
                trace(trace_path,"batch_stopped",page_id=page["page_id"],error_type=type(exc).__name__)
                raise
            except Exception as exc:
                consecutive_failures+=1
                document=DocumentResult(source_id=source_id,page_id=page["page_id"],document_type="unknown",errors=[f"extraction_failed:{type(exc).__name__}"])
                trace(trace_path,"extraction_failed",page_id=page["page_id"],error_type=type(exc).__name__)
        elif not ocr_pages:
            rejected = page["quality_gate"]["status"] == "rejected"
            error = "quality_rejected:all_retained_frames_clearly_unusable" if rejected else "ocr_failed"
            document=DocumentResult(source_id=source_id,page_id=page["page_id"],document_type="unknown",errors=[error])
        if getattr(config, "verify_inn", True) and document.document_type != "unknown":
            from .inn import verify_document_inns
            try:
                page["inn_verification"] = await verify_document_inns(ocr_pages, document, registry=registry)
            except Exception as exc:
                page["inn_verification"] = {"status": "unavailable", "error_type": type(exc).__name__}
            trace(trace_path, "inn_verification", page_id=page["page_id"], verification=page["inn_verification"])
        page["classification"]=classification.model_dump()
        page["result"]=document.model_dump(mode="json")
        page["ocr_artifact"]=str(out/"ocr"/f"{page['page_id']}.json")
        processed.append(page)
        save_json(out/"partial-results.json",{"source_id":source_id,"pages":processed})
        if consecutive_failures>=3:
            trace(trace_path,"batch_stopped",reason="three_consecutive_extraction_failures",usage=config.usage.model_dump())
            raise RuntimeError("Three consecutive extraction failures; batch stopped to preserve the API budget")
    batch_usage=config.usage.model_dump() if hasattr(config.usage,"model_dump") else vars(config.usage)
    usage={key:(value-usage_before.get(key,0) if isinstance(value,(int,float)) and not isinstance(value,bool) else value)
           for key,value in batch_usage.items()}
    result={"schema_version":1,"version":__version__,"source":{"path":str(source),"sha256":source_id},"configuration":{"sample_interval":sample_interval,"max_pages":max_pages,"model":config.model_name,"field_engine":f"selected role/header pixels via {config.vision_model_name}" if config.vision_extraction else "OCR-only agent","classification_scope":["upd","torg12","ttn","trn","unknown"], "neighbor_fallback":getattr(config,"neighbor_fallback",True), "per_leaf_frame_limit":None, "verify_inn":getattr(config,"verify_inn",True)},"pages":processed,"usage":usage,"batch_usage":batch_usage,"elapsed_seconds":round(time.monotonic()-started,2),"unit":"visible leaf; adaptive same-leaf multi-frame reading with no per-leaf frame-count limit","limitations":["Quality scores and model confidences are indicators, not calibrated probabilities.","Provisional evaluation requires customer-approved ground truth.","Registry matching verifies legal entity identity, never overwrites printed shipping addresses."]}
    result["configuration"]["local_ocr_workers"] = ocr_workers
    result["configuration"]["model_backend"] = config.provider_kind
    result["configuration"]["model_base_url"] = config.base_url
    save_json(out/"selected-pages.json",[{k:v for k,v in page.items() if k not in {"result", "classification", "inn_verification"}} for page in processed])
    if registry is not None and hasattr(registry, "stats"):
        result["registry_usage"] = registry.stats()
    try:
        export_csv(out,processed)
        from .report import write_report
        write_report(out,result)
        # Promote the final JSON only after the other exports succeeded.
        save_json(out/"results.json",result)
        trace(trace_path,"run_completed",page_count=len(processed),elapsed_seconds=result["elapsed_seconds"],usage=usage)
    except BaseException:
        # A failed final export/trace must not leave a current "completed"
        # results path. Keep all generated data in history and the current
        # partial-results checkpoint, rather than deleting or hiding evidence.
        try:
            archive_restart_artifacts(out, ("results.json", "metrics.json", "fields.csv", "report.html"), reason="failed-finalization")
        except OSError:
            # A same-directory rename needs no copy of a large result. Preserve
            # the original exception if exhausted storage also blocks history.
            current = out/"results.json"
            if current.is_file():
                try:
                    current.rename(out/f"results.interrupted-{time.time_ns()}.json")
                except OSError:
                    pass
        raise
    return result
