"""Four rendered PDF references through batch OCR classification and pixel reading.

This is a paid-provider runner, not a video-detection benchmark. It never saves
credentials. Offline API/model tests are separate and require no provider key.
"""
import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sever.agents import AgentConfig
from sever.classifier import classify_pages
from sever.ocr import read_image
from sever.pipeline import save_json,export_csv,source_hash
from sever.report import write_report
from sever.evaluation import evaluate
from sever.vision import read_document


def arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/Users/shchmisha/Downloads/buh00280420260827120747.pdf"))
    parser.add_argument("--images-dir", type=Path, default=root / "analysis/reference-pdf")
    parser.add_argument("--truth", type=Path, default=root / "analysis/reference-pdf/ground-truth.json")
    parser.add_argument("--out", type=Path, default=root / "runs/reference-pdf")
    parser.add_argument("--model", default="openai/gpt-4.1-mini", help="Model for compact OCR batch classification")
    parser.add_argument("--vision-model", default="openai/gpt-4.1-mini", help="Model for selected source role-panel reading")
    parser.add_argument("--cache-only", action="store_true", help="Revalidate saved responses only; no provider requests")
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=100000)
    args = parser.parse_args()
    if args.max_requests < 5 or args.max_tokens < 100000:
        parser.error("Four target references require a shared allowance of at least 5 requests and 100000 tokens")
    return args


async def run(args):
    source, out = args.source.expanduser().resolve(), args.out.expanduser().resolve()
    images = [(args.images_dir / f"ocr-page-{n}.png" if (args.images_dir / f"ocr-page-{n}.png").is_file() else args.images_dir / f"ocr-page-{n}.jpg").resolve() for n in range(1, 5)]
    for path in [source, args.truth, *images]:
        if not path.is_file():
            raise FileNotFoundError(path)
    truth = json.loads(args.truth.read_text())
    key = "" if getattr(args,"cache_only",False) else (os.getenv("HARNESS_MODEL_API_KEY") or getpass.getpass("RouterAI API key (process only, not saved): ")).strip().rstrip("\\")
    if not key and not getattr(args,"cache_only",False):
        raise ValueError("Provider key is required")
    config = AgentConfig(model_name=args.model, vision_model_name=getattr(args,"vision_model","openai/gpt-4.1-mini"),
                         base_url=os.getenv("HARNESS_MODEL_BASE_URL", "https://routerai.ru/api/v1").strip().rstrip("\\/"),
                         api_key=key, max_requests=1 if getattr(args,"cache_only",False) else args.max_requests, max_tokens=1000 if getattr(args,"cache_only",False) else args.max_tokens, max_output_tokens=2200)
    out.mkdir(parents=True, exist_ok=True)
    started, source_id = time.monotonic(), source_hash(source)
    trace_path = out / "trace.jsonl"
    pages, prepared = [], []
    for n in range(1,5):
        image, page_id = images[n - 1], f"reference-{n}"
        print(f"Reference PDF page {n}: local OCR", flush=True)
        ocr = read_image(image, out / "ocr-cache", enhance_header=True)
        # The input is a complete rendered PDF page, not an inferred camera outline.
        ocr.update(frame_id=page_id, timestamp_s=n - .5, full_page_visible=True)
        save_json(out / "ocr" / f"{page_id}.json", [ocr])
        prepared.append({"page_id": page_id, "ocr_pages": [ocr]})
        pages.append({"page_id": page_id, "start_s": n - 1, "end_s": n,
                      "selected_frame_id": page_id,
                      "frames": [{"frame_id": page_id, "timestamp_s": n - .5, "path": str(image),
                                  "quality": {"source": "complete rendered PDF, not camera video", "full_page_visible": True}}],
                      "ocr_artifact": str(out / "ocr" / f"{page_id}.json")})
    save_json(out / "selected-pages.json", pages)
    print("Reference PDF: one batch classification of four OCR pages", flush=True)
    classifications = await classify_pages(prepared, source_id=source_id, config=config,
                                           cache_dir=out / "llm-cache", trace_path=trace_path)
    save_json(out / "classifications.json", {page_id: value.model_dump() for page_id, value in classifications.items()})
    processed = []
    for page, prepared_page in zip(pages, prepared):
        classification = classifications[page["page_id"]]
        document = classification
        if classification.document_type != "unknown":
            print(f"{page['page_id']}: selected header pixels → {classification.document_type} fields", flush=True)
            document = await read_document(prepared_page["ocr_pages"], source_id=source_id, page_id=page["page_id"],
                                           document_type=classification.document_type, config=config,
                                           cache_dir=out / "vision-cache", trace_path=trace_path)
        page["classification"] = classification.model_dump(mode="json")
        page["result"] = document.model_dump(mode="json")
        processed.append(page)
        save_json(out / "partial-results.json", {"source_id": source_id, "pages": processed})
        print(document.document_type, [(name, field.status, field.normalized_value) for name, field in document.fields.items()], flush=True)
    result={"schema_version":1,"source":{"path":str(source),"sha256":source_id},"pages":pages,
            "elapsed_seconds":round(time.monotonic()-started,2),"usage":config.usage.model_dump(),
            "configuration":{"classification_model":config.model_name,"vision_model":config.vision_model_name,
                             "max_requests":args.max_requests,"max_tokens":args.max_tokens},
            "unit":"rendered PDF reference page; timestamps are synthetic page-index anchors, not video"}
    save_json(out/"results.json",result);export_csv(out,pages);write_report(out,result)
    report=evaluate(result,truth)
    report["unit"]="rendered PDF reference page, not video"
    report["detection"]={"rate":None,"reason":"PDF image test does not test video frame selection"}
    report["selected_frame_quality"]={"evaluated":False,"reason":"Rendered PDF references do not evaluate camera shooting quality"}
    save_json(out/"metrics.json",report)
    print(json.dumps({"usage":config.usage.model_dump(),"types":report["type_accuracy_targets"],
                      "fields":report["field_accuracy"],"e2e_fully_labeled":report["e2e_on_fully_labeled"]},ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":asyncio.run(run(arguments()))
