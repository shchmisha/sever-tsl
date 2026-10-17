"""Run supplied local videos with one ephemeral credential and one shared budget."""
import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sever.agents import AgentConfig
from sever.pipeline import process_video,save_json
from sever.inn import FnsRegistry
from sever.evaluation import evaluate
from sever.tracing import archive_restart_artifacts

async def run(args):
    key="" if args.cache_only else os.getenv("HARNESS_MODEL_API_KEY") or getpass.getpass("RouterAI API key (process only, not saved): ")
    config=AgentConfig(model_name=args.model,vision_model_name=os.getenv("HARNESS_VISION_MODEL","openai/gpt-4.1-mini"),base_url=os.getenv("HARNESS_MODEL_BASE_URL","https://routerai.ru/api/v1").rstrip("\\/"),api_key=key,max_requests=1 if args.cache_only else args.max_requests,max_tokens=1000 if args.cache_only else args.max_tokens,max_output_tokens=2200)
    summaries=[]
    registry=FnsRegistry(args.out/"inn-cache",max_unique_lookups=10)
    truth=json.loads(args.truth.read_text()) if args.truth else None
    # A previous completed summary belongs to the previous attempt. A failed
    # restart must not keep presenting it as this attempt's completed result.
    archive_restart_artifacts(args.out, ("pilot-summary.json",))
    for source in args.sources:
        out=args.out/source.stem
        if args.baseline:
            old=args.baseline/source.stem
            out.mkdir(parents=True,exist_ok=True)
            for name in ("video","ocr-cache"):
                target=out/name
                if not target.exists() and (old/name).is_dir():
                    target.symlink_to((old/name).resolve(),target_is_directory=True)
            for name in ("llm-cache","vision-cache"):
                target=args.out/name
                if not target.exists() and (args.baseline/name).is_dir():
                    target.symlink_to((args.baseline/name).resolve(),target_is_directory=True)
        result=await process_video(source,out,config=config,registry=registry,sample_interval=args.interval,max_pages=100,progress=lambda s:print(f"{source.name}: {s}",flush=True))
        if truth:
            matches=[video for video in truth["videos"] if Path(video["source"]).name==source.name]
            if len(matches)!=1:
                raise ValueError("Expected exactly one independent ground truth source")
            expected={**matches[0],"customer_approved":truth.get("customer_approved",False),"provenance":truth.get("provenance")}
            save_json(out/"metrics.json",evaluate(result,expected))
        summaries.append({"source":str(source),"out":str(out.resolve()),"pages":len(result["pages"]),"types":{t:sum(p["result"]["document_type"]==t for p in result["pages"]) for t in ["upd","torg12","ttn","trn","unknown"]},"errors":sum(bool(p["result"]["errors"]) for p in result["pages"])})
        print(json.dumps(summaries[-1],ensure_ascii=False),flush=True)
    save_json(args.out/"pilot-summary.json",{"videos":summaries,"total_usage":config.usage.model_dump(),"registry_usage":registry.stats(),"limits":{"max_requests":args.max_requests,"max_tokens":args.max_tokens,"per_leaf_frames":None}})
    print(json.dumps(config.usage.model_dump(),ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources",type=Path,nargs="+")
    parser.add_argument("--out",type=Path,default=Path("runs/pilot"))
    parser.add_argument("--cache-only",action="store_true",help="Use saved responses only; a cache miss stops before any provider request")
    parser.add_argument("--baseline",type=Path,help="Reuse preserved frame/OCR/response caches without overwriting old results")
    parser.add_argument("--truth",type=Path,help="Evaluate after processing against independent labels")
    parser.add_argument("--model",default="openai/gpt-4.1-mini")
    parser.add_argument("--interval",type=float,default=0.5)
    parser.add_argument("--max-requests",type=int,default=100)
    parser.add_argument("--max-tokens",type=int,default=250000)
    asyncio.run(run(parser.parse_args()))
