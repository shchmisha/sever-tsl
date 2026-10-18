from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import typer
from .pipeline import process_video,save_json
from .evaluation import evaluate
from .inference_gate import inference_access

app=typer.Typer(no_args_is_help=True,help="Sever TSL: video → source page images → document readings and fields.")

def agent_config(model: str, max_requests: int, max_tokens: int):
    from .agents import AgentConfig
    provider_kind=os.getenv("HARNESS_PROVIDER_KIND","remote").strip()
    key=os.getenv("HARNESS_MODEL_API_KEY","").strip().rstrip("\\")
    if not key and provider_kind != "local":
        raise typer.BadParameter("Set HARNESS_MODEL_API_KEY in the process environment; never include it in a command line or source file.")
    return AgentConfig(
        model_name=model, vision_model_name=os.getenv("HARNESS_VISION_MODEL",model),
        base_url=os.getenv("HARNESS_MODEL_BASE_URL","https://routerai.ru/api/v1").strip().rstrip("\\/"),
        provider_kind=provider_kind, api_key="local-no-key" if provider_kind=="local" else key,
        document_reader=os.getenv("HARNESS_DOCUMENT_READER", "native_image" if provider_kind=="local" else "ocr_guided"),
        max_requests=max_requests, max_tokens=max_tokens,
        input_tokens_limit=max(800000,max_tokens) if provider_kind=="local" else 800000,
        output_tokens_limit=max(120000,max_tokens) if provider_kind=="local" else 120000,
        max_output_tokens=int(os.getenv("HARNESS_MAX_OUTPUT_TOKENS","2200")),
        timeout_seconds=float(os.getenv("HARNESS_MODEL_TIMEOUT","600" if provider_kind=="local" else "60")),
        # Match llama-server --parallel; 1 keeps leaves strictly sequential.
        leaf_concurrency=int(os.getenv("HARNESS_LEAF_CONCURRENCY","1")))

@app.command()
def process(source:Path,out:Path=typer.Option(Path("runs/latest")),interval:float=0.5,max_pages:int=100,model:str=typer.Option("openai/gpt-4.1-mini",envvar="HARNESS_MODEL_NAME"),max_requests:int=100,max_tokens:int=150_000,refresh_cache:bool=False):
    """Process a video and save JSON, CSV, source frames, readings and audit trail."""
    if interval<=0 or max_pages<=0 or max_requests<=0 or max_tokens<=0:
        raise typer.BadParameter("Limits and interval must be positive")
    config=agent_config(model,max_requests,max_tokens)
    config.refresh_cache=refresh_cache
    with inference_access():
        result=asyncio.run(process_video(source,out,config=config,sample_interval=interval,max_pages=max_pages,progress=typer.echo))
    typer.echo(json.dumps({"result":str(out.resolve()/"results.json"),"page_count":len(result["pages"]),"usage":result["usage"]},ensure_ascii=False))

@app.command("documents")
def documents_command(run:Path,report:bool=typer.Option(True,help="Also rebuild report.html; the previous report is kept under history/")):
    """Build the per-document table (documents.csv, documents.json) from a finished run, without model calls."""
    from .documents import assemble_documents, export_documents
    from .report import write_report
    from .tracing import archive_restart_artifacts
    path=run/"results.json"
    if not path.is_file():
        raise typer.BadParameter(f"{path} not found")
    result=json.loads(path.read_text(encoding="utf-8"))
    assembly=assemble_documents(result)
    export_documents(run,assembly)
    if report:
        history=archive_restart_artifacts(run,("report.html",),reason="documents-export")
        try:
            write_report(run,{**result,"documents":assembly})
        except BaseException:
            # Never leave a finished run without its report: put the previous one back.
            if history is not None and (history/"report.html").is_file():
                (history/"report.html").replace(run/"report.html")
            raise
    typer.echo(json.dumps({"documents_csv":str(run.resolve()/"documents.csv"),**assembly["summary"]},ensure_ascii=False))

@app.command("evaluate")
def evaluate_command(result:Path,truth:Path,out:Path=Path("runs/metrics.json")):
    """Evaluate against independently labeled expected leaf intervals."""
    predicted=json.loads(result.read_text())
    expected=json.loads(truth.read_text())
    if "videos" in expected:
        filename=Path(predicted["source"]["path"]).name
        candidates=[v for v in expected["videos"] if Path(v.get("source",v.get("source_path",""))).name==filename]
        if len(candidates)!=1:
            raise typer.BadParameter("Ground truth must contain exactly one matching video")
        expected={**candidates[0],"customer_approved":expected.get("customer_approved",False),"provenance":expected.get("provenance")}
    report=evaluate(predicted,expected)
    save_json(out,report)
    typer.echo(json.dumps(report,ensure_ascii=False,indent=2))

@app.command()
def serve(host:str="127.0.0.1",port:int=8080):
    """Start the local processing API."""
    import uvicorn
    uvicorn.run("sever.api:app",host=host,port=port)

if __name__=="__main__":
    app()
