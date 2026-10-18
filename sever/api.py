"""Local MVP API with persistent job state and a bounded single worker."""
from __future__ import annotations
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from threading import Lock
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .pipeline import process_video,save_json
from .cli import agent_config
from .inference_gate import InferenceGateBusy, gate_path, inference_access

ROOT=Path(os.getenv("SEVER_RUNS_DIR","runs")).resolve()
WORKER=ThreadPoolExecutor(max_workers=1,thread_name_prefix="sever")
_ACTIVE_FUTURES: set[Future] = set()
_FUTURES_LOCK = Lock()


@contextmanager
def api_instance_access():
    """Only one API process may own this run directory and its worker queue."""
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / ".api-worker.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("API worker lock must be a persistent regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another API process owns this run directory; use one Uvicorn worker") from None
        yield
    finally:
        # Keep the inode stable for subsequent API starts.
        os.close(descriptor)


def reconcile_interrupted_jobs():
    """Run before admission starts, while holding the sole API instance lock.

    A previous process's in-memory queue cannot resume. Preserve every artifact
    and mark only its nonterminal job state as interrupted; never infer success
    from a partial report or relaunch recognition here.
    """
    interrupted = []
    for path in ROOT.glob("*/job.json"):
        try:
            uuid.UUID(path.parent.name)
        except ValueError:
            continue
        saved = json.loads(path.read_text())
        if saved.get("status") in {"queued", "processing"}:
            interrupted.append((path.parent.name, saved["status"]))
    if interrupted:
        with inference_access(gate_path(run_dir=ROOT)):
            for job_id, previous_status in interrupted:
                state(job_id, {"status": "failed", "error_type": "InterruptedByServiceRestart",
                               "interrupted_status": previous_status})


@asynccontextmanager
async def lifespan(app: FastAPI):
    with api_instance_access():
        reconcile_interrupted_jobs()
        try:
            yield
        finally:
            # A graceful stop cannot release ownership while an accepted
            # worker is still writing. A forced process stop releases flock;
            # the next startup then marks unfinished jobs as interrupted.
            with _FUTURES_LOCK:
                pending = tuple(_ACTIVE_FUTURES)
            if pending:
                await asyncio.to_thread(wait, pending)


app=FastAPI(title="Sever TSL",version="0.1.0",lifespan=lifespan)

class JobRequest(BaseModel):
    source_path: str
    model: str = os.getenv("HARNESS_MODEL_NAME","openai/gpt-4.1-mini")
    sample_interval: float = Field(default=0.5,gt=0,le=5)
    max_pages: int = Field(default=100,ge=1,le=300)
    max_requests: int = Field(default=100,ge=1,le=4000)
    max_tokens: int = Field(default=150000,ge=1000,le=40000000)
    refresh_cache: bool = False
    native_implementation_fingerprint: str | None = Field(default=None,pattern=r"^[a-f0-9]{64}$")

def _native_source_hashes():
    root=Path(__file__).resolve().parent
    names=("image_reader.py","image_fusion.py","native_pipeline.py","schema.py",
           "adaptive_image_reader.py","image_orientation.py","image_triage.py","image_detail.py","documents.py",
           "image_agent.py","zoom_image_reader.py","image_conflicts.py","inference_gate.py",
           "pipeline.py","video.py","frame_pool.py","quality.py","normalize.py","coverage.py",
           "agents.py","cli.py","tracing.py","api.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}

def native_implementation():
    """Fingerprint actual native code and effective nonsecret configuration."""
    local=os.getenv("HARNESS_PROVIDER_KIND","remote")=="local"
    reader=os.getenv("HARNESS_DOCUMENT_READER","native_image" if local else "ocr_guided")
    if reader!="native_image":
        return {"status":"not_active","fingerprint_sha256":None}
    from .image_reader import IMAGE_READER_VERSION,IMAGE_SCHEMA_VERSION,INSTRUCTIONS as FALLBACK_INSTRUCTIONS,_NativeReadingResponse
    from .image_agent import IMAGE_AGENT_VERSION,INSTRUCTIONS as AGENT_INSTRUCTIONS,ZOOM_INSTRUCTIONS,MAX_MODEL_REQUESTS,MAX_ZOOM_IMAGES
    from .image_orientation import ORIENTATION_VERSION,ORIENTATION_OUTPUT_LIMIT,INSTRUCTIONS as ORIENTATION_INSTRUCTIONS,_OrientationDecision
    from .zoom_image_reader import ZOOM_READER_VERSION
    from .image_triage import TRIAGE_VERSION,TRIAGE_OUTPUT_LIMIT,INSTRUCTIONS as TRIAGE_INSTRUCTIONS,_TriageDecision
    from .native_pipeline import _NO_PROGRESS_POLICY
    from . import image_conflicts
    from .schema import FIELDS_BY_TYPE
    model=os.getenv("HARNESS_MODEL_NAME","openai/gpt-4.1-mini")
    schema=json.dumps(_NativeReadingResponse.model_json_schema(),sort_keys=True,
                      ensure_ascii=False,separators=(",",":"))
    identity={"contract_version":3,"reader_version":ZOOM_READER_VERSION,
        "fallback_reader_version":IMAGE_READER_VERSION,"image_agent_version":IMAGE_AGENT_VERSION,
        "schema_version":IMAGE_SCHEMA_VERSION,
        "instructions_sha256":hashlib.sha256((AGENT_INSTRUCTIONS + ZOOM_INSTRUCTIONS).encode()).hexdigest(),
        "fallback_instructions_sha256":hashlib.sha256(FALLBACK_INSTRUCTIONS.encode()).hexdigest(),
        "schema_sha256":hashlib.sha256(schema.encode()).hexdigest(),
        "reading_output_mode":"tool_output:submit_reading",
        "fallback_output_mode":"native_output",
        "per_frame_limits":{"agent_model_requests":MAX_MODEL_REQUESTS,"zoom_images":MAX_ZOOM_IMAGES},
        "conflict_review":{"version":image_conflicts.IMAGE_CONFLICTS_VERSION,
            "output_mode":"tool_output:submit_conflict_review",
            "instructions_sha256":hashlib.sha256(image_conflicts.INSTRUCTIONS.encode()).hexdigest(),
            "type_instructions_sha256":hashlib.sha256(image_conflicts.TYPE_INSTRUCTIONS.encode()).hexdigest(),
            "model_requests":image_conflicts.MAX_MODEL_REQUESTS,
            "zoom_images":image_conflicts.MAX_ZOOM_IMAGES,
            "max_joint_frames":image_conflicts.MAX_REVIEW_FRAMES,
            "schema_sha256":hashlib.sha256(json.dumps({kind:image_conflicts._response_model(kind,list(names)).model_json_schema()
                for kind,names in FIELDS_BY_TYPE.items()},sort_keys=True,
                ensure_ascii=False,separators=(",",":")).encode()).hexdigest()},
        "orientation":{"version":ORIENTATION_VERSION,"output_limit":ORIENTATION_OUTPUT_LIMIT,
            "output_mode":"native_output",
            "instructions_sha256":hashlib.sha256(ORIENTATION_INSTRUCTIONS.encode()).hexdigest(),
            "schema_sha256":hashlib.sha256(json.dumps(_OrientationDecision.model_json_schema(),sort_keys=True,
                ensure_ascii=False,separators=(",",":")).encode()).hexdigest()},
        "triage":{"version":TRIAGE_VERSION,"output_limit":TRIAGE_OUTPUT_LIMIT,
            "output_mode":"native_output",
            "instructions_sha256":hashlib.sha256(TRIAGE_INSTRUCTIONS.encode()).hexdigest(),
            "schema_sha256":hashlib.sha256(json.dumps(_TriageDecision.model_json_schema(),sort_keys=True,
                ensure_ascii=False,separators=(",",":")).encode()).hexdigest()},
        "leaf_search":{"no_progress_stop":_NO_PROGRESS_POLICY,"orientation":"first_fresh_decision_shared_by_leaf"},
        "source_sha256":_native_source_hashes(),
        "configuration":{"document_reader":reader,"provider_kind":"local" if local else "remote",
            "model_name":model,"vision_model":os.getenv("HARNESS_VISION_MODEL",model),
            "max_output_tokens":os.getenv("HARNESS_MAX_OUTPUT_TOKENS","2200"),
            "leaf_concurrency":os.getenv("HARNESS_LEAF_CONCURRENCY","1"),
            "timeout_seconds":os.getenv("HARNESS_MODEL_TIMEOUT","600" if local else "60"),
            "model_base_url_sha256":hashlib.sha256(os.getenv("HARNESS_MODEL_BASE_URL",
                "https://routerai.ru/api/v1").strip().rstrip("\\/").encode()).hexdigest()}}
    serialized=json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return {"status":"available",**identity,
            "fingerprint_sha256":hashlib.sha256(serialized.encode()).hexdigest()}

def directory(job_id:str)->Path:
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(404,"Unknown job")
    return ROOT/job_id

def state(job_id:str,value:dict):
    path = directory(job_id)/"job.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    updated = {**previous, **value, "job_id": job_id}
    now = datetime.now(timezone.utc).isoformat()
    updated["updated_at"] = now
    if value.get("status") == "queued":
        updated.setdefault("enqueued_at", now)
    elif value.get("status") == "processing":
        updated.setdefault("started_at", now)
    elif value.get("status") in {"completed", "completed_with_errors", "failed"}:
        updated.setdefault("finished_at", now)
    save_json(path, updated)


def _forget_future(future):
    with _FUTURES_LOCK:
        _ACTIVE_FUTURES.discard(future)

def execute(job_id:str, request:JobRequest, config):
    binding=({"native_implementation_fingerprint":request.native_implementation_fingerprint}
             if request.native_implementation_fingerprint else {})
    try:
        with inference_access(gate_path(run_dir=ROOT), blocking=True):
            try:
                state(job_id,{"status":"processing","source_path":request.source_path,**binding})
                def progress(message):
                    state(job_id,{"status":"processing","source_path":request.source_path,"progress":message,**binding})
                if binding and native_implementation().get("fingerprint_sha256")!=request.native_implementation_fingerprint:
                    state(job_id,{"status":"failed","error_type":"NativeImplementationChanged",**binding})
                    return
                result=asyncio.run(process_video(Path(request.source_path),directory(job_id),config=config,sample_interval=request.sample_interval,max_pages=request.max_pages,progress=progress))
                failed=sum(bool(p["result"]["errors"]) for p in result["pages"])
                state(job_id,{"status":"completed_with_errors" if failed else "completed","page_count":len(result["pages"]),"pages_with_errors":failed,"usage":result["usage"],"results_url":f"/v1/jobs/{job_id}/results","csv_url":f"/v1/jobs/{job_id}/fields.csv","documents_csv_url":f"/v1/jobs/{job_id}/documents.csv",**binding})
            except Exception as exc:
                state(job_id,{"status":"failed","error_type":type(exc).__name__,**binding})
    except Exception as exc:
        state(job_id,{"status":"failed","error_type":type(exc).__name__,**binding})

@app.get("/health")
def health():
    import shutil,platform
    local=os.getenv("HARNESS_PROVIDER_KIND","remote")=="local"
    return {"status":"ok","ffmpeg":bool(shutil.which("ffmpeg")),"ocr_platform":platform.system(),
            "model_configured":bool(os.getenv("HARNESS_MODEL_BASE_URL")) if local else bool(os.getenv("HARNESS_MODEL_API_KEY")),
            "model_backend":"local" if local else "remote", "model_name":os.getenv("HARNESS_MODEL_NAME","openai/gpt-4.1-mini"),
            "document_reader":os.getenv("HARNESS_DOCUMENT_READER", "native_image" if local else "ocr_guided"),
            "native_response_reuse":False,"native_semantic_validation":"model","worker_count":1,
            "native_implementation":native_implementation()}

@app.post("/v1/jobs",status_code=202)
def create_job(request:JobRequest):
    source=Path(request.source_path).expanduser().resolve()
    if not source.is_file():
        raise HTTPException(400,"Source must be an existing local video file")
    if os.getenv("HARNESS_PROVIDER_KIND","remote")!="local" and request.max_tokens>500000:
        raise HTTPException(400,"Remote model jobs are limited to 500000 tokens")
    if os.getenv("HARNESS_PROVIDER_KIND","remote")!="local" and request.max_requests>400:
        raise HTTPException(400,"Remote model jobs are limited to 400 requests")
    if request.native_implementation_fingerprint and native_implementation().get("fingerprint_sha256")!=request.native_implementation_fingerprint:
        raise HTTPException(409,"Native implementation changed; start a new validation batch")
    try:
        with inference_access(gate_path(run_dir=ROOT)):
            try:
                config=agent_config(request.model,request.max_requests,request.max_tokens)
                config.refresh_cache=request.refresh_cache
            except Exception:
                raise HTTPException(503,"Model configuration is missing or invalid")
            request.source_path=str(source)
            job_id=str(uuid.uuid4())
            binding=({"native_implementation_fingerprint":request.native_implementation_fingerprint}
                     if request.native_implementation_fingerprint else {})
            state(job_id,{"status":"queued","source_path":str(source),**binding})
            try:
                future = WORKER.submit(execute,job_id,request,config)
            except Exception:
                state(job_id, {"status": "failed", "error_type": "WorkerUnavailable"})
                raise HTTPException(503,"Recognition worker is unavailable") from None
            if isinstance(future, Future):
                with _FUTURES_LOCK:
                    _ACTIVE_FUTURES.add(future)
                future.add_done_callback(_forget_future)
    except InferenceGateBusy:
        raise HTTPException(423,"Inference is locked during isolated evaluation") from None
    except OSError:
        raise HTTPException(503,"Inference gate or job storage is unavailable") from None
    return {"job_id":job_id,"status":"queued","status_url":f"/v1/jobs/{job_id}"}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id:str):
    path=directory(job_id)/"job.json"
    if not path.exists(): raise HTTPException(404,"Unknown job")
    return json.loads(path.read_text())

@app.get("/v1/jobs/{job_id}/results")
def results(job_id:str):
    path=directory(job_id)/"results.json"
    if not path.exists(): raise HTTPException(409,"Results are not available yet")
    return json.loads(path.read_text())

@app.get("/v1/jobs/{job_id}/fields.csv")
def fields(job_id:str):
    path=directory(job_id)/"fields.csv"
    if not path.exists(): raise HTTPException(409,"Results are not available yet")
    return FileResponse(path,filename=f"{job_id}-fields.csv",media_type="text/csv")

@app.get("/v1/jobs/{job_id}/documents.csv")
def documents_csv(job_id:str):
    """One row per document (form + own number); fields.csv keeps one row per sheet field."""
    path=directory(job_id)/"documents.csv"
    if not path.exists(): raise HTTPException(409,"Results are not available yet")
    return FileResponse(path,filename=f"{job_id}-documents.csv",media_type="text/csv")
