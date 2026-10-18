"""Small authenticated-team facade; the recognition API remains private.

Run on loopback behind a proxy which authenticates and *overwrites*
X-Sever-User. This module never accepts server paths or model configuration
from a browser, and never replays an uncertain submission.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import uuid
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .coverage import load_video_coverage, summarize_video_coverage

DATA = Path(os.getenv("PORTAL_DATA_DIR", "/srv/sever/customer"))
RUNS = Path(os.getenv("SEVER_RUNS_DIR", "/srv/sever/runs"))
API_URL = os.getenv("SEVER_API_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_BYTES = 4 * 1024**3
MAX_DURATION = 3600
MAX_ACTIVE = 4
MAX_TEAM_ACTIVE = 2
MIN_FREE_BYTES = 100 * 1024**2
ACTIVE = ("uploading", "validating", "submitting", "submission_uncertain", "queued", "processing")
TERMINAL = ("completed", "completed_with_errors", "failed")
EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
FILES = {"report.html", "documents.csv", "fields.csv", "results.json", "partial-results.json"}
IMAGE_DIRS = {"video", "additional-frames", "image-reading-evidence", "report-assets", "frames"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database():
    DATA.mkdir(parents=True, exist_ok=True, mode=0o750)
    connection = sqlite3.connect(DATA / "portal.sqlite3", timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                           "name TEXT NOT NULL, source TEXT NOT NULL, job_id TEXT UNIQUE, status TEXT NOT NULL, "
                           "progress TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                           "size_bytes INTEGER NOT NULL DEFAULT 0, error TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, portal_id TEXT NOT NULL, "
                           "created_at TEXT NOT NULL, page_id TEXT, field TEXT, comment TEXT NOT NULL, proposed_value TEXT)")
        connection.commit()
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def update(identifier, **values):
    values["updated_at"] = now()
    with database() as db:
        db.execute("UPDATE jobs SET " + ", ".join(f"{name}=?" for name in values) + " WHERE id=?",
                   (*values.values(), identifier))


def startup_reconcile():
    """An interrupted POST is never automatically submitted again."""
    with database() as db:
        db.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status IN ('uploading','validating')",
                   ("Загрузка прервана перезапуском. Загрузите видео заново.", now()))
        db.execute("UPDATE jobs SET status='submission_uncertain', error=?, updated_at=? WHERE status='submitting'",
                   ("Не удалось подтвердить запуск. Нужна проверка оператором; повторный запуск не выполнялся.", now()))


@asynccontextmanager
async def lifespan(app):
    parsed = urlsplit(API_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("SEVER_API_URL must be a loopback HTTP endpoint")
    startup_reconcile()
    yield


app = FastAPI(title="Север ТСЛ — тестирование", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


@app.middleware("http")
async def private_responses(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


def user(request: Request, *, mutation=False):
    owner = request.headers.get("x-sever-user", "")
    if not owner or len(owner) > 128 or not re.fullmatch(r"[A-Za-z0-9_.@-]+", owner):
        raise HTTPException(401, "Войдите в тестовый кабинет.")
    if mutation:
        if request.headers.get("x-requested-with") != "SeverPortal":
            raise HTTPException(403, "Обновите страницу и повторите действие.")
        origin = request.headers.get("origin")
        if origin:
            expected = f"{request.headers.get('x-forwarded-proto', request.url.scheme)}://{request.headers.get('host', '')}"
            if origin.rstrip("/") != expected:
                raise HTTPException(403, "Запрос с другого сайта запрещён.")
    return owner


def owned(identifier, owner):
    with database() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (identifier, owner)).fetchone()
    if row is None:
        raise HTTPException(404, "Задание не найдено.")
    return dict(row)


def job_file(job_id, name):
    try:
        if str(uuid.UUID(job_id)) != job_id:
            return None
    except (ValueError, TypeError, AttributeError):
        return None
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "\\" in name or any(p in {".", ".."} for p in parts):
        return None
    root = RUNS.resolve()
    path = root / job_id
    if path.is_symlink():
        return None
    for part in parts:
        path = path / part
        if path.is_symlink():
            return None
    if not path.resolve().is_relative_to(root / job_id) or not path.is_file():
        return None
    return path


def artifact(job_id, name):
    parts = PurePosixPath(name).parts
    if name not in FILES and not (len(parts) > 1 and parts[0] in IMAGE_DIRS and
                                 PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS):
        return None
    return job_file(job_id, name)


def video_coverage(row):
    path = job_file(row.get("job_id"), "video/video-analysis.json")
    return load_video_coverage(path) if path else summarize_video_coverage({})


def coverage_warning(row):
    """Disclose candidate limits and uncertain excluded views, never field accuracy."""
    return video_coverage(row)["warning"]


def public_job(row):
    with database() as db:
        notes = [dict(note) for note in db.execute(
            "SELECT id,created_at,page_id,field,comment,proposed_value FROM feedback WHERE portal_id=? ORDER BY created_at", (row["id"],))]
    result = {key: row[key] for key in ("id", "job_id", "name", "status", "progress", "created_at", "updated_at", "size_bytes", "error")}
    for key, filename in (("report_url", "report.html"), ("csv_url", "documents.csv"), ("results_url", "results.json")):
        result[key] = f"/jobs/{row['job_id']}/{filename}" if row["job_id"] and artifact(row["job_id"], filename) else None
    # Results finished before the per-document export keep their per-sheet table.
    if not result["csv_url"] and row["job_id"] and artifact(row["job_id"], "fields.csv"):
        result["csv_url"] = f"/jobs/{row['job_id']}/fields.csv"
    if not result["results_url"] and row["job_id"] and artifact(row["job_id"], "partial-results.json"):
        result["results_url"] = f"/jobs/{row['job_id']}/partial-results.json"
    result["feedback"] = notes
    coverage = video_coverage(row)
    result["video_coverage"] = coverage
    warning = coverage["warning"]
    result["coverage_warning"] = warning
    if warning:
        if result["status"] == "completed":
            if coverage["candidate_limit"]:
                result["status"] = "completed_with_errors"
            result["progress"] = ("Обработка завершена с неполным охватом" if coverage["candidate_limit"]
                                  else "Обработка завершена; охват требует проверки")
        if coverage["candidate_limit"]:
            result["error"] = ((result["error"] + " ") if result["error"] else "") + warning
    return result


async def backend(method, path, **kwargs):
    async with httpx.AsyncClient(base_url=API_URL, timeout=httpx.Timeout(15, connect=5), trust_env=False) as client:
        return await client.request(method, path, **kwargs)


def translated_progress(status, message):
    if status in TERMINAL:
        return {"completed": "Обработка завершена", "completed_with_errors": "Обработка завершена с ошибками",
                "failed": "Обработка остановлена"}[status]
    if status == "queued":
        return "Видео в очереди"
    if "Selecting stable page frames" in message:
        return "Выбираем кадры с документами"
    match = re.search(r"image (\d+)/(\d+)", message)
    if match:
        action = "Уточняем" if message.startswith("Reviewing") else "Распознаём"
        return f"{action} лист {match[1]} из {match[2]}"
    return {"processing": "Обработка видео продолжается", "completed": "Обработка завершена",
            "completed_with_errors": "Обработка завершена с ошибками", "failed": "Обработка остановлена"}.get(status, "")


async def refresh(row):
    if row["status"] not in {"queued", "processing"} or not row["job_id"]:
        return row
    try:
        response = await backend("GET", f"/v1/jobs/{row['job_id']}")
        response.raise_for_status()
        state = response.json()
        status = state["status"]
        if status not in {"queued", "processing", *TERMINAL}:
            return row
        error = None
        if status == "failed":
            error = ("Достигнут лимит обработки. Частичные результаты сохранены; сообщите оператору."
                     if state.get("error_type") == "BatchBudgetExceeded" else
                     "Обработка прервана. Доступные частичные результаты сохранены; сообщите оператору.")
        progress = translated_progress(status, str(state.get("progress", "")))
        if (status, progress, error) != (row["status"], row["progress"], row["error"]):
            update(row["id"], status=status, progress=progress, error=error)
            row = owned(row["id"], row["owner"])
    except (httpx.HTTPError, ValueError, KeyError):
        row = {**row, "progress": "Связь с обработчиком временно недоступна. Задание не перезапускалось."}
    return row


async def refresh_active():
    with database() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE status IN ('queued','processing')")]
    await asyncio.gather(*(refresh(row) for row in rows))


@app.get("/")
def index(request: Request):
    user(request)
    return FileResponse(Path(__file__).parent / "portal_static" / "index.html", media_type="text/html")


@app.get("/portal-api/me")
def me(request: Request):
    return {"user": user(request), "limits": {"max_upload_bytes": MAX_BYTES, "max_duration_seconds": MAX_DURATION,
            "max_active_jobs": MAX_ACTIVE, "max_team_active_jobs": MAX_TEAM_ACTIVE}}


@app.get("/portal-api/jobs")
async def jobs(request: Request):
    owner = user(request)
    with database() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE owner=? ORDER BY created_at DESC", (owner,))]
    rows = await asyncio.gather(*(refresh(row) for row in rows))
    return {"jobs": [public_job(row) for row in rows]}


@app.get("/portal-api/jobs/{identifier}")
async def job(identifier: str, request: Request):
    return public_job(await refresh(owned(identifier, user(request))))


async def probe_video(path):
    process = await asyncio.create_subprocess_exec("ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-format_whitelist", "mov,matroska,avi", "-show_entries",
        "format=duration,format_name:stream=codec_type,width,height,duration", "-of", "json", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=45)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ValueError("Не удалось открыть видео. Поддерживаются MP4, MOV, M4V, MKV, AVI и WebM.")
    try:
        data = json.loads(output)
        formats = set(data.get("format", {}).get("format_name", "").split(","))
        videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video" and s.get("width", 0) > 0 and s.get("height", 0) > 0]
        duration = float(data.get("format", {}).get("duration", 0))
        if not formats.intersection({"mov", "matroska", "avi"}) or not videos or not (0 < duration <= MAX_DURATION):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise ValueError("Видео должно содержать изображение и длиться не больше 60 минут.") from None
    return duration


def reserve(owner, name):
    identifier = str(uuid.uuid4())
    folder = DATA / "uploads"
    folder.mkdir(parents=True, exist_ok=True, mode=0o750)
    source = folder / (identifier + Path(name).suffix.lower())
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in ACTIVE)
        active = db.execute(f"SELECT owner FROM jobs WHERE status IN ({placeholders})", ACTIVE).fetchall()
        if len(active) >= MAX_ACTIVE or sum(row["owner"] == owner for row in active) >= MAX_TEAM_ACTIVE:
            raise HTTPException(429, "Очередь заполнена. Дождитесь завершения текущих видео.")
        timestamp = now()
        db.execute("INSERT INTO jobs (id,owner,name,source,status,progress,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                   (identifier, owner, name, str(source), "uploading", "Загружаем видео", timestamp, timestamp))
    return identifier, source


@app.post("/portal-api/uploads", status_code=202)
async def upload(request: Request, filename: str):
    owner = user(request, mutation=True)
    name = filename.strip()
    if not name or len(name) > 240 or "/" in name or "\\" in name or any(ord(c) < 32 for c in name) or Path(name).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, "Выберите видео MP4, MOV, M4V, MKV, AVI или WebM с обычным именем файла.")
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise HTTPException(400, "Некорректный размер загрузки.") from None
    if length < 0 or length > MAX_BYTES:
        raise HTTPException(413, "Максимальный размер видео — 4 ГиБ.")
    await refresh_active()
    identifier, source = reserve(owner, name)
    size = 0
    submitted = False
    try:
        if shutil.disk_usage(DATA).free < max(length, MIN_FREE_BYTES) + MIN_FREE_BYTES:
            raise HTTPException(507, "На сервере недостаточно места для этого видео. Сообщите оператору.")
        with source.open("xb") as stream:
            async with asyncio.timeout(7200):
                chunks = request.stream().__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(chunks.__anext__(), timeout=120)
                    except StopAsyncIteration:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Максимальный размер видео — 4 ГиБ.")
                    if shutil.disk_usage(DATA).free < len(chunk) + MIN_FREE_BYTES:
                        raise HTTPException(507, "На сервере недостаточно места. Сообщите оператору.")
                    stream.write(chunk)
        if not size:
            raise HTTPException(400, "Загружен пустой файл.")
        update(identifier, status="validating", progress="Проверяем видео", size_bytes=size)
        try:
            await probe_video(source)
        except (ValueError, asyncio.TimeoutError) as exc:
            raise HTTPException(400, str(exc) or "Проверка видео заняла слишком много времени.") from None
        health = await backend("GET", "/health")
        health.raise_for_status()
        config = health.json()
        if config.get("model_backend") != "local" or config.get("document_reader") != "native_image" or not config.get("model_name"):
            raise HTTPException(503, "Локальная модель временно недоступна. Сообщите оператору.")
        payload = {"source_path": str(source.resolve()), "model": config["model_name"], "max_requests": 4000,
                   "max_tokens": 40000000, "max_pages": 300, "sample_interval": .5, "refresh_cache": True}
        fingerprint = config.get("native_implementation", {}).get("fingerprint_sha256")
        if fingerprint:
            payload["native_implementation_fingerprint"] = fingerprint
        update(identifier, status="submitting", progress="Передаём видео на обработку")
        submitted = True
        response = await backend("POST", "/v1/jobs", json=payload)
        if response.status_code in {400, 409, 422, 423, 503}:
            submitted = False
            raise HTTPException(503, "Обработчик не принял видео. Сообщите оператору; автоматического повторного запуска нет.")
        response.raise_for_status()
        job_id = response.json()["job_id"]
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError("Invalid job identity")
        update(identifier, job_id=job_id, status="queued", progress="Видео в очереди", error=None)
        return public_job(owned(identifier, owner))
    except BaseException as exc:
        service_unavailable = isinstance(exc, (httpx.HTTPError, FileNotFoundError))
        error = ("Не удалось подтвердить запуск. Нужна проверка оператором; повторный запуск не выполнялся."
                 if submitted else exc.detail if isinstance(exc, HTTPException) else
                 "Обработчик временно недоступен. Видео не запущено; повторите загрузку позже."
                 if service_unavailable else "Загрузка или проверка видео прервана. Попробуйте снова.")
        update(identifier, status="submission_uncertain" if submitted else "failed", error=error, size_bytes=size,
               progress="Требуется проверка оператором" if submitted else "Видео не запущено")
        if not submitted:
            source.unlink(missing_ok=True)
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(503 if submitted or service_unavailable else 400, error) from None


class Feedback(BaseModel):
    page_id: str | None = Field(default=None, max_length=160)
    field: str | None = Field(default=None, max_length=160)
    comment: str = Field(min_length=1, max_length=4000)
    proposed_value: str | None = Field(default=None, max_length=4000)


@app.post("/portal-api/jobs/{identifier}/feedback", status_code=201)
def feedback(identifier: str, note: Feedback, request: Request):
    owned(identifier, user(request, mutation=True))
    if not note.comment.strip():
        raise HTTPException(422, "Опишите замечание.")
    result = {"id": str(uuid.uuid4()), "created_at": now(), **note.model_dump()}
    with database() as db:
        db.execute("INSERT INTO feedback (id,portal_id,created_at,page_id,field,comment,proposed_value) VALUES (?,?,?,?,?,?,?)",
                   (result["id"], identifier, result["created_at"], note.page_id, note.field, note.comment, note.proposed_value))
    return result


@app.get("/jobs/{job_id}/{path:path}")
async def download(job_id: str, path: str, request: Request):
    owner = user(request)
    with database() as db:
        row = db.execute("SELECT * FROM jobs WHERE job_id=? AND owner=?", (job_id, owner)).fetchone()
    target = artifact(job_id, path) if row else None
    if target is None:
        raise HTTPException(404, "Файл не найден.")
    if path == "report.html":
        row = await refresh(dict(row))
        # Presentation-only wrapper: original report and recognition values stay intact.
        content = target.read_text(encoding="utf-8")
        content = re.sub(r'\s*·?\s*<a href="trace\.jsonl">[^<]*</a>', "", content)
        banner = '<p><a href="/">← Вернуться в кабинет и оставить замечание</a></p>'
        if row["status"] == "failed":
            content = re.sub(r'<meta http-equiv="refresh"[^>]*>', "", content)
            content = content.replace("Север ТСЛ — обработка продолжается", "Север ТСЛ — обработка остановлена")
            content = content.replace("Показаны уже полученные чтения; обработка продолжается, значения ещё могут уточняться.",
                                      "Показаны чтения, полученные до остановки обработки.")
            banner += '<p role="alert"><strong>Обработка остановлена. Это частичный результат.</strong> Исходные чтения сохранены без изменений.</p>'
        coverage = video_coverage(row)
        warning = coverage["warning"]
        if warning and 'id="video-coverage-warning"' not in content:
            # The summary contains only fixed text and validated numeric timestamps.
            heading = "Видео обработано не полностью." if coverage["candidate_limit"] else "Охват видео требует проверки."
            banner += '<p role="alert"><strong>' + heading + '</strong> ' + warning + '</p>'
        content = re.sub(r'(<h1[^>]*>.*?</h1>)', lambda match: match.group(1) + banner, content, count=1, flags=re.S)
        return HTMLResponse(content)
    return FileResponse(target, filename=target.name if target.suffix == ".csv" else None)
