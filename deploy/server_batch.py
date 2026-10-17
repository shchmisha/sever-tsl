"""Drive real, sequential API jobs with a persistent shared batch allowance.

This runner neither reads ground truth nor calls a model directly. Recognition
is performed by the configured API worker; completion is not measured accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import ipaddress
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import uuid

TERMINAL = {"completed", "completed_with_errors", "failed"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
# These events all reuse an existing model response, even if its local
# validation or JSON parsing is repeated. OCR/registry caches are independent.
HISTORICAL_MODEL_RESPONSE_EVENTS = {
    "cache_hit", "classifier_cache_hit", "classifier_cache_revalidated",
    "vision_cache_hit", "vision_cache_revalidated", "vision_paid_parse_recovered",
    "image_reader_cache_hit",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def sources_from(source_dir, provenance=None):
    """Bind exact downloaded filenames and hashes; unrelated demo clips stay out."""
    source_dir = Path(source_dir).resolve()
    expected = {}
    if provenance:
        data = json.loads(Path(provenance).read_text())
        files = data["files"]
        names = [row["name"] for row in files]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate download provenance filenames")
        for row in files:
            name = row["name"]
            if Path(name).name != name or Path(name).suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("Download provenance must contain video basenames")
            expected[name] = row
        paths = [source_dir / name for name in sorted(expected)]
    else:
        paths = sorted(path for path in source_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise ValueError("No videos selected")
    result = []
    for path in paths:
        if not path.is_file() or path.resolve().parent != source_dir:
            raise ValueError("Missing video or file outside the input directory")
        sha = digest(path)
        row = expected.get(path.name)
        if row and (sha != row["sha256"] or path.stat().st_size != row["size"]):
            raise ValueError("Input does not match download provenance: " + path.name)
        result.append({"name": path.name, "path": str(path.resolve()),
                       "bytes": path.stat().st_size, "sha256": sha})
    return result


def prioritize_sources(sources, first_sources):
    """Reorder inventory records only; priority names are never inference input."""
    names = list(first_sources)
    if any(not isinstance(name, str) or not name or Path(name).name != name
           for name in names):
        raise ValueError("First sources must be exact selected video basenames")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate first-source filenames")
    inventory = {source["name"]: source for source in sources}
    missing = [name for name in names if name not in inventory]
    if missing:
        raise ValueError("First source is not in the selected inventory: " + ", ".join(missing))
    priorities = set(names)
    return [inventory[name] for name in names] + [source for source in sources
                                               if source["name"] not in priorities]


class APIError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class APIClient:
    def __init__(self, base_url):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("API URL must be HTTP(S), without embedded credentials")
        self.base_url = base_url.rstrip("/")

    def request(self, method, endpoint, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + endpoint, data=body, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            # Never persist raw response bodies, which may contain credentials.
            raise APIError("API HTTP " + str(exc.code), exc.code) from None
        except (OSError, ValueError) as exc:
            raise APIError("API transport/JSON error: " + type(exc).__name__) from None


def job_ids(run_dir):
    identifiers = set()
    for child in Path(run_dir).iterdir():
        if not child.is_dir():
            continue
        try:
            uuid.UUID(child.name)
        except ValueError:
            continue
        identifiers.add(child.name)
    return identifiers


def loopback_endpoint(value):
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
            return False
        return hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
    except (ValueError, TypeError):
        return False


def trace_observations(path):
    rows, invalid = [], 0
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    invalid += 1
            except ValueError:
                invalid += 1
    starts = [i for i, row in enumerate(rows) if row.get("event") == "run_started"]
    current = rows[starts[-1]:] if starts else rows
    events = Counter(row.get("event", "") for row in current)
    responses = [row for row in current if row.get("event") == "model_response_received"]
    attempts = max(events["model_request_started"],
                   max((int(row.get("batch_attempted_requests", 0)) for row in current
                        if row.get("event") == "model_request_started"), default=0))
    usage = {"attempted_requests": attempts, "requests": len(responses),
             "input_tokens": sum(int(row.get("input_tokens", 0)) for row in responses),
             "output_tokens": sum(int(row.get("output_tokens", 0)) for row in responses),
             "actual_cost_rub": 0.0, "costed_requests": 0}
    local_responses = 0
    verified_local_responses = 0
    backends = set()
    for row in responses:
        billing = row.get("billing", {})
        value = billing.get("value")
        endpoint = row.get("model_base_url")
        if endpoint:
            backends.add(str(endpoint))
        if billing.get("kind") == "local_inference":
            local_responses += 1
            if (row.get("provider_kind") == "local" and loopback_endpoint(endpoint)
                    and value == 0 and billing.get("currency") is None):
                verified_local_responses += 1
        if (billing.get("kind") == "provider_reported" and billing.get("currency") == "RUB"
                and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            usage["actual_cost_rub"] += value
            usage["costed_requests"] += 1
    return {"exists": Path(path).is_file(), "sha256": digest(path) if Path(path).is_file() else None,
            "run_started_count": len(starts), "run_completed": bool(events["run_completed"]),
            "initial_run": current[0] if starts else None, "event_counts": dict(events),
            "historical_model_cache_hits": sum(count for event, count in events.items()
                if event in HISTORICAL_MODEL_RESPONSE_EVENTS
                or "cache_hit" in event and event.startswith(("classifier", "vision", "model", "agent", "llm"))),
            "invalid_trace_lines": invalid, "usage": usage,
            "interrupted_trace_fragments": sum(row.get("event") == "trace_interrupted_tail_preserved" for row in rows),
            "local_inference_responses": local_responses,
            "verified_local_responses": verified_local_responses,
            "model_backends": sorted(backends),
            "nonlocal_or_unverified_responses": len(responses) - verified_local_responses,
            "requests_without_response": max(0, attempts - len(responses)),
            "responses_without_reported_cost": len(responses) - usage["costed_requests"] - local_responses}


def total_usage(jobs):
    totals = {"attempted_requests": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0,
              "actual_cost_rub": 0.0, "costed_requests": 0,
              "reserved_requests": 0, "reserved_tokens": 0,
              "requests_without_response": 0, "responses_without_reported_cost": 0,
              "local_inference_responses": 0}
    for job in jobs:
        usage = job.get("usage", {})
        for name in ("attempted_requests", "requests", "input_tokens", "output_tokens", "actual_cost_rub", "costed_requests"):
            totals[name] += usage.get(name, 0) or 0
        proof = job.get("proof", {})
        missing = proof.get("requests_without_response", 0)
        totals["requests_without_response"] += missing
        totals["responses_without_reported_cost"] += proof.get("responses_without_reported_cost", 0)
        totals["local_inference_responses"] += proof.get("local_inference_responses", 0)
        if missing:
            # Unknown billed tokens cannot be called zero. Conservatively reserve
            # this job's entire quota rather than spending its unseen remainder.
            allocation = job.get("request", {})
            totals["reserved_requests"] += max(0, allocation.get("max_requests", 0) - usage.get("attempted_requests", 0))
            totals["reserved_tokens"] += max(0, allocation.get("max_tokens", 0) - usage.get("input_tokens", 0) - usage.get("output_tokens", 0))
    totals["budget_requests"] = totals["attempted_requests"] + totals["reserved_requests"]
    totals["budget_tokens"] = totals["input_tokens"] + totals["output_tokens"] + totals["reserved_tokens"]
    totals["currency"] = "RUB"
    return totals


def cache_inventory(run_dir):
    return {name: {"json_files": sum(1 for _ in (Path(run_dir) / name).glob("*.json")),
                   "directory_exists": (Path(run_dir) / name).is_dir()}
            for name in ("llm-cache", "vision-cache", "inn-cache")}


def implementation_binding(client, *, allow_legacy_unavailable=False):
    """Accept an advertised native hash, or explicitly mark legacy health unpinned."""
    try:
        health = client.request("GET", "/health")
    except APIError as exc:
        if exc.status not in {404, 405} and not allow_legacy_unavailable:
            raise
        return {"status": "unpinned", "reason": "API health fingerprint is unavailable or unsupported"}
    native = health.get("native_implementation") if isinstance(health, dict) else None
    fingerprint = native.get("fingerprint_sha256") if isinstance(native, dict) else None
    if fingerprint is None:
        return {"status": "unpinned", "reason": "API health does not advertise a native implementation fingerprint"}
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)):
        raise ValueError("API health advertises an invalid native implementation fingerprint")
    # Never persist arbitrary health fields (or a provider's credentials).
    return {"status": "pinned", "fingerprint_sha256": fingerprint}


class BatchDriver:
    def __init__(self, args, client=None, sleep=time.sleep):
        self.args = args
        self.model = getattr(args, "model", "openai/gpt-4.1-mini")
        self.provider_kind = getattr(args, "provider_kind", "remote")
        self.job_max_tokens = getattr(args, "job_max_tokens", 500000)
        self.job_max_requests = getattr(args, "job_max_requests", 400)
        if (not self.model.strip() or self.provider_kind not in {"remote", "local"}
                or not 1000 <= self.job_max_tokens <= 40000000
                or self.provider_kind == "remote" and self.job_max_tokens > 500000):
            raise ValueError("A model and valid provider/token cap are required; remote jobs allow at most 500000 tokens")
        if (not 1 <= self.job_max_requests <= 4000
                or self.provider_kind == "remote" and self.job_max_requests > 400):
            raise ValueError("Valid per-job request cap required; local jobs allow at most 4000 requests and remote jobs at most 400 requests")
        self.run_dir = Path(args.run_dir).resolve()
        self.batch_dir = Path(args.batch_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.batch_dir / "manifest.json"
        self.client = client or APIClient(args.base_url)
        self.sleep = sleep
        first_sources = list(getattr(args, "first_source", []))
        sources = prioritize_sources(sources_from(args.source_dir, args.download_provenance), first_sources)
        settings = {"base_url": args.base_url.rstrip("/"), "max_requests": args.max_requests,
                    "max_tokens": args.max_tokens, "sample_interval": args.interval,
                    "model": self.model, "provider_kind": self.provider_kind, "job_max_tokens": self.job_max_tokens,
                    "job_max_requests": self.job_max_requests,
                    "run_dir": str(self.run_dir), "source_dir": str(Path(args.source_dir).resolve()),
                    "first_sources": first_sources, "source_order": [source["name"] for source in sources]}
        if self.manifest_path.exists():
            self.state = json.loads(self.manifest_path.read_text())
            # Old manifests were always the default remote model and 500k cap.
            previous = self.state["settings"]
            previous.setdefault("model", "openai/gpt-4.1-mini")
            previous.setdefault("provider_kind", "remote")
            previous.setdefault("job_max_tokens", 500000)
            previous.setdefault("job_max_requests", 400)
            # Before priority scheduling, the saved source list was the order.
            previous.setdefault("first_sources", [])
            previous.setdefault("source_order", [source["name"] for source in self.state["sources"]])
            if self.state["settings"] != settings or self.state["sources"] != sources:
                raise ValueError("Existing batch settings or input hashes differ; use a new batch directory")
        else:
            self.state = {"schema_version": 1, "created_at": now(), "status": "pending",
                          "settings": settings, "sources": sources, "jobs": [],
                          "download_provenance": {"path": str(Path(args.download_provenance).resolve()),
                              "sha256": digest(args.download_provenance)} if args.download_provenance else None,
                          "accuracy_evaluated": False, "customer_approved": False,
                          "limits_note": "Shared request/token ceiling across sequential API jobs; no per-leaf frame cap. Unknown response usage reserves its job quota.",
                          "limitations": ["API completion and recognized fields do not establish value accuracy or contractual acceptance.",
                                          "No ground truth is read by this driver or supplied to inference.",
                                          "Provider prompt caching is separate from local historical response-cache hits."]}
        self.check_implementation()
        self.persist()

    def check_implementation(self):
        current = implementation_binding(self.client, allow_legacy_unavailable=self.provider_kind == "remote")
        saved = self.state.get("native_implementation_binding", {"status": "unpinned"})
        if saved.get("status") == "pinned":
            fingerprint = saved.get("fingerprint_sha256")
            if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                    or any(character not in "0123456789abcdef" for character in fingerprint)):
                raise ValueError("Stored native implementation pin is invalid; use a new batch directory")
            if current.get("fingerprint_sha256") != saved.get("fingerprint_sha256"):
                raise ValueError("API native implementation changed or its fingerprint is unavailable; use a new batch directory")
        elif current["status"] == "pinned":
            if self.state["jobs"]:
                raise ValueError("Existing batch has no native implementation pin; use a new batch directory")
            self.state["native_implementation_binding"] = current
        else:
            self.state["native_implementation_binding"] = current

    def persist(self):
        self.state["updated_at"] = now()
        self.state["usage"] = total_usage(self.state["jobs"])
        counts = Counter(job["status"] for job in self.state["jobs"])
        self.state["counts"] = {"selected_videos": len(self.state["sources"]), "submitted_jobs": len(self.state["jobs"]),
                                "statuses": dict(counts), "remaining_videos": len(self.state["sources"]) - len(self.state["jobs"])}
        save(self.manifest_path, self.state)
        self.write_index()

    def write_index(self):
        def link(path, label):
            path = Path(path)
            if not path.is_file():
                return ""
            try:
                relative = path.resolve().relative_to(self.run_dir)
            except ValueError:
                return ""
            url = "/".join(quote(part, safe="") for part in relative.parts)
            return '<a href="' + html.escape(url, quote=True) + '">' + html.escape(label) + '</a>'
        rows = []
        by_name = {job["source"]["name"]: job for job in self.state["jobs"]}
        for source in self.state["sources"]:
            job = by_name.get(source["name"], {})
            summary = job.get("summary", {})
            artifacts = []
            if job.get("job_id"):
                directory = self.run_dir / job["job_id"]
                for filename, label in (("report.html", "Кадры и поля"), ("results.json", "JSON"),
                                        ("fields.csv", "CSV"), ("trace.jsonl", "Трасса"), ("job.json", "Задание")):
                    value = link(directory / filename, label)
                    if value:
                        artifacts.append(value)
            proof = link(self.batch_dir / "proof" / (job.get("job_id", "") + ".json"), "Проверка происхождения")
            if proof:
                artifacts.append(proof)
            values = [source["name"], job.get("status", "pending"), job.get("progress", ""),
                      str(summary.get("page_count", "")),
                      json.dumps(summary.get("types", {}), ensure_ascii=False),
                      str(summary.get("recognized_fields", "")), str(summary.get("unresolved_fields", "")),
                      str(summary.get("pages_with_errors", ""))]
            rows.append("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in values)
                        + "<td>" + " · ".join(artifacts) + "</td></tr>")
        usage = self.state["usage"]
        backends = sorted({endpoint for job in self.state["jobs"]
                           for endpoint in job.get("proof", {}).get("model_backends", [])})
        backend_note = ('<p>Модель: ' + html.escape(self.model) + '; обработка: '
                        + html.escape(self.provider_kind) + '; адреса из ответов модели: '
                        + html.escape(", ".join(backends) or "ещё нет ответов") + '.</p>')
        if self.provider_kind == "local":
            backend_note += ('<p>Локальных ответов модели: ' + str(usage["local_inference_responses"])
                             + '. Оплата внешнему API за подтверждённый локальный вывод: 0. Стоимость аренды сервера и GPU сюда не входит.</p>')
        content = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">'
                   '<meta http-equiv="refresh" content="15"><title>Север ТСЛ — серверный прогон</title>'
                   '<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse;width:100%}'
                   'td,th{border:1px solid #ddd;padding:8px;text-align:left}td{vertical-align:top}'
                   '.scroll{overflow:auto}a{color:#145dad}</style><body>'
                   '<h1>Север ТСЛ — серверный прогон</h1><p>Состояние: <strong>'
                   + html.escape(self.state["status"]) + '</strong>. Выбрано видео: '
                   + str(len(self.state["sources"])) + '. Обновлено: ' + html.escape(self.state["updated_at"]) + '</p>'
                   + backend_note
                   + '<p>Здесь показаны фактические задания API. Количество распознанных полей не означает проверенную точность. '
                   'Эталонные значения в обработку не передаются.</p><p>Попытки запросов: '
                   + str(usage["attempted_requests"]) + '; входные / выходные токены: '
                   + str(usage["input_tokens"]) + ' / ' + str(usage["output_tokens"])
                   + '; стоимость по ответам провайдера: ' + format(usage["actual_cost_rub"], ".4f")
                   + ' ₽; попытки без ответа: ' + str(usage["requests_without_response"])
                   + '; ответы без подтверждённой стоимости: ' + str(usage["responses_without_reported_cost"]) + '.</p>'
                   + '<p>' + link(self.manifest_path, "Манифест и общий бюджет") + '</p>'
                   + '<div class="scroll"><table><thead><tr><th>Видео</th><th>Статус</th><th>Текущий шаг</th>'
                   '<th>Листы</th><th>Типы</th><th>Распознано полей</th><th>Не разрешено</th><th>Листы с ошибками</th>'
                   '<th>Артефакты</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></body></html>')
        target = self.run_dir / "index.html"
        temporary = target.with_suffix(".html.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)

    def recover_submission(self, job):
        candidates = []
        previous = set(job.get("preexisting_job_ids", []))
        for identifier in job_ids(self.run_dir) - previous:
            path = self.run_dir / identifier / "job.json"
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            source_path = saved.get("source_path")
            # Completed/failed API states currently omit source_path. A driver
            # that crashed after POST may resume after the worker has finished.
            if not source_path:
                result_path = self.run_dir / identifier / "results.json"
                try:
                    source_path = json.loads(result_path.read_text()).get("source", {}).get("path")
                except (OSError, ValueError):
                    initial = trace_observations(self.run_dir / identifier / "trace.jsonl").get("initial_run") or {}
                    source_path = initial.get("source")
            if source_path == job["source"]["path"]:
                candidates.append(identifier)
        if len(candidates) == 1:
            job["job_id"] = candidates[0]
            job["status"] = "queued"
            job["submission_recovered_from_server_state"] = True
            self.persist()
            return True
        job["status"] = "submission_unknown"
        job["submission_candidates"] = candidates
        self.state["status"] = "needs_attention"
        self.persist()
        return False

    def observe(self, job, api_state=None, result=None):
        directory = self.run_dir / job["job_id"]
        proof = trace_observations(directory / "trace.jsonl")
        proof.update(job_id=job["job_id"], source_sha256=job["source"]["sha256"],
                     requested_model=self.model, expected_provider_kind=self.provider_kind,
                     submitted_refresh_cache=job["request"]["refresh_cache"],
                     initial_cache_inventory=job["initial_cache_inventory"],
                     job_directory_preexisted=job["job_id"] in job["preexisting_job_ids"])
        if result is not None:
            bound = result.get("source", {}).get("sha256") == job["source"]["sha256"]
            proof["result_source_sha256_matches"] = bound
            if not bound:
                raise ValueError("API result belongs to a different source hash")
            proof["source_sha256_after"] = digest(job["source"]["path"])
            proof["source_unchanged"] = proof["source_sha256_after"] == job["source"]["sha256"]
            job["usage"] = result.get("usage", proof["usage"])
            pages = result.get("pages", [])
            fields = [value for page in pages for value in page.get("result", {}).get("fields", {}).values()]
            job["summary"] = {"page_count": len(pages),
                "types": dict(Counter(page.get("result", {}).get("document_type", "unknown") for page in pages)),
                "recognized_fields": sum(value.get("status") == "recognized" for value in fields),
                "unresolved_fields": sum(value.get("status") != "recognized" for value in fields),
                "pages_with_errors": sum(bool(page.get("result", {}).get("errors")) for page in pages),
                "errors": [{"page_id": page.get("page_id"), "errors": page["result"]["errors"]}
                           for page in pages if page.get("result", {}).get("errors")],
                "elapsed_seconds": result.get("elapsed_seconds"), "accuracy_evaluated": False}
        elif api_state and api_state.get("usage"):
            job["usage"] = api_state["usage"]
        else:
            job["usage"] = proof["usage"]
        for filename in ("results.json", "fields.csv", "report.html", "selected-pages.json", "classifications.json"):
            path = directory / filename
            proof.setdefault("artifacts", {})[filename] = {"exists": path.is_file(),
                "sha256": digest(path) if path.is_file() else None}
        if result is not None:
            issues = []
            if not proof["exists"] or proof["run_started_count"] != 1 or not proof["run_completed"]:
                issues.append("Missing or incomplete single-run trace")
            if proof["invalid_trace_lines"]:
                issues.append("Trace contains malformed or non-object JSON lines")
            if proof["interrupted_trace_fragments"]:
                issues.append("Trace contains preserved interrupted fragments")
            initial = proof.get("initial_run") or {}
            if initial.get("source_id") != job["source"]["sha256"]:
                issues.append("Trace source hash differs or is missing")
            if initial.get("previous_artifacts") or proof["job_directory_preexisted"]:
                issues.append("Job directory contains a previous run")
            if proof["historical_model_cache_hits"]:
                issues.append("Historical model response cache was used")
            if self.provider_kind == "local" and proof["nonlocal_or_unverified_responses"]:
                issues.append("Local batch contains a remote or unverified model response")
            proof["local_provider_verified"] = (proof["verified_local_responses"] == proof["usage"]["requests"]
                                                 if proof["usage"]["requests"] else None)
            if not proof["source_unchanged"]:
                issues.append("Source file changed during the job")
            for name in ("results.json", "fields.csv", "report.html"):
                if not proof["artifacts"][name]["exists"]:
                    issues.append("Missing artifact: " + name)
            proof["verification_issues"] = issues
            proof["fresh_run_verified"] = not issues
        job["proof"] = proof
        save(self.batch_dir / "proof" / (job["job_id"] + ".json"), proof)

    def poll(self, job):
        deadline = time.monotonic() + self.args.job_timeout
        failures = 0
        last_progress = None
        while time.monotonic() < deadline:
            try:
                status = self.client.request("GET", "/v1/jobs/" + job["job_id"])
                failures = 0
            except APIError as exc:
                failures += 1
                job["poll_error"] = str(exc)
                self.state["status"] = "waiting_for_api"
                self.persist()
                if failures >= 6:
                    return False
                self.sleep(self.args.poll_seconds)
                continue
            job["status"] = status["status"]
            job["progress"] = status.get("progress", "")
            job["api_state"] = status
            self.observe(job, status)
            self.state["status"] = "processing"
            self.persist()
            current = (job["status"], job["progress"])
            if current != last_progress:
                print(json.dumps({"source": job["source"]["name"], "job_id": job["job_id"],
                                  "status": job["status"], "progress": job["progress"]}, ensure_ascii=False), flush=True)
                last_progress = current
            if job["status"] in TERMINAL:
                if job["status"].startswith("completed"):
                    result = self.client.request("GET", "/v1/jobs/" + job["job_id"] + "/results")
                    self.observe(job, status, result)
                job["finalized"] = True
                job["finished_at"] = now()
                self.persist()
                return True
            self.sleep(self.args.poll_seconds)
        self.state["status"] = "waiting_for_job"
        job["poll_error"] = "Polling timeout; existing job retained and will not be submitted again"
        self.persist()
        return False

    def run(self):
        self.check_implementation()
        self.state["status"] = "processing"
        self.persist()
        for source in self.state["sources"]:
            job = next((row for row in self.state["jobs"] if row["source"] == source), None)
            if job and job["status"] in TERMINAL and job.get("finalized"):
                if self.provider_kind == "local" and job.get("proof", {}).get("nonlocal_or_unverified_responses"):
                    self.state["status"] = "provider_verification_failed"
                    self.persist()
                    return self.state
                if job.get("api_state", {}).get("error_type") == "FatalProviderError":
                    self.state["status"] = "provider_failure"
                    self.persist()
                    return self.state
                continue
            if job and not job.get("job_id"):
                if not self.recover_submission(job):
                    return self.state
            if job is None:
                usage = total_usage(self.state["jobs"])
                remaining_requests = self.args.max_requests - usage["budget_requests"]
                remaining_tokens = self.args.max_tokens - usage["budget_tokens"]
                if remaining_requests <= 0 or remaining_tokens < 1000:
                    self.state["status"] = "budget_exhausted"
                    self.persist()
                    return self.state
                if digest(source["path"]) != source["sha256"]:
                    raise ValueError("Video changed after manifest creation")
                self.check_implementation()
                request = {"source_path": source["path"], "model": self.model,
                           "sample_interval": self.args.interval, "max_pages": 300,
                           "max_requests": min(self.job_max_requests, remaining_requests),
                           "max_tokens": min(self.job_max_tokens, remaining_tokens), "refresh_cache": True}
                binding = self.state["native_implementation_binding"]
                if binding["status"] == "pinned":
                    request["native_implementation_fingerprint"] = binding["fingerprint_sha256"]
                job = {"source": source, "status": "submitting", "created_at": now(), "request": request,
                       "preexisting_job_ids": sorted(job_ids(self.run_dir)),
                       "initial_cache_inventory": cache_inventory(self.run_dir)}
                self.state["jobs"].append(job)
                # A restart during POST must recover the existing server job or
                # stop for attention, never blindly issue another paid job.
                self.persist()
                try:
                    response = self.client.request("POST", "/v1/jobs", request)
                    identifier = str(uuid.UUID(response["job_id"]))
                except APIError as exc:
                    job["submission_error"] = str(exc)
                    if not self.recover_submission(job):
                        return self.state
                else:
                    job["job_id"] = identifier
                    job["status"] = "queued"
                    self.persist()
            if not self.poll(job):
                return self.state
            if self.provider_kind == "local" and job.get("proof", {}).get("nonlocal_or_unverified_responses"):
                self.state["status"] = "provider_verification_failed"
                self.persist()
                return self.state
            if job.get("api_state", {}).get("error_type") == "FatalProviderError":
                self.state["status"] = "provider_failure"
                self.persist()
                return self.state
        statuses = {job["status"] for job in self.state["jobs"]}
        self.state["status"] = "completed" if statuses == {"completed"} else "completed_with_errors"
        self.persist()
        return self.state


@contextmanager
def exclusive_batch(batch_dir):
    Path(batch_dir).mkdir(parents=True, exist_ok=True)
    with (Path(batch_dir) / "driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This batch already has an active driver") from None
        yield


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/srv/sever/input"))
    parser.add_argument("--run-dir", type=Path, default=Path("/srv/sever/runs"))
    parser.add_argument("--batch-dir", type=Path, default=Path("/srv/sever/runs/batches/server-validation"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--download-provenance", type=Path)
    parser.add_argument("--first-source", action="append", default=[], metavar="FILENAME",
                        help="Process this exact selected video basename first; repeat for ordered priorities. All other selected videos remain.")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--provider-kind", choices=("remote", "local"), default="remote")
    parser.add_argument("--job-max-tokens", type=int, default=500000)
    parser.add_argument("--job-max-requests", type=int, default=400,
                        help="Per-video ceiling, also bounded by remaining shared requests; local maximum 4000, remote maximum 400")
    parser.add_argument("--max-requests", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900000)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--job-timeout", type=float, default=14400)
    args = parser.parse_args(argv)
    if args.max_requests < 1 or args.max_tokens < 1000 or not 0 < args.interval <= 5 or args.poll_seconds <= 0 or args.job_timeout <= 0:
        parser.error("Positive limits required; max_tokens >=1000 and interval <=5")
    if not args.model.strip() or not 1000 <= args.job_max_tokens <= 40000000:
        parser.error("A model and job-max-tokens between 1000 and 40000000 are required")
    if not 1 <= args.job_max_requests <= 4000:
        parser.error("job-max-requests must be between 1 and 4000")
    if args.provider_kind == "remote" and args.job_max_tokens > 500000:
        parser.error("Remote jobs support at most 500000 tokens; larger caps require --provider-kind local")
    if args.provider_kind == "remote" and args.job_max_requests > 400:
        parser.error("Remote jobs support at most 400 requests; larger caps require --provider-kind local")
    return args


def main():
    args = arguments()
    with exclusive_batch(args.batch_dir):
        driver = BatchDriver(args)
        try:
            state = driver.run()
        except BaseException as exc:
            driver.state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "driver_error"
            driver.state["driver_error_type"] = type(exc).__name__
            driver.persist()
            raise
        print(json.dumps({"status": state["status"], "counts": state["counts"], "usage": state["usage"],
                          "manifest": str(driver.manifest_path)}, ensure_ascii=False), flush=True)
        return 0 if state["status"] in {"completed", "completed_with_errors"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
