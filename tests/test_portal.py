"""Facade integration with offline transport. These do not measure model accuracy."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sever import portal

AUTH = {"X-Sever-User": "team-a"}
MUTATE = {**AUTH, "X-Requested-With": "SeverPortal"}


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(portal, "DATA", tmp_path / "customer")
    monkeypatch.setattr(portal, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(portal, "API_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(portal.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    calls = []
    states = {}

    async def probe(path):
        assert path.is_file()
        return 10

    async def backend(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/health":
            body = {"model_backend": "local", "document_reader": "native_image", "model_name": "local-test-model",
                    "native_implementation": {"fingerprint_sha256": "a" * 64}}
        elif method == "POST" and path == "/v1/jobs":
            identifier = str(uuid.uuid4())
            states[identifier] = {"status": "queued"}
            body = {"job_id": identifier, "status": "queued"}
        elif method == "GET" and path.startswith("/v1/jobs/"):
            body = states[path.rsplit("/", 1)[1]]
        else:
            pytest.fail(f"Unexpected offline backend request {method} {path}")
        return httpx.Response(202 if method == "POST" else 200, json=body, request=httpx.Request(method, "http://localhost" + path))

    monkeypatch.setattr(portal, "probe_video", probe)
    monkeypatch.setattr(portal, "backend", backend)
    with TestClient(portal.app) as client:
        yield client, calls, states


def upload(client, headers=None, name="customer.mp4", data=b"offline video"):
    return client.post("/portal-api/uploads", params={"filename": name}, content=data, headers=headers or MUTATE)


def test_header_required_no_old_validation_runs_or_internal_api(environment):
    client, calls, _ = environment
    for path in ("/", "/portal-api/me", "/portal-api/jobs", "/portal-api/jobs/missing", "/jobs/missing/report.html"):
        assert client.get(path).status_code == 401
    assert client.get("/portal-api/jobs", headers=AUTH).json() == {"jobs": []}
    for path in ("/api/health", "/v1/jobs", "/health", "/docs", "/openapi.json"):
        assert client.get(path, headers=AUTH).status_code == 404
    assert calls == []
    response = client.get("/portal-api/me", headers=AUTH)
    assert response.json()["limits"]["max_upload_bytes"] == 4 * 1024**3
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_csrf_custom_header_and_https_origin(environment):
    client, calls, _ = environment
    assert upload(client, headers=AUTH).status_code == 403
    assert upload(client, headers={**MUTATE, "Origin": "https://evil.invalid", "X-Forwarded-Proto": "https"}).status_code == 403
    assert not calls
    assert upload(client, headers={**MUTATE, "Origin": "https://testserver", "X-Forwarded-Proto": "https"}).status_code == 202


def test_fresh_upload_fixed_local_config_and_generated_source(environment):
    client, calls, _ = environment
    response = upload(client)
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "queued" and job["name"] == "customer.mp4"
    assert job["report_url"] is None and job["size_bytes"] == len(b"offline video")
    assert "source" not in job and "owner" not in job
    posts = [call for call in calls if call[0] == "POST"]
    assert len(posts) == 1
    payload = posts[0][2]["json"]
    assert payload["model"] == "local-test-model" and payload["refresh_cache"] is True
    assert payload["max_requests"] == 4000 and payload["max_tokens"] == 40000000
    assert payload["sample_interval"] == .5 and payload["max_pages"] == 300
    assert payload["native_implementation_fingerprint"] == "a" * 64
    source = Path(payload["source_path"])
    assert source.is_relative_to(portal.DATA / "uploads")
    assert source.name != "customer.mp4" and source.read_bytes() == b"offline video"


@pytest.mark.parametrize("name", ["../video.mp4", "folder/video.mp4", "folder\\video.mp4", "bad\x00.mp4", "file.txt", ""])
def test_invalid_filename_never_submitted(environment, name):
    client, calls, _ = environment
    assert upload(client, name=name).status_code == 400
    assert not calls


def test_limits_reject_before_and_during_stream(environment, monkeypatch):
    client, calls, _ = environment
    monkeypatch.setattr(portal, "MAX_BYTES", 4)
    assert upload(client, data=b"12345").status_code == 413
    assert client.post("/portal-api/uploads?filename=stream.mp4", content=iter([b"123", b"45"]), headers=MUTATE).status_code == 413
    assert not calls
    rows = client.get("/portal-api/jobs", headers=AUTH).json()["jobs"]
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert not list((portal.DATA / "uploads").glob("*.mp4"))


def test_invalid_video_or_storage_failure_never_submitted(environment, monkeypatch):
    client, calls, _ = environment

    async def invalid(path):
        raise ValueError("Видео не читается")

    monkeypatch.setattr(portal, "probe_video", invalid)
    assert upload(client).status_code == 400
    monkeypatch.setattr(portal.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    assert upload(client).status_code == 507
    assert not calls
    assert all(j["status"] == "failed" for j in client.get("/portal-api/jobs", headers=AUTH).json()["jobs"])


def test_owner_artifact_allowlist_traversal_symlinks_feedback_preserves_result(environment):
    client, _, _ = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    directory.mkdir(parents=True)
    original = b'{"pages":[],"unchanged":true}'
    (directory / "results.json").write_bytes(original)
    (directory / "report.html").write_text("<h1>Offline fixture</h1>")
    (directory / "trace.jsonl").write_text("internal trace")
    (directory / "report-assets").mkdir()
    (directory / "report-assets" / "preview.jpg").write_bytes(b"jpeg")
    outside = portal.DATA / "secret.jpg"
    outside.write_bytes(b"private")
    (directory / "report-assets" / "link.jpg").symlink_to(outside)
    other = {"X-Sever-User": "team-b"}
    assert client.get(f"/portal-api/jobs/{job['id']}", headers=other).status_code == 404
    assert client.get("/portal-api/jobs", headers=other).json() == {"jobs": []}
    for path in ("results.json", "report.html", "report-assets/preview.jpg"):
        url = f"/jobs/{job['job_id']}/{path}"
        assert client.get(url, headers=AUTH).status_code == 200
        assert client.get(url, headers=other).status_code == 404
    for path in ("trace.jsonl", "job.json", "report-assets/link.jpg", "../secret.jpg", "%2e%2e/secret.jpg", "report-assets/../../secret.jpg"):
        assert client.get(f"/jobs/{job['job_id']}/{path}", headers=AUTH).status_code == 404
    assert portal.artifact(job["job_id"], "/etc/passwd") is None
    assert portal.artifact(job["job_id"], "report-assets\\preview.jpg") is None
    note = {"page_id": "leaf-1", "field": "document_number", "comment": "Номер прочитан неверно", "proposed_value": "123"}
    assert client.post(f"/portal-api/jobs/{job['id']}/feedback", json=note, headers={**other, "X-Requested-With": "SeverPortal"}).status_code == 404
    response = client.post(f"/portal-api/jobs/{job['id']}/feedback", json=note, headers=MUTATE)
    assert response.status_code == 201
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["feedback"][0]["proposed_value"] == "123"
    assert current["results_url"] == f"/jobs/{job['job_id']}/results.json"
    assert (directory / "results.json").read_bytes() == original


def test_atomic_admission_limits_and_completed_slots_release(environment):
    client, calls, states = environment
    first = upload(client).json()
    second = upload(client).json()
    assert upload(client).status_code == 429
    states[first["job_id"]] = {"status": "completed"}
    assert upload(client).status_code == 202
    assert sum(call[0] == "POST" for call in calls) == 3
    assert client.get(f"/portal-api/jobs/{first['id']}", headers=AUTH).json()["status"] == "completed"


def test_concurrent_reservations_enforce_global_and_team_limit(environment):
    def reserve(index):
        try:
            return portal.reserve("parallel", f"video{index}.mp4")[0]
        except HTTPException as exc:
            assert exc.status_code == 429
            return None
    with ThreadPoolExecutor(max_workers=6) as pool:
        admitted = list(pool.map(reserve, range(6)))
    assert sum(bool(v) for v in admitted) == 2
    portal.reserve("second", "one.mp4")
    portal.reserve("third", "two.mp4")
    with pytest.raises(HTTPException) as error:
        portal.reserve("fourth", "three.mp4")
    assert error.value.status_code == 429


def test_uncertain_submission_retained_without_replay(environment, monkeypatch):
    client, calls, _ = environment
    original = portal.backend

    async def interrupted(method, path, **kwargs):
        if method == "POST":
            calls.append((method, path, kwargs))
            raise httpx.ReadTimeout("connection lost after request sent")
        return await original(method, path, **kwargs)

    monkeypatch.setattr(portal, "backend", interrupted)
    response = upload(client)
    assert response.status_code == 503
    for _ in range(3):
        rows = client.get("/portal-api/jobs", headers=AUTH).json()["jobs"]
        assert rows[0]["status"] == "submission_uncertain"
    portal.startup_reconcile()
    assert len([c for c in calls if c[0] == "POST"]) == 1
    assert len(list((portal.DATA / "uploads").glob("*.mp4"))) == 1


def test_startup_interrupted_uploads_fail_and_submissions_uncertain(environment):
    first, _ = portal.reserve("team-a", "one.mp4")
    second, _ = portal.reserve("team-a", "two.mp4")
    portal.update(second, status="submitting")
    portal.startup_reconcile()
    assert portal.owned(first, "team-a")["status"] == "failed"
    assert portal.owned(second, "team-a")["status"] == "submission_uncertain"


def test_progress_is_customer_readable_and_failure_preserved(environment):
    client, _, states = environment
    job = upload(client).json()
    states[job["job_id"]] = {"status": "processing", "progress": "Reading image 2/10 at 42.0s, view 3"}
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["progress"] == "Распознаём лист 2 из 10"
    states[job["job_id"]] = {"status": "failed", "error_type": "BatchBudgetExceeded"}
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["status"] == "failed" and "лимит" in current["error"]


@pytest.mark.parametrize("status,label", [("completed", "Обработка завершена"), ("completed_with_errors", "Обработка завершена с ошибками"), ("failed", "Обработка остановлена")])
def test_terminal_progress_ignores_last_reading_message(status, label):
    assert portal.translated_progress(status, "Reading image 2/10 at 42.0s, view 3") == label


def test_stopped_report_is_wrapped_without_modifying_evidence(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    directory.mkdir(parents=True)
    original = '<meta http-equiv="refresh" content="10"><h1>Север ТСЛ — обработка продолжается</h1><p>Показаны уже полученные чтения; обработка продолжается, значения ещё могут уточняться.</p><p><a href="partial-results.json">JSON</a> · <a href="trace.jsonl">Трасса шагов</a></p><p>Номер 123</p>'
    (directory / "report.html").write_text(original)
    (directory / "partial-results.json").write_text('{"number":"123"}')
    states[job["job_id"]] = {"status": "failed", "error_type": "InterruptedByServiceRestart"}
    # The report alone refreshes terminal state, even after the dashboard closes.
    response = client.get(f"/jobs/{job['job_id']}/report.html", headers=AUTH)
    assert "Это частичный результат" in response.text and 'href="/"' in response.text
    assert "trace.jsonl" not in response.text and 'http-equiv="refresh"' not in response.text
    assert "Номер 123" in response.text
    assert (directory / "report.html").read_text() == original
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["results_url"].endswith("/partial-results.json")
    assert current["status"] == "failed"


def test_unavailable_model_is_service_failure_not_invalid_video(environment, monkeypatch):
    client, calls, _ = environment
    async def unavailable(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return httpx.Response(503, request=httpx.Request(method, "http://localhost" + path))
    monkeypatch.setattr(portal, "backend", unavailable)
    response = upload(client)
    assert response.status_code == 503 and "недоступен" in response.json()["detail"]
    assert all(method == "GET" for method, _, _ in calls)
    assert client.get("/portal-api/jobs", headers=AUTH).json()["jobs"][0]["status"] == "failed"


def test_candidate_limit_is_partial_coverage_in_history_job_and_report(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    (directory / "video").mkdir(parents=True)
    analysis = {"pages": [{"page_id": "page-0001", "segmentation": {"limitations": [
        "visual_segmentation_is_heuristic", "page_limit_reached:321_candidates_found:300_retained"]}}]}
    analysis_bytes = json.dumps(analysis).encode()
    (directory / "video" / "video-analysis.json").write_bytes(analysis_bytes)
    report = '<h1>Север ТСЛ — результат обработки</h1><p>Номер 123</p>'
    result_bytes = b'{"status":"completed","pages":[]}'
    (directory / "report.html").write_text(report)
    (directory / "results.json").write_bytes(result_bytes)
    states[job["job_id"]] = {"status": "completed"}
    detail = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    history = client.get("/portal-api/jobs", headers=AUTH).json()["jobs"][0]
    for current in (detail, history):
        assert current["status"] == "completed_with_errors"
        assert "321" in current["coverage_warning"] and "300" in current["coverage_warning"]
        assert "ещё 21 не обрабатывались" in current["error"]
    rendered = client.get(detail["report_url"], headers=AUTH)
    assert "Видео обработано не полностью" in rendered.text
    assert "ещё 21 не обрабатывались" in rendered.text and "Номер 123" in rendered.text
    assert (directory / "report.html").read_text() == report
    assert (directory / "results.json").read_bytes() == result_bytes
    assert (directory / "video" / "video-analysis.json").read_bytes() == analysis_bytes
    assert client.get(f"/jobs/{job['job_id']}/video/video-analysis.json", headers=AUTH).status_code == 404


def test_regular_segmentation_limits_do_not_claim_partial_coverage(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"] / "video"
    directory.mkdir(parents=True)
    (directory / "video-analysis.json").write_text(json.dumps({"pages": [{"segmentation": {
        "limitations": ["visual_segmentation_is_heuristic", "identical_looking_leaves_may_merge"]}}]}))
    states[job["job_id"]] = {"status": "completed"}
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["status"] == "completed" and current["coverage_warning"] is None


def test_remote_model_is_not_used(environment, monkeypatch):
    client, calls, _ = environment
    async def remote(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return httpx.Response(200, json={"model_backend": "remote", "document_reader": "native_image", "model_name": "remote"}, request=httpx.Request(method, "http://localhost" + path))
    monkeypatch.setattr(portal, "backend", remote)
    assert upload(client).status_code == 503
    assert all(method == "GET" for method, _, _ in calls)


@pytest.mark.parametrize("duration,has_video,accepted", [(60, True, True), (3600, True, True), (3601, True, False), (0, True, False), (1, False, False), (float('nan'), True, False)])
def test_real_probe_parser_duration_and_video_stream(monkeypatch, duration, has_video, accepted):
    # Exercise the actual parser using ffprobe-shaped output, without invoking a model.
    class Process:
        returncode = 0
        async def communicate(self):
            return json.dumps({"format": {"duration": duration, "format_name": "mov,mp4,m4a,3gp,3g2,mj2"}, "streams": [{"codec_type": "video", "width": 640, "height": 480}] if has_video else []}).encode(), None
    async def subprocess(*args, **kwargs):
        assert args[0] == "ffprobe"
        assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        assert args[args.index("-format_whitelist") + 1] == "mov,matroska,avi"
        return Process()
    monkeypatch.setattr(portal.asyncio, "create_subprocess_exec", subprocess)
    if accepted:
        assert asyncio.run(portal.probe_video(Path("offline.mp4"))) == duration
    else:
        with pytest.raises(ValueError):
            asyncio.run(portal.probe_video(Path("offline.mp4")))


def test_discarded_views_visible_as_uncertain_coverage_not_known_missing_documents(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    (directory / "video").mkdir(parents=True)
    analysis = {"sampling": {"discarded_transition_samples": [3], "discarded_transition_evidence": [
        {"sample_index": 3, "timestamp_s": 1.25, "source_pts": 125, "path": "/private/source.png",
         "reason": "no_stable_brief_view_confirmed"}]}}
    analysis_bytes = json.dumps(analysis).encode()
    (directory / "video" / "video-analysis.json").write_bytes(analysis_bytes)
    report = '<h1>Север ТСЛ — результат обработки</h1><p>Номер 123</p>'
    (directory / "report.html").write_text(report)
    states[job["job_id"]] = {"status": "completed"}
    detail = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    history = client.get("/portal-api/jobs", headers=AUTH).json()["jobs"][0]
    for current in (detail, history):
        assert current["status"] == "completed"
        assert current["error"] is None
        # Uncertain excluded views are kept in video_coverage and the report's technical details, not on the card.
        assert current["progress"] == "Обработка завершена"
        assert current["coverage_warning"] is None
        assert current["video_coverage"]["uncertain_view_count"] == 1
        assert "число исключений не равно числу пропущенных документов" in current["video_coverage"]["warning"]
        assert "/private" not in json.dumps(current)
    response = client.get(detail["report_url"], headers=AUTH)
    assert "Охват видео требует проверки" not in response.text
    assert "Видео обработано не полностью" not in response.text
    assert (directory / "report.html").read_text() == report
    assert (directory / "video" / "video-analysis.json").read_bytes() == analysis_bytes


def test_portal_serves_separate_safe_coverage_warning(environment):
    client, _, _ = environment
    page = client.get("/", headers=AUTH).text
    assert "node('p', 'job-coverage')" in page
    assert "refs.coverage.textContent = coverageWarning" in page
    assert "refs.error.textContent.includes(coverageWarning)" in page
    assert "progress,docs,error,coverage,report" in page
    assert ".job-coverage" in page and "color:var(--amber)" in page


def _portal_sheet(page_id, start, kind, **values):
    from sever.schema import FIELDS_BY_TYPE
    fields = {name: ({"raw_value": values[name], "normalized_value": values[name], "status": "recognized", "evidence": []}
                     if name in values else {"raw_value": None, "normalized_value": None, "status": "unreadable", "evidence": []})
              for name in FIELDS_BY_TYPE[kind]}
    return {"page_id": page_id, "start_s": start, "end_s": start + 4.0, "frames": [],
            "result": {"document_type": kind, "fields": fields, "errors": []}}


def test_job_card_summary_lists_documents_and_other_forms(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    directory.mkdir(parents=True)
    other = _portal_sheet("page-3", 8.0, "upd")
    other["result"] = {"document_type": "unknown", "fields": {}, "errors": []}
    other["image_recognition"] = {"stop_reason": "out_of_scope_form_confirmed", "views": [{"observed_form": "CMR"}]}
    result = {"source": {"path": "v.mp4", "sha256": "0" * 64}, "pages": [
        _portal_sheet("page-1", 0.0, "ttn", document_number="2591", document_date="23.08.2026", consignor="ООО Ромашка"),
        _portal_sheet("page-2", 4.0, "ttn", document_number="2591"), other]}
    (directory / "results.json").write_text(json.dumps(result, ensure_ascii=False))
    states[job["job_id"]] = {"status": "completed"}
    summary = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()["summary"]
    assert summary["preliminary"] is False and summary["sheets_total"] == 3
    [document] = summary["documents"]
    assert document["title"] == "ТТН № 2591 от 23.08.2026" and document["sheets"] == 2
    assert {field["label"]: field["value"] or field["note"] for field in document["fields"]} == {
        "Грузоотправитель": "ООО Ромашка", "Грузополучатель": "не удалось прочитать"}
    assert summary["other_forms"] == [{"form": "CMR", "sheets": 1, "out_of_scope": True}]


def test_stopped_job_card_summary_is_preliminary(environment):
    client, _, states = environment
    job = upload(client).json()
    directory = portal.RUNS / job["job_id"]
    directory.mkdir(parents=True)
    partial = {"status": "failed", "pages": [_portal_sheet("page-1", 0.0, "upd", invoice_number="7")]}
    (directory / "partial-results.json").write_text(json.dumps(partial, ensure_ascii=False))
    states[job["job_id"]] = {"status": "failed", "error_type": "InterruptedByServiceRestart"}
    current = client.get(f"/portal-api/jobs/{job['id']}", headers=AUTH).json()
    assert current["status"] == "failed"
    assert current["summary"]["preliminary"] is True
    assert current["summary"]["documents"][0]["title"] == "УПД № 7"


def test_report_shows_the_customers_file_name_not_the_stored_upload_name(environment):
    client, _, states = environment
    job = upload(client, name="ТВ_2.mp4").json()
    directory = portal.RUNS / job["job_id"]
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({"source_path": "/srv/sever/customer/uploads/abc-123.mp4"}))
    (directory / "report.html").write_text('<h1>Север ТСЛ — результат обработки</h1><p class="lead"><strong>abc-123.mp4</strong> · найдено документов: 2</p>')
    states[job["job_id"]] = {"status": "completed"}
    page = client.get(f"/jobs/{job['job_id']}/report.html", headers=AUTH).text
    assert '<p class="lead"><strong>ТВ_2.mp4</strong>' in page and "abc-123.mp4" not in page
    assert (directory / "report.html").read_text().count("abc-123.mp4") == 1
