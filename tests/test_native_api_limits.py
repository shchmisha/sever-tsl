"""Token policies through the real API, with no background inference launched."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from sever import api
from sever import image_reader
from sever import image_agent, image_orientation, zoom_image_reader, image_conflicts, field_verification
from sever.inference_gate import InferenceGateBusy, evaluation_access, gate_path


@pytest.fixture
def isolated_jobs(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"Only API admission is exercised; video decoding is forbidden")
    root = tmp_path / "runs"
    configured, submitted = [], []

    def configure(model, max_requests, max_tokens):
        config = SimpleNamespace(model=model, max_requests=max_requests,
                                 max_tokens=max_tokens, refresh_cache=False)
        configured.append(config)
        return config

    class RecordingWorker:
        def submit(self, function, *args):
            # Record scheduling without executing the function or creating an
            # inference task; the test remains entirely offline.
            submitted.append((function, args))

    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "agent_config", configure)
    monkeypatch.setattr(api, "WORKER", RecordingWorker())
    with TestClient(api.app) as client:
        yield client, source, root, configured, submitted


def test_local_native_job_admits_eight_million_tokens_without_launching_inference(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source),
        "model": "offline-policy-fixture", "max_tokens": 8_000_000, "refresh_cache": True})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["status"] == "queued"
    assert status["source_path"] == str(source.resolve())
    assert len(configured) == len(submitted) == 1
    assert configured[0].max_tokens == 8_000_000 and configured[0].refresh_cache is True
    function, (submitted_id, request, config) = submitted[0]
    assert function is api.execute and submitted_id == job_id and config is configured[0]
    assert request.max_tokens == 8_000_000
    assert list((root / job_id).iterdir()) == [root / job_id / "job.json"]


@pytest.mark.parametrize("provider", ["remote", None])
def test_remote_over_budget_rejected_before_configuration_or_queue(monkeypatch, isolated_jobs, provider):
    if provider is None:
        monkeypatch.delenv("HARNESS_PROVIDER_KIND", raising=False)
    else:
        monkeypatch.setenv("HARNESS_PROVIDER_KIND", provider)
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source), "max_tokens": 500_001})
    assert response.status_code == 400
    assert response.json()["detail"] == "Remote model jobs are limited to 500000 tokens"
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_remote_exact_budget_remains_admitted(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "remote")
    client, source, _, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source), "max_tokens": 500_000})
    assert response.status_code == 202
    assert len(configured) == len(submitted) == 1
    assert configured[0].max_tokens == 500_000


@pytest.mark.parametrize("provider", ["local", "remote"])
def test_above_forty_million_rejected_by_request_schema_before_queue(monkeypatch, isolated_jobs, provider):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", provider)
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source), "max_tokens": 40_000_001})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "max_tokens"] and error["type"] == "less_than_equal"
               for error in response.json()["detail"])
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_local_zoom_allowance_is_preserved_in_configuration_and_queued_request(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, _, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source),
        "max_requests": 4000, "max_tokens": 40_000_000})
    assert response.status_code == 202, response.text
    assert len(configured) == len(submitted) == 1
    assert (configured[0].max_requests, configured[0].max_tokens) == (4000, 40_000_000)
    _, (_, request, _) = submitted[0]
    assert (request.max_requests, request.max_tokens) == (4000, 40_000_000)
    assert client.get(f"/v1/jobs/{response.json()['job_id']}").json()["status"] == "queued"


@pytest.mark.parametrize("provider", ["remote", None])
def test_remote_request_over_budget_rejected_before_configuration_or_queue(monkeypatch, isolated_jobs, provider):
    if provider is None:
        monkeypatch.delenv("HARNESS_PROVIDER_KIND", raising=False)
    else:
        monkeypatch.setenv("HARNESS_PROVIDER_KIND", provider)
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source), "max_requests": 401})
    assert response.status_code == 400
    assert response.json()["detail"] == "Remote model jobs are limited to 400 requests"
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_remote_exact_request_and_token_caps_remain_admitted(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "remote")
    client, source, _, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source),
        "max_requests": 400, "max_tokens": 500_000})
    assert response.status_code == 202
    assert len(configured) == len(submitted) == 1
    assert (configured[0].max_requests, configured[0].max_tokens) == (400, 500_000)


@pytest.mark.parametrize("field,value", [("max_requests", 0), ("max_requests", 4001),
                                         ("max_tokens", 999)])
def test_invalid_local_zoom_allowance_cannot_configure_or_queue(monkeypatch, isolated_jobs, field, value):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source), field: value})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", field] for error in response.json()["detail"])
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_native_health_fingerprint_is_stable_covers_source_prompt_schema_and_no_secrets(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    monkeypatch.setenv("HARNESS_MODEL_API_KEY", "DO-NOT-EXPOSE-THIS-KEY")
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "http://user:PRIVATE-ENDPOINT-PASSWORD@127.0.0.1:8090/v1")
    client, _, _, configured, submitted = isolated_jobs
    first = client.get("/health").json()["native_implementation"]
    assert first["status"] == "available" and len(first["fingerprint_sha256"]) == 64
    assert {"image_reader.py", "image_fusion.py", "native_pipeline.py", "api.py",
            "image_agent.py", "zoom_image_reader.py", "inference_gate.py"} <= set(first["source_sha256"])
    assert first["reader_version"] == zoom_image_reader.ZOOM_READER_VERSION
    assert first["image_agent_version"] == image_agent.IMAGE_AGENT_VERSION
    assert first["reading_output_mode"] == "tool_output:submit_reading"
    assert first["per_frame_limits"] == {"agent_model_requests": image_agent.MAX_MODEL_REQUESTS,
                                          "zoom_images": image_agent.MAX_ZOOM_IMAGES}
    assert all(len(value) == 64 for value in first["source_sha256"].values())
    assert first == client.get("/health").json()["native_implementation"]
    response = client.get("/health").text
    assert "DO-NOT-EXPOSE-THIS-KEY" not in response and "PRIVATE-ENDPOINT-PASSWORD" not in response
    monkeypatch.setenv("HARNESS_MODEL_API_KEY", "a different private key")
    assert first["fingerprint_sha256"] == client.get("/health").json()["native_implementation"]["fingerprint_sha256"]
    monkeypatch.setattr(image_reader, "INSTRUCTIONS", image_reader.INSTRUCTIONS + "\nNew generic image task.")
    changed_fallback = client.get("/health").json()["native_implementation"]
    assert changed_fallback["fallback_instructions_sha256"] != first["fallback_instructions_sha256"]
    # The agent's captured instruction string is the actual active model input.
    assert changed_fallback["instructions_sha256"] == first["instructions_sha256"]
    monkeypatch.setattr(image_agent, "INSTRUCTIONS", image_agent.INSTRUCTIONS + "\nNew generic active task.")
    changed_prompt = client.get("/health").json()["native_implementation"]
    assert changed_prompt["instructions_sha256"] != first["instructions_sha256"]
    assert changed_prompt["fingerprint_sha256"] != first["fingerprint_sha256"]
    changed_sources = dict(first["source_sha256"], **{"image_fusion.py": "b" * 64})
    monkeypatch.setattr(api, "_native_source_hashes", lambda: changed_sources)
    changed_code = client.get("/health").json()["native_implementation"]
    assert changed_code["fingerprint_sha256"] != changed_prompt["fingerprint_sha256"]
    schema = image_reader._NativeReadingResponse.model_json_schema()
    monkeypatch.setattr(image_reader._NativeReadingResponse, "model_json_schema", lambda: {**schema, "description": "Changed output schema"})
    changed_schema = client.get("/health").json()["native_implementation"]
    assert changed_schema["schema_sha256"] != changed_code["schema_sha256"]
    assert changed_schema["fingerprint_sha256"] != changed_code["fingerprint_sha256"]
    assert configured == submitted == []


@pytest.mark.parametrize("module,name,value", [
    (image_agent, "IMAGE_AGENT_VERSION", "changed-agent-version"),
    (image_agent, "ZOOM_INSTRUCTIONS", "Changed generic source zoom instructions"),
    (image_agent, "MAX_MODEL_REQUESTS", 9),
    (image_agent, "MAX_ZOOM_IMAGES", 9),
    (zoom_image_reader, "ZOOM_READER_VERSION", "changed-wrapper-version"),
    (image_orientation, "ORIENTATION_VERSION", "changed-orientation-version"),
    (image_orientation, "ORIENTATION_OUTPUT_LIMIT", 129),
    (image_orientation, "INSTRUCTIONS", "Changed generic orientation instructions"),
    (image_conflicts, "IMAGE_CONFLICTS_VERSION", "changed-joint-review-version"),
    (image_conflicts, "INSTRUCTIONS", "Changed generic conflict review instructions"),
    (image_conflicts, "TYPE_INSTRUCTIONS", "Changed generic joint form review instructions"),
    (image_conflicts, "MAX_MODEL_REQUESTS", 13),
    (image_conflicts, "MAX_ZOOM_IMAGES", 13),
    (image_conflicts, "MAX_REVIEW_FRAMES", 3),
    (field_verification, "FIELD_VERIFICATION_VERSION", "changed-detail-version"),
    (field_verification, "LOCALIZATION_INSTRUCTIONS", "Changed field localization task"),
    (field_verification, "TRANSCRIPTION_INSTRUCTIONS", "Changed field detail task"),
    (field_verification, "DETAIL_EDGE", 512),
    (field_verification, "DETAIL_OVERLAP", 32),
    (field_verification, "MAX_REQUESTS_PER_FRAME", 4),
])
def test_loaded_zoom_and_orientation_tasks_change_pin_without_file_edits(monkeypatch, isolated_jobs, module, name, value):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, _, _, configured, submitted = isolated_jobs
    before = client.get("/health").json()["native_implementation"]["fingerprint_sha256"]
    monkeypatch.setattr(module, name, value)
    after = client.get("/health").json()["native_implementation"]["fingerprint_sha256"]
    assert after != before and configured == submitted == []


def test_loaded_orientation_schema_changes_pin(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, _, _, configured, submitted = isolated_jobs
    before = client.get("/health").json()["native_implementation"]
    schema = image_orientation._OrientationDecision.model_json_schema()
    monkeypatch.setattr(image_orientation._OrientationDecision, "model_json_schema",
                        lambda: {**schema, "description": "Changed orientation output contract"})
    after = client.get("/health").json()["native_implementation"]
    assert after["orientation"]["schema_sha256"] != before["orientation"]["schema_sha256"]
    assert after["fingerprint_sha256"] != before["fingerprint_sha256"]
    assert configured == submitted == []


@pytest.mark.parametrize("name,value", [("HARNESS_VISION_MODEL", "different-vision-model"),
    ("HARNESS_MAX_OUTPUT_TOKENS", "4096"), ("HARNESS_MODEL_TIMEOUT", "123")])
def test_effective_native_model_settings_change_health_pin(monkeypatch, isolated_jobs, name, value):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, _, _, configured, submitted = isolated_jobs
    initial = client.get("/health").json()["native_implementation"]["fingerprint_sha256"]
    monkeypatch.setenv(name, value)
    assert client.get("/health").json()["native_implementation"]["fingerprint_sha256"] != initial
    assert configured == submitted == []


def test_ocr_guided_health_explicitly_reports_no_native_pin(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "ocr_guided")
    client, _, _, configured, submitted = isolated_jobs
    assert client.get("/health").json()["native_implementation"] == {"status": "not_active", "fingerprint_sha256": None}
    assert configured == submitted == []


def test_mismatched_native_pin_rejected_before_configuration_or_queue(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, source, root, configured, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source),
        "native_implementation_fingerprint": "0" * 64})
    assert response.status_code == 409 and "implementation changed" in response.json()["detail"]
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_matching_native_pin_is_preserved_and_worker_blocks_change_before_inference(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, source, _, configured, submitted = isolated_jobs
    identity = client.get("/health").json()["native_implementation"]
    fingerprint = identity["fingerprint_sha256"]
    response = client.post("/v1/jobs", json={"source_path": str(source),
        "native_implementation_fingerprint": fingerprint})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["status"] == "queued" and status["native_implementation_fingerprint"] == fingerprint
    function, (submitted_id, request, config) = submitted[0]
    assert request.native_implementation_fingerprint == fingerprint and len(configured) == 1
    monkeypatch.setattr(api, "_native_source_hashes", lambda: {**identity["source_sha256"], "native_pipeline.py": "b" * 64})
    called = []
    monkeypatch.setattr(api, "process_video", lambda *args, **kwargs: called.append(True))
    function(submitted_id, request, config)
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error_type"] == "NativeImplementationChanged"
    assert status["native_implementation_fingerprint"] == fingerprint and called == []


def test_worker_fingerprint_check_fault_is_recorded_instead_of_leaving_queued_job(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_DOCUMENT_READER", "native_image")
    client, source, _, _, submitted = isolated_jobs
    fingerprint = client.get("/health").json()["native_implementation"]["fingerprint_sha256"]
    response = client.post("/v1/jobs", json={"source_path": str(source), "native_implementation_fingerprint": fingerprint})
    assert response.status_code == 202
    function, arguments = submitted[0]

    def unavailable_identity():
        raise OSError("Injected file-read fault")

    monkeypatch.setattr(api, "native_implementation", unavailable_identity)
    monkeypatch.setattr(api, "process_video", lambda *args, **kwargs: pytest.fail("Inference is forbidden"))
    function(*arguments)
    status = client.get(f"/v1/jobs/{response.json()['job_id']}").json()
    assert status["status"] == "failed" and status["error_type"] == "OSError"
    assert status["native_implementation_fingerprint"] == fingerprint


def test_evaluation_gate_blocks_admission_before_configuration_or_queue(monkeypatch, isolated_jobs):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, configured, submitted = isolated_jobs
    path = gate_path(run_dir=root)
    with evaluation_access(path):
        response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 423
    assert "isolated evaluation" in response.json()["detail"]
    assert configured == submitted == []
    assert list(root.glob("*/job.json")) == []
    # The exclusive gate has been released; the same request can be admitted.
    assert client.post("/v1/jobs", json={"source_path": str(source)}).status_code == 202
    assert len(configured) == len(submitted) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_worker_holds_shared_gate_during_progress_and_terminal_state(monkeypatch, isolated_jobs, failed):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, _, submitted = isolated_jobs
    path = gate_path(run_dir=root)
    events = []
    original_state = api.state

    def checked_state(job_id, value):
        with pytest.raises(InferenceGateBusy):
            with evaluation_access(path):
                pytest.fail("Evaluation acquired the gate during queued or active inference")
        events.append(value["status"])
        original_state(job_id, value)

    async def controlled_processing(source, out, *, progress, **kwargs):
        assert client.get(f"/v1/jobs/{out.name}").json()["status"] == "processing"
        with pytest.raises(InferenceGateBusy):
            with evaluation_access(path):
                pytest.fail("Evaluation acquired the gate while the worker processes")
        progress("Controlled offline processing")
        if failed:
            raise RuntimeError("Controlled offline processing failure")
        return {"pages": [], "usage": {"requests": 0}}

    monkeypatch.setattr(api, "state", checked_state)
    monkeypatch.setattr(api, "process_video", controlled_processing)
    response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 202
    function, args = submitted[0]
    function(*args)
    expected = "failed" if failed else "completed"
    assert events == ["queued", "processing", "processing", expected]
    assert client.get(f"/v1/jobs/{response.json()['job_id']}").json()["status"] == expected
    with evaluation_access(path):
        pass


def test_accepted_worker_waits_and_runs_after_exclusive_gate_releases(monkeypatch, isolated_jobs):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, _, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 202
    entered, processed = Event(), []
    original_access = api.inference_access

    def observed_access(path, *, blocking=False):
        assert blocking is True
        entered.set()
        return original_access(path, blocking=blocking)

    async def controlled_processing(*args, **kwargs):
        processed.append(True)
        return {"pages": [], "usage": {"requests": 0}}

    monkeypatch.setattr(api, "inference_access", observed_access)
    monkeypatch.setattr(api, "process_video", controlled_processing)
    function, args = submitted[0]
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with evaluation_access(gate_path(run_dir=root)):
            future = pool.submit(function, *args)
            assert entered.wait(timeout=2), "Queued worker did not attempt its shared lock"
            assert not future.done() and processed == []
            status = client.get(f"/v1/jobs/{response.json()['job_id']}").json()
            assert status["status"] == "queued"
        future.result(timeout=2)
    finally:
        pool.shutdown(wait=True)
    status = client.get(f"/v1/jobs/{response.json()['job_id']}").json()
    assert status["status"] == "completed" and processed == [True]


def test_unavailable_inference_gate_fails_admission_visibly_without_queue(monkeypatch, isolated_jobs):
    from contextlib import contextmanager

    @contextmanager
    def unavailable(*args, **kwargs):
        raise PermissionError("Injected unavailable gate")
        yield

    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, root, configured, submitted = isolated_jobs
    monkeypatch.setattr(api, "inference_access", unavailable)
    response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 503 and "unavailable" in response.json()["detail"]
    assert configured == submitted == [] and not list(root.glob("*/job.json"))


def test_worker_gate_storage_failure_does_not_leave_queued_or_start_processing(monkeypatch, isolated_jobs):
    from contextlib import contextmanager

    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    client, source, _, _, submitted = isolated_jobs
    response = client.post("/v1/jobs", json={"source_path": str(source)})
    assert response.status_code == 202

    @contextmanager
    def unavailable(*args, **kwargs):
        raise PermissionError("Injected unavailable gate")
        yield

    monkeypatch.setattr(api, "inference_access", unavailable)
    monkeypatch.setattr(api, "process_video", lambda *args, **kwargs: pytest.fail("Model processing is forbidden"))
    function, args = submitted[0]
    function(*args)
    status = client.get(f"/v1/jobs/{response.json()['job_id']}").json()
    assert status["status"] == "failed" and status["error_type"] == "PermissionError"
