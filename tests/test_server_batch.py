"""Paid-job driver safety using a controlled local API, with no provider calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

spec = importlib.util.spec_from_file_location("server_batch", Path(__file__).parents[1] / "deploy/server_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def settings(tmp_path, *, requests=120, tokens=900000):
    source = tmp_path / "inputs"
    source.mkdir(exist_ok=True)
    return SimpleNamespace(source_dir=source, run_dir=tmp_path / "runs", batch_dir=tmp_path / "runs/batches/check",
                           base_url="http://127.0.0.1:8080", download_provenance=None,
                           max_requests=requests, max_tokens=tokens, interval=.5, poll_seconds=.01, job_timeout=30)


class ControlledAPI:
    """Writes the worker's real artifact shape, but never processes or pays."""
    def __init__(self, run_dir, fail=False, lose_post_reply=False, provider_kind="remote", model_base_url=None):
        self.run_dir = run_dir
        self.fail = fail
        self.lose_post_reply = lose_post_reply
        self.posts = []
        self.provider_kind = provider_kind
        self.model_base_url = model_base_url or ("http://127.0.0.1:8090/v1" if provider_kind == "local" else "https://routerai.ru/api/v1")
        self.health = {"status": "ok"}  # Legacy API has no implementation pin.

    def request(self, method, endpoint, payload=None):
        if endpoint == "/health":
            return self.health
        if method == "POST":
            self.posts.append(payload)
            job_id = str(uuid.uuid4())
            directory = self.run_dir / job_id
            directory.mkdir(parents=True)
            sha = batch.digest(payload["source_path"])
            trace = [{"event": "run_started", "source": payload["source_path"], "source_id": sha, "previous_artifacts": None},
                     {"event": "model_request_started", "batch_attempted_requests": 1},
                     {"event": "model_response_received", "input_tokens": 1300, "output_tokens": 100,
                      "provider_kind": self.provider_kind, "model_base_url": self.model_base_url,
                      "billing": ({"kind": "local_inference", "currency": None, "value": 0} if self.provider_kind == "local"
                                  else {"kind": "provider_reported", "currency": "RUB", "value": .1})}]
            status = {"job_id": job_id, "source_path": payload["source_path"],
                      "status": "failed" if self.fail else "completed"}
            if payload.get("native_implementation_fingerprint"):
                status["native_implementation_fingerprint"] = payload["native_implementation_fingerprint"]
            if self.fail:
                status["error_type"] = "BatchBudgetExceeded"
            else:
                trace.append({"event": "run_completed"})
                batch.save(directory / "results.json", {"source": {"path": payload["source_path"], "sha256": sha},
                    "pages": [{"page_id": "p1", "result": {"document_type": "unknown", "fields": {}, "errors": []}}],
                    "usage": {"attempted_requests": 1, "requests": 1, "input_tokens": 1300,
                              "output_tokens": 100, "actual_cost_rub": 0 if self.provider_kind == "local" else .1,
                              "costed_requests": 0 if self.provider_kind == "local" else 1}})
                (directory / "report.html").write_text("<p>Actual controlled report</p>")
                (directory / "fields.csv").write_text("field,value\n")
            (directory / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
            batch.save(directory / "job.json", status)
            if self.lose_post_reply:
                raise batch.APIError("Controlled lost POST reply")
            return {"job_id": job_id}
        job_id = endpoint.split("/")[3]
        file = "results.json" if endpoint.endswith("/results") else "job.json"
        return json.loads((self.run_dir / job_id / file).read_text())


def test_only_public_provenance_videos_are_selected_and_bound(tmp_path):
    args = settings(tmp_path)
    video = args.source_dir / "public.mp4"
    video.write_bytes(b"public source")
    (args.source_dir / "input-4s.mp4").write_bytes(b"unrelated demonstration clip")
    provenance = tmp_path / "downloads.json"
    batch.save(provenance, {"files": [{"name": video.name, "size": video.stat().st_size,
                                      "sha256": batch.digest(video)}]})
    selected = batch.sources_from(args.source_dir, provenance)
    assert [source["name"] for source in selected] == ["public.mp4"]
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        batch.sources_from(args.source_dir, provenance)


def test_actual_api_shape_shared_budget_and_restart_does_not_resubmit(tmp_path):
    args = settings(tmp_path, requests=3, tokens=5000)
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed"
    assert [request["max_requests"] for request in client.posts] == [3, 2]
    assert [request["max_tokens"] for request in client.posts] == [5000, 3600]
    assert all(request["refresh_cache"] is True for request in client.posts)
    assert state["usage"]["input_tokens"] == 2600
    assert state["usage"]["actual_cost_rub"] == pytest.approx(.2)
    assert all(job["proof"]["result_source_sha256_matches"] for job in state["jobs"])
    assert all(job["proof"]["historical_model_cache_hits"] == 0 for job in state["jobs"])
    assert all(job["proof"]["fresh_run_verified"] for job in state["jobs"])
    resumed = batch.BatchDriver(args, client=client).run()
    assert resumed["status"] == "completed" and len(client.posts) == 2
    assert "report.html" in (args.run_dir / "index.html").read_text()


@pytest.mark.parametrize("priorities,order", [
    ([], ["a.mp4", "b.mp4", "c.mp4"]),
    (["c.mp4"], ["c.mp4", "a.mp4", "b.mp4"]),
    (["c.mp4", "b.mp4"], ["c.mp4", "b.mp4", "a.mp4"]),
])
def test_first_sources_only_reorder_bound_inventory_and_resume_without_resubmission(tmp_path, priorities, order):
    args = settings(tmp_path)
    args.first_source = priorities
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(("original " + name).encode())
    originals = batch.sources_from(args.source_dir)
    client = ControlledAPI(args.run_dir)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed"
    assert [source["name"] for source in state["sources"]] == order
    assert {source["name"]: source for source in state["sources"]} == {source["name"]: source for source in originals}
    assert [Path(payload["source_path"]).name for payload in client.posts] == order
    assert state["counts"]["selected_videos"] == state["counts"]["submitted_jobs"] == 3
    assert state["settings"]["first_sources"] == priorities
    assert state["settings"]["source_order"] == order
    assert all(set(payload) == {"source_path", "model", "sample_interval", "max_pages",
                                "max_requests", "max_tokens", "refresh_cache"} for payload in client.posts)
    assert all(job["proof"]["result_source_sha256_matches"] for job in state["jobs"])
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 3


@pytest.mark.parametrize("priorities,error", [
    (["missing.mp4"], "not in the selected inventory"),
    (["a.mp4", "a.mp4"], "Duplicate first-source"),
    (["A.mp4"], "not in the selected inventory"),
    (["inputs/a.mp4"], "exact selected video basenames"),
    ([""], "exact selected video basenames"),
])
def test_invalid_first_sources_fail_before_any_job_is_submitted(tmp_path, priorities, error):
    args = settings(tmp_path)
    args.first_source = priorities
    (args.source_dir / "a.mp4").write_bytes(b"original")
    client = ControlledAPI(args.run_dir)
    with pytest.raises(ValueError, match=error):
        batch.BatchDriver(args, client=client)
    assert client.posts == []
    assert not (args.batch_dir / "manifest.json").exists()


def test_first_source_cannot_add_an_existing_video_outside_download_inventory(tmp_path):
    args = settings(tmp_path)
    public = args.source_dir / "a.mp4"
    public.write_bytes(b"original")
    (args.source_dir / "unrelated.mp4").write_bytes(b"not selected")
    args.download_provenance = tmp_path / "downloads.json"
    batch.save(args.download_provenance, {"files": [{"name": public.name, "size": public.stat().st_size,
                                                    "sha256": batch.digest(public)}]})
    args.first_source = ["unrelated.mp4"]
    client = ControlledAPI(args.run_dir)
    with pytest.raises(ValueError, match="not in the selected inventory"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []
    args.first_source = ["a.mp4"]
    driver = batch.BatchDriver(args, client=client)
    assert driver.state["sources"] == batch.sources_from(args.source_dir, args.download_provenance)
    assert driver.state["download_provenance"]["sha256"] == batch.digest(args.download_provenance)


@pytest.mark.parametrize("original,changed", [
    ([], ["a.mp4"]),  # Same effective order still changes the recorded priorities.
    (["a.mp4"], []),
    (["c.mp4"], ["b.mp4"]),
    (["c.mp4", "a.mp4"], ["a.mp4", "c.mp4"]),
])
def test_changing_first_sources_cannot_resume_an_existing_batch(tmp_path, original, changed):
    args = settings(tmp_path)
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    args.first_source = original
    client = ControlledAPI(args.run_dir)
    batch.BatchDriver(args, client=client)
    saved = (args.batch_dir / "manifest.json").read_bytes()
    args.first_source = changed
    with pytest.raises(ValueError, match="settings or input hashes differ"):
        batch.BatchDriver(args, client=client)
    assert (args.batch_dir / "manifest.json").read_bytes() == saved
    assert client.posts == []


def test_changed_manifest_source_order_is_rejected_on_resume(tmp_path):
    args = settings(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    driver = batch.BatchDriver(args, client=ControlledAPI(args.run_dir))
    driver.state["settings"]["source_order"] = ["b.mp4", "a.mp4"]
    batch.save(driver.manifest_path, driver.state)
    with pytest.raises(ValueError, match="settings or input hashes differ"):
        batch.BatchDriver(args, client=ControlledAPI(args.run_dir))


def test_lost_submission_reply_recovers_existing_job_without_second_post(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"original")
    client = ControlledAPI(args.run_dir, lose_post_reply=True)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed" and len(client.posts) == 1
    assert state["jobs"][0]["submission_recovered_from_server_state"] is True
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 1


def test_restart_after_completed_state_before_result_fetch_finalizes_without_post(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"original")

    class InterruptedResultAPI(ControlledAPI):
        interrupted = False

        def request(self, method, endpoint, payload=None):
            if endpoint.endswith("/results") and not self.interrupted:
                self.interrupted = True
                raise batch.APIError("Controlled interruption after completed status")
            return super().request(method, endpoint, payload)

    client = InterruptedResultAPI(args.run_dir)
    with pytest.raises(batch.APIError):
        batch.BatchDriver(args, client=client).run()
    saved = json.loads((args.batch_dir / "manifest.json").read_text())
    assert saved["jobs"][0]["status"] == "completed"
    assert not saved["jobs"][0].get("finalized")
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed" and len(client.posts) == 1
    assert state["jobs"][0]["finalized"] is True


def test_recover_post_after_worker_finished_and_api_state_dropped_source_path(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"original")

    class FinishedLostReplyAPI(ControlledAPI):
        def request(self, method, endpoint, payload=None):
            if method == "POST":
                try:
                    return super().request(method, endpoint, payload)
                except batch.APIError:
                    identifier = next(iter(batch.job_ids(self.run_dir)))
                    path = self.run_dir / identifier / "job.json"
                    state = json.loads(path.read_text())
                    state.pop("source_path")
                    batch.save(path, state)
                    raise
            return super().request(method, endpoint, payload)

    client = FinishedLostReplyAPI(args.run_dir, lose_post_reply=True)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed" and len(client.posts) == 1
    assert state["jobs"][0]["submission_recovered_from_server_state"] is True


def test_submission_without_identifiable_server_job_stops_instead_of_retry(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"original")
    driver = batch.BatchDriver(args, client=ControlledAPI(args.run_dir))
    source = driver.state["sources"][0]
    driver.state["jobs"].append({"source": source, "status": "submitting", "preexisting_job_ids": []})
    driver.persist()
    state = driver.run()
    assert state["status"] == "needs_attention"
    assert driver.client.posts == []


def test_failed_job_trace_usage_counts_toward_budget_and_is_not_repeated(tmp_path):
    args = settings(tmp_path, requests=1, tokens=5000)
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir, fail=True)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "budget_exhausted"
    assert state["usage"]["input_tokens"] == 1300 and state["usage"]["attempted_requests"] == 1
    assert state["jobs"][0]["status"] == "failed"
    assert state["jobs"][0]["proof"]["run_completed"] is False
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 1


def test_unanswered_request_reserves_unknown_tokens_instead_of_assuming_zero():
    jobs = [{"usage": {"attempted_requests": 2, "requests": 1, "input_tokens": 1000, "output_tokens": 100},
             "proof": {"requests_without_response": 1}, "request": {"max_requests": 8, "max_tokens": 50000}}]
    usage = batch.total_usage(jobs)
    assert usage["budget_tokens"] == 50000 and usage["budget_requests"] == 8
    assert usage["requests_without_response"] == 1


def test_index_escapes_filenames_and_progress_and_has_no_unavailable_artifact_links(tmp_path):
    args = settings(tmp_path)
    video = args.source_dir / '<img onerror="alert(1)">.mp4'
    video.write_bytes(b"original")
    driver = batch.BatchDriver(args, client=ControlledAPI(args.run_dir))
    content = (args.run_dir / "index.html").read_text()
    assert '<img onerror="alert(1)">' not in content
    assert '&lt;img onerror=' in content
    assert 'href="report.html"' not in content


def test_local_model_and_job_budget_are_submitted_and_proved_without_remote_billing(tmp_path):
    args = settings(tmp_path, requests=2000, tokens=20000000)
    args.model = "qwen38-27b-q6"
    args.provider_kind = "local"
    args.job_max_tokens = 8000000
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir, provider_kind="local")
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed"
    assert all(row["model"] == "qwen38-27b-q6" for row in client.posts)
    assert [row["max_tokens"] for row in client.posts] == [8000000, 8000000]
    assert state["settings"]["provider_kind"] == "local"
    assert state["settings"]["model"] == args.model
    assert state["usage"]["actual_cost_rub"] == 0
    assert state["usage"]["local_inference_responses"] == 2
    assert state["usage"]["responses_without_reported_cost"] == 0
    assert all(job["proof"]["local_provider_verified"] for job in state["jobs"])
    assert all(job["proof"]["fresh_run_verified"] for job in state["jobs"])
    index = (args.run_dir / "index.html").read_text()
    assert args.model in index and "http://127.0.0.1:8090/v1" in index and "Стоимость аренды сервера" in index
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 2


@pytest.mark.parametrize("kind,endpoint", [("remote", "https://routerai.ru/api/v1"),
                                          ("local", "https://routerai.ru/api/v1")])
def test_local_batch_flags_wrong_provider_or_remote_endpoint_and_stops_next_submission(tmp_path, kind, endpoint):
    args = settings(tmp_path)
    args.provider_kind = "local"
    args.model = "local-model"
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir, provider_kind=kind, model_base_url=endpoint)
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "provider_verification_failed" and len(client.posts) == 1
    assert state["jobs"][0]["proof"]["fresh_run_verified"] is False
    assert state["jobs"][0]["proof"]["local_provider_verified"] is False
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 1


def test_local_job_quota_is_capped_by_remaining_cumulative_tokens(tmp_path):
    args = settings(tmp_path, tokens=2100)
    args.provider_kind = "local"
    args.job_max_tokens = 2000000
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir, provider_kind="local")
    batch.BatchDriver(args, client=client).run()
    assert client.posts[0]["max_tokens"] == 2100


def test_old_remote_manifest_without_new_options_resumes_without_repeated_posts(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    state = batch.BatchDriver(args, client=client).run()
    for name in ("model", "provider_kind", "job_max_tokens", "job_max_requests", "first_sources", "source_order"):
        del state["settings"][name]
    state.pop("native_implementation_binding")
    batch.save(args.batch_dir / "manifest.json", state)
    resumed = batch.BatchDriver(args, client=client).run()
    assert resumed["status"] == "completed" and len(client.posts) == 1
    assert resumed["settings"]["provider_kind"] == "remote"
    assert resumed["settings"]["job_max_requests"] == 400
    assert resumed["settings"]["first_sources"] == []
    assert resumed["settings"]["source_order"] == ["a.mp4"]


def test_switching_model_during_existing_batch_is_rejected(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    batch.BatchDriver(args, client=ControlledAPI(args.run_dir))
    args.model = "different-model"
    with pytest.raises(ValueError, match="settings or input hashes differ"):
        batch.BatchDriver(args, client=ControlledAPI(args.run_dir))


def test_cli_local_budget_options_and_remote_guard():
    defaults = batch.arguments([])
    assert (defaults.model, defaults.provider_kind, defaults.job_max_tokens, defaults.job_max_requests) == ("openai/gpt-4.1-mini", "remote", 500000, 400)
    assert defaults.first_source == []
    local = batch.arguments(["--model", "qwen38-27b-q6", "--provider-kind", "local",
                             "--job-max-tokens", "40000000", "--job-max-requests", "4000",
                             "--max-requests", "22000", "--max-tokens", "440000000"])
    assert local.model == "qwen38-27b-q6" and local.job_max_tokens == 40000000
    assert local.job_max_requests == 4000 and local.max_requests == 22000 and local.max_tokens == 440000000
    assert batch.arguments(["--job-max-tokens", "500000"]).job_max_tokens == 500000
    with pytest.raises(SystemExit):
        batch.arguments(["--job-max-tokens", "500001"])
    with pytest.raises(SystemExit):
        batch.arguments(["--provider-kind", "local", "--job-max-tokens", "40000001"])


def test_cli_first_source_options_preserve_exact_repeat_order():
    args = batch.arguments(["--first-source", "c.mp4", "--first-source", "a.mp4"])
    assert args.first_source == ["c.mp4", "a.mp4"]


@pytest.mark.parametrize("kind,cap", [("remote", 500001), ("remote", 8000000), ("local", 40000001)])
def test_driver_rejects_invalid_provider_cap_before_submitting_job(tmp_path, kind, cap):
    args = settings(tmp_path)
    args.provider_kind = kind
    args.job_max_tokens = cap
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir, provider_kind=kind)
    with pytest.raises(ValueError, match="valid provider/token cap"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []


def test_local_zoom_job_request_and_token_allowances_are_not_silently_clamped_to_legacy_caps(tmp_path):
    args = settings(tmp_path, requests=10000, tokens=100_000_000)
    args.provider_kind = "local"
    args.job_max_requests = 4000
    args.job_max_tokens = 40_000_000
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir, provider_kind="local")
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed"
    assert [row["max_requests"] for row in client.posts] == [4000, 4000]
    assert [row["max_tokens"] for row in client.posts] == [40_000_000, 40_000_000]
    assert state["settings"]["job_max_requests"] == 4000
    assert state["settings"]["job_max_tokens"] == 40_000_000
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 2


def test_zoom_job_allowances_cannot_exceed_remaining_shared_requests_or_tokens(tmp_path):
    args = settings(tmp_path, requests=600, tokens=40_000_000)
    args.provider_kind = "local"
    args.job_max_requests = 2000
    args.job_max_tokens = 40_000_000
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir, provider_kind="local")
    state = batch.BatchDriver(args, client=client).run()
    assert state["status"] == "completed"
    assert [row["max_requests"] for row in client.posts] == [600, 599]
    assert [row["max_tokens"] for row in client.posts] == [40_000_000, 39_998_600]


def test_old_local_manifest_keeps_legacy_request_cap_without_resubmission(tmp_path):
    args = settings(tmp_path)
    args.provider_kind = "local"
    args.job_max_tokens = 8_000_000
    (args.source_dir / "a.mp4").write_bytes(b"original")
    client = ControlledAPI(args.run_dir, provider_kind="local")
    state = batch.BatchDriver(args, client=client).run()
    del state["settings"]["job_max_requests"]
    batch.save(args.batch_dir / "manifest.json", state)
    resumed = batch.BatchDriver(args, client=client).run()
    assert resumed["status"] == "completed" and len(client.posts) == 1
    assert resumed["settings"]["job_max_requests"] == 400


def test_changed_job_request_cap_requires_new_batch_directory_even_before_first_submission(tmp_path):
    args = settings(tmp_path)
    args.provider_kind = "local"
    (args.source_dir / "a.mp4").write_bytes(b"original")
    client = ControlledAPI(args.run_dir, provider_kind="local")
    batch.BatchDriver(args, client=client)
    args.job_max_requests = 2000
    with pytest.raises(ValueError, match="settings or input hashes differ"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []


@pytest.mark.parametrize("kind,cap", [("remote", 401), ("remote", 2000),
                                      ("local", 0), ("local", 4001)])
def test_invalid_job_request_cap_rejected_before_submission(tmp_path, kind, cap):
    args = settings(tmp_path)
    args.provider_kind = kind
    args.job_max_requests = cap
    (args.source_dir / "a.mp4").write_bytes(b"original")
    client = ControlledAPI(args.run_dir, provider_kind=kind)
    with pytest.raises(ValueError, match="request cap"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []
    assert not (args.batch_dir / "manifest.json").exists()


@pytest.mark.parametrize("options", [["--job-max-requests", "401"],
    ["--provider-kind", "local", "--job-max-requests", "4001"],
    ["--provider-kind", "local", "--job-max-requests", "0"]])
def test_cli_rejects_invalid_job_request_allowance(options):
    with pytest.raises(SystemExit):
        batch.arguments(options)


def test_historical_model_responses_include_generic_and_revalidated_events_but_not_ocr(tmp_path):
    events = ["run_started", "cache_hit", "classifier_cache_hit", "classifier_cache_revalidated",
              "vision_cache_hit", "vision_cache_revalidated", "vision_paid_parse_recovered",
              "image_reader_cache_hit",
              "ocr_cache_hit", "orientation_probe_cache_hit", "registry_cache_hit", "run_completed"]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps({"event": event}) + "\n" for event in events))
    proof = batch.trace_observations(path)
    assert proof["historical_model_cache_hits"] == 7
    assert proof["invalid_trace_lines"] == proof["interrupted_trace_fragments"] == 0


@pytest.mark.parametrize("extra,issue", [
    ('{"event":', "malformed"),
    ('[]\n', "malformed"),
    (json.dumps({"event": "trace_interrupted_tail_preserved"}) + "\n", "interrupted"),
    (json.dumps({"event": "cache_hit"}) + "\n", "Historical model response cache"),
    (json.dumps({"event": "image_reader_cache_hit"}) + "\n", "Historical model response cache"),
])
def test_completed_output_cannot_claim_fresh_proof_with_broken_trace_or_cached_model(tmp_path, extra, issue):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")

    class AlteredTraceAPI(ControlledAPI):
        def request(self, method, endpoint, payload=None):
            result = super().request(method, endpoint, payload)
            if endpoint.endswith("/results"):
                path = self.run_dir / endpoint.split("/")[3] / "trace.jsonl"
                with path.open("a") as stream:
                    stream.write(extra)
            return result

    state = batch.BatchDriver(args, client=AlteredTraceAPI(args.run_dir)).run()
    assert state["status"] == "completed"  # Operational completion remains distinct.
    proof = state["jobs"][0]["proof"]
    assert proof["fresh_run_verified"] is False
    assert any(issue in message for message in proof["verification_issues"])


def test_preserved_interrupted_fragment_before_run_start_is_still_reported(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [{"event": "trace_interrupted_tail_preserved"}, {"event": "run_started"},
            {"event": "run_completed"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    proof = batch.trace_observations(path)
    assert proof["run_started_count"] == 1 and proof["run_completed"]
    assert proof["interrupted_trace_fragments"] == 1


def native_health(fingerprint="a" * 64):
    return {"status": "ok", "native_implementation": {"fingerprint_sha256": fingerprint}}


def test_native_batch_pins_health_and_passes_same_pin_to_each_job_and_resume(tmp_path):
    args = settings(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())
    client = ControlledAPI(args.run_dir)
    client.health = native_health()
    state = batch.BatchDriver(args, client=client).run()
    assert state["native_implementation_binding"] == {"status": "pinned", "fingerprint_sha256": "a" * 64}
    assert all(request["native_implementation_fingerprint"] == "a" * 64 for request in client.posts)
    assert all(job["api_state"]["native_implementation_fingerprint"] == "a" * 64 for job in state["jobs"])
    batch.BatchDriver(args, client=client).run()
    assert len(client.posts) == 2


@pytest.mark.parametrize("changed", [native_health("b" * 64), {"status": "ok"}])
def test_changed_or_missing_native_pin_refuses_resume_without_mutating_manifest(tmp_path, changed):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    client.health = native_health()
    batch.BatchDriver(args, client=client).run()
    saved = (args.batch_dir / "manifest.json").read_bytes()
    client.health = changed
    with pytest.raises(ValueError, match="implementation changed or its fingerprint is unavailable"):
        batch.BatchDriver(args, client=client)
    assert len(client.posts) == 1 and (args.batch_dir / "manifest.json").read_bytes() == saved


def test_implementation_change_between_completed_jobs_prevents_next_submission(tmp_path):
    args = settings(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        (args.source_dir / name).write_bytes(name.encode())

    class ChangedWorkerAPI(ControlledAPI):
        def request(self, method, endpoint, payload=None):
            result = super().request(method, endpoint, payload)
            if endpoint.endswith("/results"):
                self.health = native_health("b" * 64)
            return result

    client = ChangedWorkerAPI(args.run_dir)
    client.health = native_health()
    driver = batch.BatchDriver(args, client=client)
    with pytest.raises(ValueError, match="implementation changed"):
        driver.run()
    assert len(client.posts) == 1 and len(driver.state["jobs"]) == 1
    assert driver.state["jobs"][0]["finalized"]


def test_change_after_constructor_is_checked_before_run_or_any_submission(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    client.health = native_health()
    driver = batch.BatchDriver(args, client=client)
    client.health = native_health("b" * 64)
    with pytest.raises(ValueError, match="implementation changed"):
        driver.run()
    assert client.posts == [] and driver.state["status"] == "pending"


def test_used_unpinned_legacy_batch_cannot_be_retroactively_claimed_native_pinned(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    state = batch.BatchDriver(args, client=client).run()
    assert state["native_implementation_binding"]["status"] == "unpinned"
    assert "native_implementation_fingerprint" not in client.posts[0]
    client.health = native_health()
    with pytest.raises(ValueError, match="has no native implementation pin"):
        batch.BatchDriver(args, client=client)
    assert len(client.posts) == 1


def test_empty_unpinned_batch_can_pin_current_worker_before_its_first_job(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    driver = batch.BatchDriver(args, client=client)
    assert driver.state["native_implementation_binding"]["status"] == "unpinned"
    client.health = native_health()
    result = driver.run()
    assert result["native_implementation_binding"]["status"] == "pinned"
    assert client.posts[0]["native_implementation_fingerprint"] == "a" * 64


def test_legacy_missing_health_route_remains_explicitly_unpinned(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")

    class LegacyAPI(ControlledAPI):
        def request(self, method, endpoint, payload=None):
            if endpoint == "/health":
                raise batch.APIError("Legacy route missing", 404)
            return super().request(method, endpoint, payload)

    state = batch.BatchDriver(args, client=LegacyAPI(args.run_dir)).run()
    assert state["status"] == "completed"
    assert state["native_implementation_binding"]["status"] == "unpinned"


@pytest.mark.parametrize("provider", ["local", "remote"])
def test_unavailable_unknown_health_is_compatible_only_with_unpinned_remote_api(tmp_path, provider):
    args = settings(tmp_path)
    args.provider_kind = provider
    (args.source_dir / "a.mp4").write_bytes(b"source")

    class UnavailableHealthAPI(ControlledAPI):
        def request(self, method, endpoint, payload=None):
            if endpoint == "/health":
                raise batch.APIError("Legacy unknown health failure", 503)
            return super().request(method, endpoint, payload)

    client = UnavailableHealthAPI(args.run_dir, provider_kind=provider)
    if provider == "local":
        with pytest.raises(batch.APIError):
            batch.BatchDriver(args, client=client)
        assert client.posts == []
    else:
        state = batch.BatchDriver(args, client=client).run()
        assert state["status"] == "completed" and len(client.posts) == 1
        assert state["native_implementation_binding"]["status"] == "unpinned"


def test_pinned_remote_batch_still_refuses_unavailable_health(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    client.health = native_health()
    batch.BatchDriver(args, client=client)

    def unavailable(*args, **kwargs):
        raise batch.APIError("Health now unavailable", 503)

    client.request = unavailable
    with pytest.raises(ValueError, match="fingerprint is unavailable"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []


@pytest.mark.parametrize("fingerprint", ["bad", 123, "A" * 64])
def test_malformed_advertised_native_pin_stops_before_manifest_or_jobs(tmp_path, fingerprint):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    client.health = native_health(fingerprint)
    with pytest.raises(ValueError, match="invalid native implementation fingerprint"):
        batch.BatchDriver(args, client=client)
    assert client.posts == [] and not (args.batch_dir / "manifest.json").exists()


def test_corrupted_saved_pin_cannot_match_unreported_health_and_resume(tmp_path):
    args = settings(tmp_path)
    (args.source_dir / "a.mp4").write_bytes(b"source")
    client = ControlledAPI(args.run_dir)
    driver = batch.BatchDriver(args, client=client)
    driver.state["native_implementation_binding"] = {"status": "pinned"}
    driver.persist()
    with pytest.raises(ValueError, match="Stored native implementation pin is invalid"):
        batch.BatchDriver(args, client=client)
    assert client.posts == []
