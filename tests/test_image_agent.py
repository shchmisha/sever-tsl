import asyncio
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError
from sever.image_agent import _Source, read_frame_agent
from sever.image_detail import _rotate, _source_box


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "private-filename-never-supplied.png"
    image = Image.new("RGB", (1600, 1000))
    image.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(1000) for x in range(1600)])
    image.save(path)
    return {"frame_id": "private-frame-id", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def config(**kwargs):
    return AgentConfig(model_name="offline-test", vision_model_name="offline-test", provider_kind="local",
                       base_url="http://127.0.0.1:8090/v1", document_reader="native_image", **kwargs)


def reading(panel_id, text="Visible synthetic heading"):
    return {"reading": {"document_type": "unknown", "type_confidence": .8,
        "observed_form": text, "visible_text": text, "heading_text": text,
        "heading_panel_ids": [panel_id], "complete_page_visible": False,
        "form_scope": "insufficient_evidence", "fields": {}}}


def content(messages):
    return [item for message in messages for part in message.parts
            if isinstance(getattr(part, "content", None), list) for item in part.content]


def last_panel(messages):
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
                return part.content["panel_id"]
    descriptions = [item for item in content(messages) if isinstance(item, str) and item.startswith("Actual source panel_id=")]
    return descriptions[-1].split("=", 1)[1].split(";", 1)[0]


def response(payload):
    return ModelResponse(parts=[ToolCallPart("submit_reading", payload, tool_call_id="final-reading")],
                         usage=RequestUsage(input_tokens=50, output_tokens=20))


def zoom_call(call_id="crop-1", bbox=None):
    return ModelResponse(parts=[ToolCallPart("zoom_source", bbox or {"left": .2, "top": .15, "right": .55, "bottom": .45},
                                            tool_call_id=call_id)],
                         usage=RequestUsage(input_tokens=30, output_tokens=10))


def run(frame, tmp_path, fn, *, settings=None, rotation=90):
    return asyncio.run(read_frame_agent(frame, source_id="private-source-id", page_id="private-page-id",
        config=settings or config(), cache_dir=tmp_path / "evidence", trace_path=tmp_path / "trace.jsonl",
        rotation=rotation, model=FunctionModel(fn, model_name="offline-function-model")))


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_native_crop_pixels_and_source_geometry_round_trip(frame, tmp_path, rotation):
    source = _Source(frame, tmp_path / "evidence", rotation)
    identifier, data, panel = source.zoom(.203, .157, .553, .457)
    with Image.open(frame["path"]) as original:
        expected = _rotate(original, rotation).crop(panel["upright_bbox_pixels_top_left"])
        actual = Image.open(BytesIO(data))
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()
    assert panel["source_bbox_pixels_top_left"] == list(_source_box(panel["upright_bbox_pixels_top_left"], (1600, 1000), rotation))
    assert panel["upload_sha256"] == hashlib.sha256(data).hexdigest()
    assert Path(panel["upload_path"]).read_bytes() == data
    assert panel["source_sha256"] == frame["source_sha256"]
    assert panel["prep"]["resize"] == "none" and panel["prep"]["format"] == "PNG"
    left, top, right, bottom = panel["source_bbox_pixels_top_left"]
    points = []
    for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]:
        u, v = [r[0]*x + r[1]*y + r[2] for r in panel["source_to_upload_pixels"][:2]]
        points.append((u, v))
        assert [r[0]*u + r[1]*v + r[2] for r in panel["upload_to_source_pixels"][:2]] == pytest.approx([x, y])
    assert min(p[0] for p in points) == pytest.approx(0)
    assert min(p[1] for p in points) == pytest.approx(0)
    assert max(p[0] for p in points) == pytest.approx(actual.width)
    assert max(p[1] for p in points) == pytest.approx(actual.height)


def test_tool_output_two_actual_requests_and_full_proof_without_native_grammar(frame, tmp_path):
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(deepcopy(messages))
        assert [tool.name for tool in info.output_tools] == ["submit_reading"]
        assert [tool.name for tool in info.function_tools] == ["zoom_source", "zoom_regions"]
        assert info.model_request_parameters.output_mode == "tool"
        assert info.model_request_parameters.output_object is None
        assert info.output_tools[0].strict is False
        assert "reading" in info.output_tools[0].parameters_json_schema["properties"]
        if len(calls) == 1:
            return zoom_call()
        images = [item for item in content(messages) if isinstance(item, BinaryContent)]
        assert len(images) == 2 and images[-1].media_type == "image/png"
        return response(reading(last_panel(messages)))
    result = run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == settings.usage.attempted_requests == 2
    assert settings.usage.input_tokens == 80 and settings.usage.output_tokens == 30
    assert settings.usage.tool_calls == 1
    assert result["recognition"]["zoom_images"] == 1
    proof = result["recognition"]["pixel_proof"]
    first_image = next(item for item in content(calls[0]) if isinstance(item, BinaryContent))
    assert Image.open(BytesIO(first_image.data)).size == (640, 1024)
    images = [item for item in content(calls[1]) if isinstance(item, BinaryContent)]
    for actual, panel in zip(images, proof["panels"].values(), strict=True):
        assert hashlib.sha256(actual.data).hexdigest() == panel["upload_sha256"]
        assert actual.data == Path(panel["upload_path"]).read_bytes()
    evidence = json.loads(next((tmp_path / "evidence").glob("image-agent-*.json")).read_text())
    assert len(evidence["raw_responses"]) == 2 and len(evidence["tool_proofs"]) == 1
    assert evidence["raw_responses"][0]["parts"][0]["tool_name"] == "zoom_source"
    assert evidence["raw_responses"][1]["parts"][0]["tool_name"] == "submit_reading"
    assert evidence["decision"]["heading_panel_ids"] == [evidence["tool_proofs"][0]["panel_id"]]
    assert sum(row["event"] == "model_response_received" for row in rows(tmp_path)) == 2
    assert sum(row["event"] == "image_agent_zoom" for row in rows(tmp_path)) == 1
    assert hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest() == frame["source_sha256"]
    for messages in calls:
        sent_text = "\n".join(str(item) for item in content(messages) if isinstance(item, str))
        for private in (frame["path"], frame["frame_id"], frame["source_sha256"], "private-source-id", "private-page-id"):
            assert private not in sent_text


def test_batched_regions_deliver_all_native_pixels_in_one_model_round_trip(frame, tmp_path):
    calls, settings = [], config()
    regions = [{"left": .1, "top": .15, "right": .4, "bottom": .35},
               {"left": .2, "top": .5, "right": .7, "bottom": .8}]
    def respond(messages, info):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("zoom_regions", {"regions": regions}, tool_call_id="batch")],
                                 usage=RequestUsage(input_tokens=30, output_tokens=10))
        returned = [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)][-1]
        assert len(returned["panels"]) == 2
        images = [x for x in content(messages) if isinstance(x, BinaryContent)]
        assert len(images) == 3
        with Image.open(frame["path"]) as original:
            upright = _rotate(original, 90)
            for image, region in zip(images[1:], regions, strict=True):
                import math
                box = (math.floor(region["left"] * upright.width), math.floor(region["top"] * upright.height),
                       math.ceil(region["right"] * upright.width), math.ceil(region["bottom"] * upright.height))
                assert Image.open(BytesIO(image.data)).tobytes() == upright.crop(box).tobytes()
        return response(reading(returned["panels"][0]["panel_id"]))
    result = run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == 2
    assert settings.usage.tool_calls == 1
    assert result["recognition"]["zoom_images"] == 2
    assert len(result["recognition"]["pixel_proof"]["panels"]) == 3


@pytest.mark.parametrize("shared_limit,expected", [(100, AnalysisError), (8, BatchBudgetExceeded)])
def test_local_model_request_allowance_does_not_claim_global_exhaustion(frame, tmp_path, monkeypatch, shared_limit, expected):
    from sever import image_agent
    from pydantic_ai.exceptions import UsageLimitExceeded
    settings = config(max_requests=shared_limit)
    async def exhaust(self, *args, deps, usage, **kwargs):
        # The run counts its own requests; the shared hooks record attempts.
        deps.requests_started = 8
        usage.requests = deps.config.usage.attempted_requests = 8
        raise UsageLimitExceeded("The next request would exceed the request_limit of 8")
    monkeypatch.setattr(image_agent.Agent, "run", exhaust)
    with pytest.raises(expected) as caught:
        run(frame, tmp_path, lambda *_: None, settings=settings)
    if shared_limit == 100:
        assert isinstance(caught.value, image_agent.ViewBudgetExceeded)
        assert not isinstance(caught.value, BatchBudgetExceeded)
    evidence = json.loads(next((tmp_path / "evidence").glob("image-agent-*.json")).read_text())
    assert evidence["status"] == ("view_budget_exhausted" if shared_limit == 100 else "budget_exhausted")


def test_lost_evidence_is_not_disguised_as_recoverable_view_budget(frame, tmp_path, monkeypatch):
    from sever import image_agent
    from pydantic_ai.exceptions import UsageLimitExceeded
    settings = config(max_requests=100)
    async def exhaust(self, *args, deps, usage, **kwargs):
        deps.requests_started = 8
        usage.requests = usage.attempted_requests = 8
        raise UsageLimitExceeded("Local request limit")
    def failing_store(*args, **kwargs):
        raise OSError("private-storage-detail")
    monkeypatch.setattr(image_agent.Agent, "run", exhaust)
    monkeypatch.setattr(image_agent, "_write_evidence", failing_store)
    with pytest.raises(image_agent.EvidencePersistenceError) as caught:
        run(frame, tmp_path, lambda *_: None, settings=settings)
    assert not isinstance(caught.value, image_agent.ViewBudgetExceeded)
    assert "private-" not in str(caught.value)
    assert settings.usage.requests == 8


def test_repeated_run_reads_no_previous_answers_and_uses_unique_evidence_files(frame, tmp_path, monkeypatch):
    calls = []
    old = tmp_path / "evidence" / "image-agent-historical.json"
    old.parent.mkdir()
    old.write_text('{"decision": "STALE-ANSWER-SENTINEL"}')
    read_bytes, read_text = Path.read_bytes, Path.read_text
    def checked_bytes(path, *args, **kwargs):
        assert path == Path(frame["path"]), "Reader tried to load a historical answer or uploaded crop"
        return read_bytes(path, *args, **kwargs)
    def no_text_reads(*args, **kwargs):
        raise AssertionError("Reader loaded a saved text answer")
    def respond(messages, info):
        calls.append(deepcopy(messages))
        assert "STALE-ANSWER-SENTINEL" not in str(messages)
        return response(reading(last_panel(messages), f"Fresh response {len(calls)}"))
    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(Path, "read_text", no_text_reads)
    first = run(frame, tmp_path, respond)
    second = run(frame, tmp_path, respond)
    assert len(calls) == 2
    assert first["recognition"]["reading"]["visible_text"] == "Fresh response 1"
    assert second["recognition"]["reading"]["visible_text"] == "Fresh response 2"
    first_paths = {panel["upload_path"] for panel in first["panels"].values()}
    second_paths = {panel["upload_path"] for panel in second["panels"].values()}
    assert first_paths.isdisjoint(second_paths)
    assert "STALE-ANSWER-SENTINEL" in read_text(old)


@pytest.mark.parametrize("kind", ["token_preflight", "token_response"])
def test_budget_stops_charge_real_calls_and_never_replay(frame, tmp_path, kind):
    calls = []
    settings = config(**{"requests": {"max_requests": 1}, "token_preflight": {"max_tokens": 1},
                         "token_response": {"input_tokens_limit": 29}, "request_cap": {}}[kind])
    def respond(messages, info):
        calls.append(1)
        return zoom_call(str(len(calls)))
    with pytest.raises(BatchBudgetExceeded):
        run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == {"requests": 1, "token_preflight": 0, "token_response": 1, "request_cap": 8}[kind]
    assert settings.usage.requests == len(calls)
    assert settings.usage.input_tokens == 30 * len(calls)
    assert settings.usage.output_tokens == 10 * len(calls)
    records = list((tmp_path / "evidence").glob("image-agent-*.json"))
    if calls:
        assert len(records) == 1
        evidence = json.loads(records[0].read_text())
        assert evidence["status"] == "budget_exhausted"
        assert len(evidence["raw_responses"]) == len(calls)
    else:
        assert records == []
    assert sum(row["event"] == "model_response_received" for row in rows(tmp_path)) == len(calls)


def test_crop_cap_is_explicit_and_does_not_supply_reused_images(frame, tmp_path):
    source = _Source(frame, tmp_path, 0)
    identifiers = [source.zoom(0, 0, .1, .1)[0] for _ in range(8)]
    with pytest.raises(AnalysisError, match="crop limit"):
        source.zoom(0, 0, .1, .1)
    assert len(set(identifiers)) == len(source.proof["panels"]) == 8


@pytest.mark.parametrize("stage", ["expected_sha", "after_first_response", "after_final_response"])
def test_changed_source_cannot_supply_another_crop_or_final_answer(frame, tmp_path, stage):
    calls, settings = [], config()
    if stage == "expected_sha":
        frame["source_sha256"] = "0" * 64
    def respond(messages, info):
        calls.append(1)
        Image.new("RGB", (1600, 1000), "yellow").save(frame["path"])
        return zoom_call() if stage == "after_first_response" else response(reading(last_panel(messages)))
    with pytest.raises(AnalysisError):
        run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == (0 if stage == "expected_sha" else 1)
    assert settings.usage.requests == len(calls)
    assert not any(row["event"] == "image_agent_zoom" for row in rows(tmp_path)) if calls else True


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_fatal_provider_status_is_preserved_even_when_failure_evidence_breaks(frame, tmp_path, monkeypatch, status):
    from sever import image_agent
    settings, calls = config(), []
    trace = image_agent._trace
    def flaky_trace(path, event, *args, **kwargs):
        if event == "image_agent_error":
            raise OSError("storage failed")
        return trace(path, event, *args, **kwargs)
    def broken_store(*args, **kwargs):
        raise OSError("storage failed")
    monkeypatch.setattr(image_agent, "_trace", flaky_trace)
    monkeypatch.setattr(image_agent, "_write_evidence", broken_store)
    def respond(messages, info):
        calls.append(1)
        raise ModelHTTPError(status, "offline-test", "private-provider-body")
    with pytest.raises(FatalProviderError, match=f"HTTP {status}") as exc:
        run(frame, tmp_path, respond, settings=settings)
    assert "private-provider-body" not in str(exc.value)
    assert len(calls) == settings.usage.requests == settings.usage.attempted_requests == 1


def test_malformed_output_tool_preserves_raw_tool_response_and_final_response(frame, tmp_path):
    calls = []
    def respond(messages, info):
        calls.append(1)
        return zoom_call() if len(calls) == 1 else response({"reading": {"malformed": True}})
    with pytest.raises(AnalysisError):
        run(frame, tmp_path, respond)
    assert len(calls) == 2
    evidence = json.loads(next((tmp_path / "evidence").glob("image-agent-*.json")).read_text())
    assert evidence["status"] == "failed"
    assert len(evidence["raw_responses"]) == 2 and len(evidence["tool_proofs"]) == 1


@pytest.mark.parametrize("kind", ["fatal", "budget", "ordinary", "success"])
def test_cleanup_cannot_mask_existing_stop_and_is_visible_after_success(frame, tmp_path, monkeypatch, kind):
    from sever import image_agent
    calls, clients, settings = [], [], config(input_tokens_limit=29) if kind == "budget" else config()
    class Client:
        def __init__(self, **kwargs):
            self.closed = 0
            clients.append(self)
        async def close(self):
            self.closed += 1
            raise OSError("private-cleanup-detail")
    def respond(messages, info):
        calls.append(1)
        if kind == "fatal":
            raise FatalProviderError("private-provider-detail")
        if kind == "ordinary":
            raise RuntimeError("private-provider-detail")
        return response(reading(last_panel(messages)))
    monkeypatch.setattr(image_agent, "AsyncOpenAI", Client)
    monkeypatch.setattr(image_agent, "OpenAIProvider", lambda **kwargs: object())
    monkeypatch.setattr(image_agent, "_RouterAIChatModel", lambda *args, **kwargs: FunctionModel(respond))
    expected = {"fatal": FatalProviderError, "budget": BatchBudgetExceeded}.get(kind, AnalysisError)
    match = {"fatal": "provider failure", "budget": "budget exhausted", "ordinary": "RuntimeError", "success": "Cannot close"}[kind]
    with pytest.raises(expected, match=match) as exc:
        asyncio.run(read_frame_agent(frame, source_id="s", page_id="p", config=settings,
            cache_dir=tmp_path / "evidence", trace_path=tmp_path / "trace.jsonl", rotation=0))
    assert len(clients) == clients[0].closed == len(calls) == settings.usage.requests == 1
    assert "private-" not in str(exc.value)


@pytest.mark.parametrize("kind", ["budget", "ordinary"])
def test_failure_storage_error_preserves_budget_and_reports_ordinary_failure(frame, tmp_path, monkeypatch, kind):
    from sever import image_agent
    settings = config(input_tokens_limit=29) if kind == "budget" else config()
    original_trace = image_agent._trace
    def failing_trace(path, event, *args, **kwargs):
        if event == "image_agent_error":
            raise OSError("private-storage-detail")
        return original_trace(path, event, *args, **kwargs)
    def failing_store(*args, **kwargs):
        raise OSError("private-storage-detail")
    monkeypatch.setattr(image_agent, "_trace", failing_trace)
    monkeypatch.setattr(image_agent, "_write_evidence", failing_store)
    def respond(messages, info):
        if kind == "ordinary":
            raise RuntimeError("private-provider-detail")
        return response(reading(last_panel(messages)))
    with pytest.raises(BatchBudgetExceeded if kind == "budget" else AnalysisError,
                       match="budget exhausted" if kind == "budget" else "Cannot preserve") as exc:
        run(frame, tmp_path, respond, settings=settings)
    assert "private-" not in str(exc.value)
    assert settings.usage.requests == 1


def test_parallel_crop_overflow_returns_explicit_no_image_then_submits(frame, tmp_path):
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[zoom_call(f"zoom-{i}").parts[0] for i in range(9)],
                                 usage=RequestUsage(input_tokens=30, output_tokens=10))
        assert info.function_tools == []
        assert [tool.name for tool in info.output_tools] == ["submit_reading"]
        returns = [part.content for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart)]
        assert len(returns) == 9
        assert returns[-1]["status"] == "zoom_limit_reached"
        assert returns[-1]["no_image_returned"] is True
        assert "panel_id" not in returns[-1]
        images = [item for item in content(messages) if isinstance(item, BinaryContent)]
        assert len(images) == 9  # one context plus eight real crops
        return response(reading(returns[-2]["panel_id"]))
    result = run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == 2
    assert result["recognition"]["zoom_images"] == 8
    proof = result["recognition"]["pixel_proof"]
    assert len(proof["panels"]) == 9
    evidence = json.loads(next((tmp_path / "evidence").glob("image-agent-*.json")).read_text())
    assert evidence["status"] == "success"
    assert len(evidence["tool_proofs"]) == 8 and len(evidence["tool_declines"]) == 1
    assert sum(row["event"] == "image_agent_zoom_declined" for row in rows(tmp_path)) == 1


def test_last_request_reserves_submission_without_another_zoom(frame, tmp_path):
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(1)
        if len(calls) < 8:
            assert [tool.name for tool in info.function_tools] == ["zoom_source", "zoom_regions"]
            return zoom_call(f"zoom-{len(calls)}")
        assert info.function_tools == []
        assert [tool.name for tool in info.output_tools] == ["submit_reading"]
        return response(reading(last_panel(messages)))
    result = run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == settings.usage.requests == 8
    assert result["recognition"]["zoom_images"] == 7


def test_last_shared_request_reserves_submission_and_charges_actual_call(frame, tmp_path):
    calls, settings = [], config(max_requests=3)
    settings.usage.requests = settings.usage.attempted_requests = 2
    def respond(messages, info):
        calls.append(1)
        assert info.function_tools == []
        return response(reading(last_panel(messages)))
    result = run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == 1 and settings.usage.requests == 3
    assert result["recognition"]["zoom_images"] == 0
    with pytest.raises(BatchBudgetExceeded):
        run(frame, tmp_path, respond, settings=settings)
    assert len(calls) == 1 and settings.usage.requests == 3


def test_concurrent_frame_reads_keep_their_own_request_allowance(frame, tmp_path):
    """Two interleaved agent runs share the batch budget, never each other's cap."""
    settings = config(max_requests=40)

    def model():
        calls = []
        async def respond(messages, info):
            calls.append(1)
            await asyncio.sleep(0)
            if len(calls) <= 6:
                return zoom_call(call_id=f"crop-{len(calls)}")
            return response(reading(last_panel(messages)))
        return FunctionModel(respond, model_name="offline-function-model")

    async def both():
        return await asyncio.gather(*(read_frame_agent(frame, source_id="s", page_id="p", config=settings,
            cache_dir=tmp_path / f"evidence-{n}", trace_path=tmp_path / "trace.jsonl", rotation=90,
            model=model()) for n in range(2)))

    results = asyncio.run(both())
    # Each run makes 7 requests; with a shared-relative cap the second run would
    # have hit 8 after the first run's requests and stopped as a budget failure.
    assert [r["recognition"]["model_requests"] for r in results] == [7, 7]
    assert settings.usage.requests == settings.usage.attempted_requests == 14
    assert settings.usage.tool_calls == 12
    for n in range(2):
        evidence = json.loads(next((tmp_path / f"evidence-{n}").glob("image-agent-*.json")).read_text())
        assert evidence["usage_delta"] == {"requests": 7, "input_tokens": 6 * 30 + 50,
                                           "output_tokens": 6 * 10 + 20, "tool_calls": 6}
