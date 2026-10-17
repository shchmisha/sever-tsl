"""Initial leaf OCR concurrency must preserve source order and failure semantics."""
import asyncio
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident

import pytest

from sever import classifier, ocr, pipeline, quality, report, video
from sever.agents import AgentConfig
from sever.schema import DocumentResult


def fixture(monkeypatch, tmp_path, page_count=3, *, identical=False, reject_last=False):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"independent local OCR pool fixture")
    pages = []
    for n in range(page_count):
        frames = []
        for f in range(4):
            path = tmp_path / f"p{n}f{f}.png"
            path.write_bytes(b"same content" if identical else f"actual frame bytes {n}:{f}".encode())
            frames.append({"frame_id": path.stem, "path": str(path), "timestamp_s": n + f / 10,
                           "quality": {"relative_selection_score": 4 - f}})
        pages.append({"page_id": f"leaf-{n}", "start_s": n, "end_s": n + 1,
                      "selected_frame_id": frames[0]["frame_id"], "frames": frames})
    monkeypatch.setattr(video, "analyze_video", lambda *a, **k: deepcopy(pages))
    def gate(page):
        rejected = reject_last and page["page_id"] == f"leaf-{page_count-1}"
        return {"status": "rejected" if rejected else "accepted", "selected_frame_id": page["selected_frame_id"],
                "accepted_frame_ids": [] if rejected else [f["frame_id"] for f in page["frames"] if f["frame_id"] != "p0f1"]}
    monkeypatch.setattr(quality, "evaluate_page_frames", gate)
    monkeypatch.setattr(report, "write_report", lambda out, result: (out / "report.html").write_text("fixture"))
    return source


def observation():
    return {"engine": "controlled-local-OCR", "regions": [{"text": "outside required forms", "bbox": [.1, .1, .3, .02]}],
            "processing": {"full_page_visible": True}}


def classify_capture(monkeypatch, captured, assert_ready=lambda: None):
    async def classify(prepared, *, source_id, **kwargs):
        assert_ready()
        captured.append(deepcopy(prepared))
        return {p["page_id"]: DocumentResult(source_id=source_id, page_id=p["page_id"], document_type="unknown")
                for p in prepared}
    monkeypatch.setattr(classifier, "classify_pages", classify)


def execute(monkeypatch, source, out, workers, progress=None):
    monkeypatch.setenv("SEVER_OCR_WORKERS", str(workers))
    return asyncio.run(pipeline.process_video(source, out, config=AgentConfig(verify_inn=False), progress=progress))


def initial_events(out):
    events = {"page_selected", "ocr_read", "ocr_failed", "ocr_empty", "ocr_fallback_selected"}
    return [{k: v for k, v in row.items() if k != "time"}
            for row in map(json.loads, (out / "trace.jsonl").read_text().splitlines()) if row["event"] in events]


def test_pool_matches_serial_selection_attempts_classification_and_original_trace_order(monkeypatch, tmp_path):
    source = fixture(monkeypatch, tmp_path, reject_last=True)
    captured = []
    classify_capture(monkeypatch, captured)
    def reader(path, *args, **kwargs):
        if path.stem == "p0f0":
            raise RuntimeError("operational OCR failure")
        if path.stem == "p0f2":
            return {"engine": "controlled-local-OCR", "regions": [], "processing": {}}
        return observation()
    monkeypatch.setattr(ocr, "read_image", reader)
    serial_progress, pool_progress = [], []
    serial = execute(monkeypatch, source, tmp_path / "serial", 1, serial_progress.append)
    pool = execute(monkeypatch, source, tmp_path / "pool", 4, pool_progress.append)
    assert captured[0] == captured[1]
    projection = lambda result: [{k: p[k] for k in ("page_id", "selected_frame_id", "quality_gate", "ocr_attempts", "result")}
                                 for p in result["pages"]]
    assert projection(serial) == projection(pool)
    assert initial_events(tmp_path / "serial") == initial_events(tmp_path / "pool")
    assert serial_progress == pool_progress
    assert serial["usage"] == pool["usage"]
    assert pool["configuration"]["local_ocr_workers"] == 4
    assert pool["pages"][0]["selected_frame_id"] == "p0f3"
    assert [a["frame_id"] for a in pool["pages"][0]["ocr_attempts"]] == ["p0f0", "p0f2", "p0f3"]


def test_distinct_leaves_overlap_but_progress_and_traces_stay_on_coordinator(monkeypatch, tmp_path):
    source = fixture(monkeypatch, tmp_path, page_count=2)
    rendezvous = Barrier(2, timeout=5)
    second_done = Event()
    completed = []
    caller = get_ident()
    io_threads = []
    original_trace = pipeline.trace
    def tracked_trace(*args, **kwargs):
        io_threads.append(get_ident())
        return original_trace(*args, **kwargs)
    monkeypatch.setattr(pipeline, "trace", tracked_trace)
    def reader(path, *args, **kwargs):
        assert get_ident() != caller
        rendezvous.wait()
        if path.stem == "p0f0":
            assert second_done.wait(5)
        else:
            completed.append(path.stem)
            second_done.set()
        return observation()
    monkeypatch.setattr(ocr, "read_image", reader)
    captured = []
    classify_capture(monkeypatch, captured, lambda: (rendezvous.n_waiting == 0 and second_done.is_set()) or pytest.fail("classification before local OCR"))
    result = execute(monkeypatch, source, tmp_path / "pool", 4, lambda message: io_threads.append(get_ident()))
    assert [p["page_id"] for p in captured[0]] == ["leaf-0", "leaf-1"]
    assert {get_ident()} == set(io_threads)
    assert completed == ["p1f0"]
    assert [(e["event"], e.get("frame_id")) for e in initial_events(tmp_path / "pool") if e["event"] == "ocr_read"] == [
        ("ocr_read", "p0f0"), ("ocr_read", "p1f0")]
    assert all(not p["result"]["errors"] for p in result["pages"])


def test_duplicate_content_is_serialized_even_with_different_frame_paths(monkeypatch, tmp_path):
    source = fixture(monkeypatch, tmp_path, page_count=4, identical=True)
    current = peak = 0
    counter = Lock()
    calls = []
    def reader(path, *args, **kwargs):
        nonlocal current, peak
        with counter:
            current += 1
            peak = max(peak, current)
            calls.append(path.stem)
        # A bounded wait releases the GIL and exposes overlapping unprotected
        # cache writes; every caller must still bind its own frame metadata.
        Event().wait(.025)
        with counter:
            current -= 1
        return observation()
    monkeypatch.setattr(ocr, "read_image", reader)
    captured = []
    classify_capture(monkeypatch, captured, lambda: current == 0 or pytest.fail("OCR still active"))
    execute(monkeypatch, source, tmp_path / "pool", 4)
    assert peak == 1 and len(calls) == 4
    assert [p["ocr_pages"][0]["frame_id"] for p in captured[0]] == [f"p{i}f0" for i in range(4)]


@pytest.mark.parametrize("workers", [1, 4])
def test_every_local_failure_stops_before_model_or_final_results(monkeypatch, tmp_path, workers):
    source = fixture(monkeypatch, tmp_path, page_count=2)
    calls = defaultdict(list)
    counter = Lock()
    def reader(path, *args, **kwargs):
        with counter:
            calls[path.stem[:2]].append(path.stem)
        raise RuntimeError("private error detail must not enter trace")
    monkeypatch.setattr(ocr, "read_image", reader)
    captured = []
    classify_capture(monkeypatch, captured)
    out = tmp_path / "out"
    with pytest.raises(pipeline.LocalOCRUnavailable):
        execute(monkeypatch, source, out, workers)
    assert captured == [] and not (out / "results.json").exists()
    assert calls["p0"] == ["p0f0", "p0f2", "p0f3"]
    assert calls["p1"] == ["p1f0", "p1f1", "p1f2", "p1f3"]
    rows = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["reason"] == "local_ocr_unavailable" and rows[-1]["model_requests_attempted"] == 0
    assert "private error detail" not in (out / "trace.jsonl").read_text()


@pytest.mark.parametrize("value", ["0", "-1", "33", "not-an-int"])
def test_invalid_pool_setting_fails_before_any_model_or_local_read(monkeypatch, tmp_path, value):
    source = fixture(monkeypatch, tmp_path)
    monkeypatch.setenv("SEVER_OCR_WORKERS", value)
    calls = []
    monkeypatch.setattr(ocr, "read_image", lambda *args, **kwargs: calls.append(args))
    classify_capture(monkeypatch, calls)
    with pytest.raises(ValueError, match="SEVER_OCR_WORKERS"):
        asyncio.run(pipeline.process_video(source, tmp_path / "out", config=AgentConfig(verify_inn=False)))
    assert calls == []
