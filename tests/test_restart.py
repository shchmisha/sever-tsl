import asyncio
from copy import deepcopy
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sever import agents, classifier, ocr, pipeline, report, video
from sever.agents import AgentConfig
from sever.schema import DocumentResult
from sever.tracing import append_trace_row, archive_restart_artifacts


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("fragment", [b'{"event":"unfinished', b'{"text":"\xd0', b'{"event":"broken"\n'])
def test_interrupted_terminal_bytes_preserved_and_both_trace_writers_restart_cleanly(tmp_path, fragment):
    path = tmp_path / "trace.jsonl"
    prefix = b'{"event":"first"}\n'
    path.write_bytes(prefix + fragment)
    pipeline.trace(path, "run_started", source="local")
    agents._trace(path, "cache_hit", "s", "p")
    events = rows(path)
    assert [row["event"] for row in events] == ["first", "trace_interrupted_tail_preserved", "run_started", "cache_hit"]
    preserved = Path(events[1]["raw_fragment_path"])
    assert preserved.read_bytes() == fragment
    assert path.read_bytes().startswith(prefix)
    assert path.read_bytes().endswith(b"\n")


def test_valid_terminal_record_without_newline_is_not_discarded(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"event":"existing","text":"Кадр"}')
    append_trace_row(path, {"event": "next"})
    assert [row["event"] for row in rows(path)] == ["existing", "next"]
    assert not list(tmp_path.glob("*.bin"))


def test_failed_fragment_preservation_never_truncates_original_trace(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    original = b'{"event":"ok"}\n{"event":"torn'
    path.write_bytes(original)
    original_open = Path.open
    def open_path(self, *args, **kwargs):
        if self.suffix == ".bin":
            raise OSError(errno.ENOSPC, "mock disk full")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError):
        append_trace_row(path, {"event": "new"})
    assert path.read_bytes() == original


def test_restart_renames_only_generated_small_artifacts(tmp_path):
    partial = {"pages": [{"page_id": str(i)} for i in range(3)]}
    (tmp_path / "partial-results.json").write_text(json.dumps(partial))
    (tmp_path / "metrics.json").write_text('{"old":true}')
    (tmp_path / "frames").mkdir();(tmp_path / "frames/original.jpg").write_bytes(b"source pixels")
    (tmp_path / "ocr-cache").mkdir();(tmp_path / "ocr-cache/cached.json").write_text('{}')
    history = archive_restart_artifacts(tmp_path, ("partial-results.json", "results.json", "metrics.json"))
    assert json.loads((history / "partial-results.json").read_text()) == partial
    assert (history / "metrics.json").is_file()
    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "frames/original.jpg").read_bytes() == b"source pixels"
    assert (tmp_path / "ocr-cache/cached.json").is_file()


def offline_pipeline(monkeypatch):
    pages = [{"page_id": "leaf", "start_s": 0., "end_s": 1., "selected_frame_id": None, "frames": []}]
    monkeypatch.setattr(video, "analyze_video", lambda *a, **kw: deepcopy(pages))
    def forbid_ocr(*a, **kw):
        raise AssertionError("Restart consistency tests must not run native OCR")
    monkeypatch.setattr(ocr, "read_image", forbid_ocr)
    async def classify(prepared, *, source_id, **kwargs):
        return {page["page_id"]: DocumentResult(source_id=source_id, page_id=page["page_id"], document_type="unknown") for page in prepared}
    monkeypatch.setattr(classifier, "classify_pages", classify)
    monkeypatch.setattr(report, "write_report", lambda out, result: (out / "report.html").write_text("offline report"))
    return AgentConfig(verify_inn=False)


def test_failed_restart_cannot_present_old_final_metrics_or_lose_three_leaf_checkpoint(tmp_path, monkeypatch):
    config = offline_pipeline(monkeypatch)
    source = tmp_path / "video.mp4";source.write_bytes(b"offline source")
    out = tmp_path / "out";out.mkdir()
    previous = {"pages": [{"page_id": str(i)} for i in range(3)]}
    (out / "partial-results.json").write_text(json.dumps(previous))
    (out / "results.json").write_text('{"completed":"old"}')
    (out / "metrics.json").write_text('{"rate":1}')
    async def failed(*a, **kw):
        raise RuntimeError("offline classification failure")
    monkeypatch.setattr(classifier, "classify_pages", failed)
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.process_video(source, out, config=config))
    assert not (out / "results.json").exists() and not (out / "metrics.json").exists()
    checkpoints = list((out / "history").glob("*/partial-results.json"))
    assert len(checkpoints) == 1 and json.loads(checkpoints[0].read_text()) == previous
    assert not any(row["event"] == "run_completed" for row in rows(out / "trace.jsonl"))


@pytest.mark.parametrize("failure", ["csv", "completion_trace"])
def test_failed_finalization_preserves_partial_and_removes_current_completed_artifacts(tmp_path, monkeypatch, failure):
    config = offline_pipeline(monkeypatch)
    source = tmp_path / "video.mp4";source.write_bytes(b"offline source")
    out = tmp_path / "out"
    if failure == "csv":
        def broken(*a, **kw):
            raise OSError(errno.ENOSPC, "mock disk full")
        monkeypatch.setattr(pipeline, "export_csv", broken)
    else:
        actual_trace = pipeline.trace
        def broken(path, event, **data):
            if event == "run_completed":
                raise OSError(errno.ENOSPC, "mock disk full")
            actual_trace(path, event, **data)
        monkeypatch.setattr(pipeline, "trace", broken)
    with pytest.raises(OSError):
        asyncio.run(pipeline.process_video(source, out, config=config))
    assert not (out / "results.json").exists()
    assert len(json.loads((out / "partial-results.json").read_text())["pages"]) == 1
    assert len(list((out / "history").glob("failed-finalization-*/results.json"))) == (1 if failure == "completion_trace" else 0)


def test_failed_pilot_restart_does_not_keep_prior_completed_summary(tmp_path, monkeypatch):
    from tools import run_pilot
    old = {"videos": [{"source": "old"}]}
    (tmp_path / "pilot-summary.json").write_text(json.dumps(old))
    async def failed(*a, **kw):
        raise OSError(errno.ENOSPC, "mock disk full")
    monkeypatch.setattr(run_pilot, "process_video", failed)
    args = SimpleNamespace(cache_only=True, model="offline", max_requests=1, max_tokens=1000,
                           out=tmp_path, truth=None, baseline=None, sources=[Path("offline.mp4")], interval=.5)
    with pytest.raises(OSError):
        asyncio.run(run_pilot.run(args))
    assert not (tmp_path / "pilot-summary.json").exists()
    assert json.loads(next((tmp_path / "history").glob("*/pilot-summary.json")).read_text()) == old
