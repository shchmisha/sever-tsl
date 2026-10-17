"""Run comparison works on artifacts of any native_image version."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_runs  # noqa: E402

SHA = "a" * 64


def write_run(directory, rows, *, pages, elapsed=100.0, complete=True, sha=SHA, configuration=None):
    directory.mkdir(parents=True)
    with (directory / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    results = {"source": {"sha256": sha}, "elapsed_seconds": elapsed, "pages": pages,
               "configuration": configuration or {}}
    name = "results.json" if complete else "partial-results.json"
    (directory / name).write_text(json.dumps(results, ensure_ascii=False))
    return directory


def page(page_id, kind="upd", stop="all_required_fields_confirmed", statuses=("recognized",) * 6,
         start=0.0, end=2.0, frame_time=1.0, values=None):
    names = ["invoice_number", "invoice_date", "consignor", "consignor_address", "consignee", "consignee_address"]
    values = values or {"invoice_number": "42"}
    fields = {name: {"status": status, "raw_value": values.get(name, "x") if status == "recognized" else None,
                     "normalized_value": values.get(name, "x") if status == "recognized" else None, "evidence": []}
              for name, status in zip(names, statuses)} if kind == "upd" else {}
    return {"page_id": page_id, "start_s": start, "end_s": end, "selected_frame_id": "f1",
            "frames": [{"frame_id": "f1", "timestamp_s": frame_time}],
            "result": {"document_type": kind, "fields": fields, "errors": []},
            "image_recognition": {"stop_reason": stop, "views": [{}, {}]}}


EPOCH = 1_792_000_000.0


def iso(seconds):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(EPOCH + seconds, timezone.utc).isoformat()


def calls(page_id, start, kinds):
    """Trace rows for one leaf, in both trace time formats: pipeline epoch and hook ISO."""
    rows, moment = [{"time": EPOCH + start, "event": "page_selected", "page_id": page_id}], start
    for event, requests in kinds:
        rows.append({"time": iso(moment), "event": event, "page_id": page_id})
        for _ in range(requests):
            rows.append({"time": iso(moment), "event": "model_request_started", "page_id": page_id})
            moment += 2
            rows.append({"time": iso(moment), "event": "model_response_received",
                         "page_id": page_id, "input_tokens": 1000, "output_tokens": 100})
    return rows


def test_main_style_trace_is_attributed_by_call_kind_and_timed_from_timestamps(tmp_path):
    rows = calls("p1", 0.0, [("image_orientation_observation", 1), ("image_agent_observation", 3),
                             ("image_orientation_observation", 1), ("image_agent_observation", 2)])
    totals, leaves = compare_runs.trace_stats(compare_runs.read_trace(write_run(tmp_path / "a", rows, pages=[]) / "trace.jsonl"))
    assert totals["requests"] == totals["responses"] == 7
    assert totals["call_kinds"] == {"orientation": 2, "field_agent": 5}
    assert totals["model_seconds"] == 14.0 and totals["input_tokens"] == 7000
    assert "server_timings" not in totals
    assert leaves["p1"]["leaf_seconds"] == 14.0


def test_branch_style_trace_uses_recorded_elapsed_and_server_timings(tmp_path):
    rows = [{"time": EPOCH, "event": "page_selected", "page_id": "p1"},
            {"time": "2026-10-17T00:00:00+00:00", "event": "image_triage_observation", "page_id": "p1"},
            {"time": "2026-10-17T00:00:00+00:00", "event": "model_request_started", "page_id": "p1"},
            {"time": "2026-10-17T00:00:09+00:00", "event": "model_response_received", "page_id": "p1",
             "input_tokens": 1200, "output_tokens": 20, "elapsed_seconds": 1.5,
             "server_timings": {"cache_n": 900, "prompt_n": 300, "prompt_ms": 100.0, "predicted_n": 20, "predicted_ms": 400.0}}]
    totals, _ = compare_runs.trace_stats(compare_runs.read_trace(write_run(tmp_path / "b", rows, pages=[]) / "trace.jsonl"))
    assert totals["call_kinds"] == {"triage": 1} and totals["model_seconds"] == 1.5
    assert totals["server_timings"] == {"responses": 1, "cache_n": 900, "prompt_n": 300, "prompt_ms": 100.0,
                                        "predicted_n": 20, "predicted_ms": 400.0}


def truth(tmp_path, consignee_address="Тула"):
    names = {"invoice_number": "42", "consignee_address": consignee_address}
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({"videos": [{"source_id": "v", "source_sha256": SHA, "pages": [
        {"page_id": "v-1", "start_s": 0.0, "end_s": 2.0, "document_type": "upd",
         "fields": {name: {"value": value, "status": "recognized"} for name, value in names.items()}},
        {"page_id": "v-2", "start_s": 2.0, "end_s": 4.0, "document_type": "unknown", "fields": {}}]}]},
        ensure_ascii=False))
    return path


def test_compare_reports_ratios_accuracy_and_changed_fields(tmp_path):
    a_rows = calls("p1", 0.0, [("image_orientation_observation", 1), ("image_agent_observation", 6)])
    b_rows = calls("p1", 100.0, [("image_orientation_observation", 1), ("image_triage_observation", 1),
                                 ("image_agent_observation", 2)])
    a = write_run(tmp_path / "a", a_rows, elapsed=200.0, pages=[page("p1", values={"invoice_number": "42", "consignee_address": "Тула"}),
                                                              page("p2", kind="unknown", start=2.0, end=4.0, frame_time=3.0)])
    b = write_run(tmp_path / "b", b_rows, elapsed=50.0, pages=[page("p1", values={"invoice_number": "42", "consignee_address": "Тверь"}),
                                                             page("p2", kind="unknown", start=2.0, end=4.0, frame_time=3.0)],
                  configuration={"leaf_concurrency": 2})
    report = compare_runs.compare(a, b, label_a="main", label_b="phase1", truth=truth(tmp_path))
    assert report["ratio_b_over_a"]["elapsed_seconds"] == 0.25
    assert report["ratio_b_over_a"]["model_requests"] == pytest.approx(4 / 7, abs=1e-3)
    main, phase1 = report["runs"]["main"], report["runs"]["phase1"]
    assert main["accuracy"]["labelled_fields"] == {"correct": 2, "total": 2}
    assert phase1["accuracy"]["labelled_fields"] == {"correct": 1, "total": 2}
    assert phase1["accuracy"]["unknown_rejection"] == {"correct": 1, "total": 1}
    assert phase1["summary"]["leaf_concurrency"] == 2 and main["summary"]["leaf_concurrency"] == 1
    text = compare_runs.markdown(report)
    assert "| Model requests | 7 | 4 |" in text
    assert "v-1 consignee_address: main=ok, phase1=wrong" in text
    assert not any("overlapped" in note for note in report["notes"])


def test_overlapping_or_unfinished_runs_are_flagged_and_not_scored(tmp_path):
    rows = calls("p1", 0.0, [("image_agent_observation", 2)])
    a = write_run(tmp_path / "a", rows, pages=[page("p1")])
    b = write_run(tmp_path / "b", rows, pages=[page("p1")], complete=False)
    report = compare_runs.compare(a, b, truth=truth(tmp_path))
    assert report["runs"]["B"]["accuracy"] == {"status": "run_incomplete_not_scored"}
    assert any("overlapped" in note for note in report["notes"])
    assert any("did not finish" in note for note in report["notes"])


def test_video_without_labels_is_reported_not_scored(tmp_path):
    rows = calls("p1", 0.0, [("image_agent_observation", 1)])
    a = write_run(tmp_path / "a", rows, pages=[page("p1")], sha="b" * 64)
    b = write_run(tmp_path / "b", calls("p1", 50.0, [("image_agent_observation", 1)]), pages=[page("p1")], sha="b" * 64)
    report = compare_runs.compare(a, b, truth=truth(tmp_path))
    assert report["runs"]["A"]["accuracy"] == {"status": "no_labels_for_this_video"}


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    a = write_run(tmp_path / "a", calls("p1", 0.0, [("image_agent_observation", 1)]), pages=[page("p1")])
    b = write_run(tmp_path / "b", calls("p1", 50.0, [("image_agent_observation", 1)]), pages=[page("p1")])
    out, md = tmp_path / "c.json", tmp_path / "c.md"
    compare_runs.main(["--a", str(a), "--b", str(b), "--label-a", "main", "--label-b", "phase1",
                       "--out", str(out), "--markdown", str(md)])
    assert json.loads(out.read_text())["labels"] == ["main", "phase1"]
    assert md.read_text().startswith("# main vs phase1")
    with pytest.raises(SystemExit):
        compare_runs.main(["--a", str(a), "--b", str(b), "--label-a", "x", "--label-b", "x", "--out", str(out)])
