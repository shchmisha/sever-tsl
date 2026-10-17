"""Saved-file report guards: no provider or real corpus writes."""
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("delivery_builder", Path(__file__).parents[1] / "tools/build_delivery.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_execution_counts_new_calls_separately_from_cache_and_unknown_billing():
    response = {"event": "model_response_received", "time": 2, "source_id": "s", "page_id": "p",
                "input_tokens": 12, "output_tokens": 3,
                "billing": {"kind": "provider_reported", "currency": "RUB", "value": .4}}
    rows = [{"event": "model_response_received", "time": 0, "input_tokens": 999},
            {"event": "run_started", "time": 1}, {"event": "vision_cache_revalidated"},
            {"event": "model_request_started"}, response, response.copy(),
            {"event": "model_request_started"}, {"event": "run_completed", "time": 4, "elapsed_seconds": 3}]
    result = builder.execution_snapshot(rows)
    assert result["mode"] == "mixed_cache_and_paid"
    assert result["attempted_requests"] == 2 and result["recorded_responses"] == 1
    assert result["actual_cost_rub"] == .4 and result["input_tokens"] == 12
    assert result["attempts_without_recorded_response"] == 1
    assert result["cold_runtime_seconds"] is None and result["recorded_elapsed_seconds"] == 3


def fixture_comparison(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text('{"reference":"independent"}')
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text('{"old":true}'); new.write_text('{"new":true}')
    ratio = {"correct": 1, "total": 1}
    counts = {key: ratio.copy() for key in ("candidate_coverage", "type_accuracy_targets", "unknown_rejection", "e2e_on_fully_labeled", "labeled_fields")}
    metrics = {key: value.copy() for key, value in counts.items() if key != "labeled_fields"}
    metrics["field_accuracy"] = {"upd.invoice_number": ratio.copy()}
    entry = {"status": "complete_provisional", "source": {"sha256": "source-hash"}, "run_id": "new",
             "truth": {"labeled_fields": 1, "fully_labeled_target_intervals": 1, "leaf_intervals": 1},
             "artifacts": {"results.json": {"path": "new.json"}}, "metrics": metrics}
    value = {"truth_sha256": builder.file_hash(truth), "baseline": counts, "final": counts,
             "by_source": [{"source": entry["source"], "result_inputs": {
                 "baseline": {"path": "old.json", "sha256": builder.file_hash(old)},
                 "final": {"path": "new.json", "sha256": builder.file_hash(new)}}}]}
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(value))
    return truth, old, new, entry, value, path


def test_comparison_uses_generic_denominators_and_rejects_changed_result(tmp_path):
    truth, old, new, entry, value, path = fixture_comparison(tmp_path)
    result = builder.comparison_snapshot(tmp_path, path, truth, [entry])
    assert result["status"] == "verified_provisional"
    assert result["expected_labeled_fields"] == 1 and result["expected_fully_labeled_targets"] == 1
    new.write_text('{"new":"changed"}')
    stale = builder.comparison_snapshot(tmp_path, path, truth, [entry])
    assert stale["metrics"] is None
    assert "final results SHA is missing or stale" in stale["pending_reasons"]


def test_comparison_with_changed_truth_or_pending_run_never_shows_metrics(tmp_path):
    truth, old, new, entry, value, path = fixture_comparison(tmp_path)
    truth.write_text('{"reference":"revised"}')
    entry["status"] = "pending"
    result = builder.comparison_snapshot(tmp_path, path, truth, [entry])
    assert result["metrics"] is None
    assert len(result["pending_reasons"]) == 2


def test_recovered_export_is_not_presented_as_cold_runtime():
    result = builder.execution_snapshot([{"event": "run_started", "time": 1},
        {"event": "vision_cache_revalidated"}, {"event": "exports_recovered", "time": 9,
        "source_complete_proven": True, "replay_provider_calls": 0, "parent_elapsed_seconds": 100}], recovery_valid=True)
    assert result["status"] == "complete_recovered"
    assert result["recorded_elapsed_seconds"] is None
    assert result["cold_runtime_seconds"] is None
    assert result["recovery"]["parent_elapsed_seconds"] == 100


def test_recovery_needs_current_result_sha_page_count_and_existing_provenance(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (tmp_path / "audit.json").write_text('{"source_complete_proven":true}')
    result = {"source": {"sha256": "source"}, "pages": [{"result": {"document_type": "unknown"}}]}
    (folder / "results.json").write_text(json.dumps(result))
    (folder / "metrics.json").write_text(json.dumps({"type_accuracy_all_expected": {"correct": 1, "total": 1}, "field_accuracy": {}}))
    truth = {"source_sha256": "source", "pages": [{"document_type": "unknown", "fields": {}}]}
    recovery = {"event": "exports_recovered", "time": 2, "source_complete_proven": True,
                "results_sha256": builder.file_hash(folder / "results.json"), "page_count": 1,
                "recovery_provenance_path": "audit.json", "replay_provider_calls": 0}
    def trace():
        (folder / "trace.jsonl").write_text(json.dumps({"event": "run_started", "time": 1}) + '\n' + json.dumps(recovery) + '\n')
    trace()
    assert builder.load_run(tmp_path, folder, truth, "video")["status"] == "complete_provisional"
    recovery["results_sha256"] = "obsolete"
    trace()
    invalid = builder.load_run(tmp_path, folder, truth, "video")
    assert invalid["status"] == "pending" and invalid["metrics"] is None
    assert any("matching result SHA" in reason for reason in invalid["pending_reasons"])
