"""Excluded sampling views stay visible without becoming invented missing documents."""
import json

from sever.coverage import load_video_coverage, summarize_video_coverage
from sever.report import write_report


def analysis_fixture():
    return {"source": "/private/customer/secret.mp4", "sampling": {
        "discarded_transition_samples": [3, 9],
        "discarded_transition_evidence": [
            {"sample_index": 3, "timestamp_s": 1.25, "source_pts": 125,
             "reason": "no_stable_brief_view_confirmed", "path": "/private/source.png"},
            {"sample_index": 9, "timestamp_s": 4.5,
             "reason": "partial_page_overlap_or_insufficient_visible_printed_detail"}]}}


def test_summary_preserves_moments_and_reasons_without_private_metadata():
    summary = summarize_video_coverage(analysis_fixture())
    assert summary["requires_review"] and summary["uncertain_view_count"] == 2
    assert summary["uncertain_views"] == [
        {"timestamp_s": 1.25, "reason": "no_stable_brief_view_confirmed"},
        {"timestamp_s": 4.5, "reason": "partial_page_overlap_or_insufficient_visible_printed_detail"}]
    assert "число исключений не равно числу пропущенных документов" in summary["warning"]
    assert "/private" not in json.dumps(summary) and "source_pts" not in json.dumps(summary)
    assert "complete" not in summary


def test_malformed_metadata_does_not_leak_unknown_reason_or_nonfinite_times():
    data = {"sampling": {"discarded_transition_evidence": [None, {
        "sample_index": 1, "timestamp_s": float("nan"), "reason": "/secret/<script>"}, {
        "sample_index": 1, "timestamp_s": 7, "reason": "no_stable_brief_view_confirmed"}],
        "discarded_transition_samples": [1, 2, True, "not-an-index"]},
        "pages": [None, {"segmentation": {"limitations": None}}]}
    summary = summarize_video_coverage(data)
    assert summary["uncertain_view_count"] == 2
    assert all(view["timestamp_s"] is None for view in summary["uncertain_views"])
    assert "/secret" not in json.dumps(summary) and "NaN" not in json.dumps(summary)


def test_missing_manifest_does_not_assert_complete_coverage(tmp_path):
    summary = load_video_coverage(tmp_path / "missing.json")
    assert not summary["requires_review"] and summary["warning"] is None
    assert "complete" not in summary
    assert summarize_video_coverage(None) == summary


def test_candidate_limit_and_uncertain_exclusions_remain_distinct():
    data = analysis_fixture()
    data["pages"] = [{"segmentation": {"limitations": ["page_limit_reached:321_candidates_found:300_retained"]}}]
    summary = summarize_video_coverage(data)
    assert summary["candidate_limit"] == {"found": 321, "retained": 300, "unprocessed": 21}
    assert "ещё 21 не обрабатывались" in summary["warning"]
    assert summary["uncertain_view_count"] == 2


def test_report_shows_coverage_in_partial_and_final_without_changing_values(tmp_path):
    summary = summarize_video_coverage(analysis_fixture())
    for status in ("processing", "completed"):
        result = {"status": status, "source": {"path": "input.mp4", "sha256": "hash"},
                  "elapsed_seconds": 3.4, "pages": [], "video_coverage": summary}
        before = json.dumps(result)
        write_report(tmp_path, result)
        report = (tmp_path / "report.html").read_text()
        assert 'id="video-coverage-warning"' in report
        assert "Охват видео требует проверки" in report
        assert "1.25 с" in report and "4.50 с" in report
        assert "Короткий показ не подтверждён соседними кадрами" in report
        assert "Видео обработано не полностью" not in report
        assert json.dumps(result) == before
