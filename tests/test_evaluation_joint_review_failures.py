"""Synthetic in-memory evaluation cases; no labels, fixtures or source files."""
import pytest

from sever.evaluation import evaluate, _processing_failures


REVIEW_FAILURES = [
    "conflict_review_failed",
    "conflict_review_budget_or_provider_failure",
    "type_conflict_review_failed",
    "type_conflict_review_budget_or_provider_failure",
]


def unknown_case(errors):
    result = {"pages": [{"page_id": "synthetic-prediction", "selected_frame_id": "synthetic-frame",
        "frames": [{"frame_id": "synthetic-frame", "timestamp_s": 1}],
        "result": {"document_type": "unknown", "fields": {}, "errors": errors}}]}
    expected = {"pages": [{"page_id": "synthetic-interval", "start_s": 0, "end_s": 2,
                           "document_type": "unknown", "fields": {}}]}
    return evaluate(result, expected)


@pytest.mark.parametrize("code", REVIEW_FAILURES)
@pytest.mark.parametrize("suffix", ["", ":AnalysisError", ":frame-id:OSError"])
def test_joint_review_operational_failure_cannot_earn_unknown_rejection_credit(code, suffix):
    error = "native_image: " + code + suffix
    metrics = unknown_case([error])
    assert metrics["candidate_coverage"]["correct"] == 1
    assert metrics["unknown_rejection"] == {"correct": 0, "total": 1, "rate": 0}
    assert metrics["type_accuracy_all_expected"]["correct"] == 0
    assert metrics["details"][0]["processing_failed"] is True
    assert metrics["details"][0]["type_correct"] is False
    assert metrics["failure_reasons"]["processing_failed"] == 1
    assert metrics["processing_failures"] == {"candidate_leaves": 1, "reasons": {error: 1}}


@pytest.mark.parametrize("errors", [[], ["image_fusion:conflicting_document_types"],
    ["native_image: uncertain_heading"], ["native_image: out_of_scope_form"]])
def test_semantic_unknown_and_out_of_scope_remain_distinct_from_operational_failure(errors):
    metrics = unknown_case(errors)
    assert metrics["unknown_rejection"] == {"correct": 1, "total": 1, "rate": 1}
    assert metrics["details"][0]["processing_failed"] is False
    assert metrics["processing_failures"] == {"candidate_leaves": 0, "reasons": {}}


def test_failure_matching_uses_exact_namespace_and_code_not_transcription_substrings():
    nonfailures = ["model_text:native_image: conflict_review_failed",
        "image_fusion:type_conflict_review_failed", "native_image: conflict_review_failed_semantic",
        "native_image: type_conflict_review_budget_or_provider_failure_semantic"]
    assert _processing_failures({"errors": nonfailures}) == []


def test_multiple_review_failures_remain_visible_without_multiplying_failed_leaf_count():
    errors = ["native_image: " + code for code in REVIEW_FAILURES]
    metrics = unknown_case(errors)
    assert metrics["processing_failures"] == {"candidate_leaves": 1,
        "reasons": {error: 1 for error in errors}}
    assert metrics["failure_reasons"]["processing_failed"] == 1
    assert metrics["unknown_rejection"]["correct"] == 0
