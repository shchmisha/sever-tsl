import pytest

from sever.evaluation import evaluate
from sever.normalize import comparison, normalize


def _ttn_fields():
    return {"document_number": "0001", "document_date": "01.08.2026", "consignor": "ООО А", "consignee": "ООО Б"}


def _prediction(kind="unknown", fields=None, timestamp=1, errors=None, **page_values):
    return {
        "page_id": "prediction", "selected_frame_id": "primary",
        "frames": [{"frame_id": "primary", "timestamp_s": timestamp}],
        "result": {
            "document_type": kind, "errors": errors or [],
            "fields": {name: {"status": "recognized" if value is not None else "not_present", "normalized_value": value}
                       for name, value in (fields or {}).items()},
        }, **page_values,
    }

def test_misses_count_against_type_fields_and_e2e():
    truth={"pages":[{"start_s":0,"end_s":2,"document_type":"ttn","fields":{"document_number":"0001","document_date":"01.08.2026","consignor":"ООО А","consignee":"ООО Б"}}]}
    report=evaluate({"pages":[]},truth)
    assert report["e2e"]["rate"] == 0
    assert report["field_accuracy"]["ttn.document_number"]["accuracy"] == 0

def test_one_prediction_cannot_cover_two_expected_leaves():
    truth={"pages":[{"start_s":0,"end_s":2,"document_type":"unknown"},{"start_s":2,"end_s":4,"document_type":"unknown"}]}
    actual={"pages":[{"page_id":"p1","frames":[{"timestamp_s":1},{"timestamp_s":3}],"result":{"document_type":"unknown"}}]}
    assert evaluate(actual,truth)["detection"]["correct"] == 1

def test_unlabeled_field_does_not_count_as_success():
    truth={"pages":[{"start_s":0,"end_s":2,"document_type":"ttn","fields":{}}]}
    report = evaluate({"pages":[{"frames":[{"timestamp_s":1}],"result":{"document_type":"ttn"}}]},truth)
    assert report["e2e"]["correct"] == 0
    assert report["e2e_on_fully_labeled"]["total"] == 0
    assert report["e2e_on_fully_labeled"]["rate"] is None
    assert report["details"][0]["e2e_success"] is None
    assert report["details"][0]["e2e_evaluated"] is False

def test_dates_reject_impossible_and_numbers_keep_zeros():
    assert normalize("document_date","31.02.2026") is None
    assert normalize("invoice_date","03.06.2026") == "2026-06-03"
    assert normalize("document_number","№ 0001/26") == "0001/26"


@pytest.mark.parametrize("error", [
    "ocr_failed", "extraction_failed:RuntimeError",
    "classification: no OCR evidence; downgraded to unknown",
    "classification: invalid evidence; downgraded to unknown",
    "classification: foreign or invalid OCR evidence ids rejected",
])
def test_failed_unknown_is_not_a_correct_negative_control(error):
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "unknown"}]}
    report = evaluate({"pages": [_prediction(errors=[error])]}, truth)
    assert report["detection"]["correct"] == 1  # Candidate coverage is separate from extraction.
    assert report["type_accuracy_all_expected"]["correct"] == 0
    assert report["unknown_rejection"]["correct"] == 0
    assert report["processing_failures"]["candidate_leaves"] == 1
    assert report["details"][0]["processing_failed"] is True


def test_out_of_scope_rejection_with_explanatory_warning_still_counts():
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "unknown"}]}
    report = evaluate({"pages": [_prediction(errors=["classification: CMR or Belarus TN-2 is outside the contract"])]}, truth)
    assert report["unknown_rejection"]["correct"] == 1
    assert report["processing_failures"]["candidate_leaves"] == 0


@pytest.mark.parametrize("error", [
    "native_image: quality_gate_rejected",
    "native_image:quality_gate_rejected:all_source_frames_unusable",
    "native_image: image_reading_failed",
    "native_image: image_reading_failed:frame-1:TimeoutError",
    "native_image: image_reading_failed:extra-pts-1234:AnalysisError",
    "native_image: model_budget_or_provider_failure",
    "native_image: model_budget_or_provider_failure:BatchBudgetExceeded",
    "native_image: model_budget_or_provider_failure:FatalProviderError",
    "native_image: additional_frame_pool_failed",
    "native_image: additional_frame_pool_failed:OSError",
])
def test_native_operational_failure_cannot_earn_unknown_rejection_credit(error):
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "unknown"}]}
    report = evaluate({"pages": [_prediction(errors=[error])]}, truth)
    assert report["candidate_coverage"]["correct"] == report["candidate_coverage"]["total"] == 1
    assert report["candidate_coverage"]["rate"] == 1.
    assert report["unknown_rejection"] == {"correct": 0, "total": 1, "rate": 0.}
    assert report["type_accuracy_all_expected"] == {"correct": 0, "total": 1, "rate": 0.}
    assert report["processing_failures"] == {"candidate_leaves": 1, "reasons": {error: 1}}
    assert report["failure_reasons"]["processing_failed"] == 1
    assert report["details"][0]["detected"] is True
    assert report["details"][0]["processing_failed"] is True
    assert report["details"][0]["type_correct"] is False


@pytest.mark.parametrize("warning", [
    "native_image: own heading unsupported: CMR",
    "Own form lacks a quoted supported heading in actual supplied panels",
    "image_fusion:conflicting_document_types",
    "native_image: form_validation: image_reading_failed appears in printed text",
])
def test_native_semantic_unknown_remains_eligible_without_operational_failure(warning):
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "unknown"}]}
    report = evaluate({"pages": [_prediction(errors=[warning])]}, truth)
    assert report["unknown_rejection"] == {"correct": 1, "total": 1, "rate": 1.}
    assert report["processing_failures"] == {"candidate_leaves": 0, "reasons": {}}
    assert report["details"][0]["processing_failed"] is False
    assert report["failure_reasons"][warning] == 1


def test_native_failures_and_misses_preserve_all_field_and_target_denominators():
    fields = _ttn_fields()
    failure = "native_image: image_reading_failed:primary:TimeoutError"
    truth = {"pages": [
        {"start_s": 0, "end_s": 2, "document_type": "unknown"},
        {"start_s": 2, "end_s": 4, "document_type": "unknown"},
        {"start_s": 4, "end_s": 6, "document_type": "ttn", "fields": fields},
        {"start_s": 6, "end_s": 8, "document_type": "ttn", "fields": fields},
    ]}
    actual = {"pages": [
        _prediction(errors=[failure]),
        _prediction(timestamp=3, errors=["native_image: own heading unsupported: CMR"]),
        _prediction("ttn", fields, timestamp=5, errors=[failure]),
    ]}
    report = evaluate(actual, truth)
    assert report["candidate_coverage"]["correct"] == 3 and report["candidate_coverage"]["total"] == 4
    assert report["unknown_rejection"] == {"correct": 1, "total": 2, "rate": .5}
    assert report["type_accuracy_all_expected"] == {"correct": 1, "total": 4, "rate": .25}
    assert report["type_accuracy_targets"] == {"correct": 0, "total": 2, "rate": 0.}
    assert all(value == {"correct": 0, "total": 2, "accuracy": 0.}
               for value in report["field_accuracy"].values())
    assert set(report["field_accuracy"]) == {f"ttn.{name}" for name in fields}
    assert report["e2e_on_fully_labeled"]["correct"] == 0 and report["e2e_on_fully_labeled"]["total"] == 2
    assert report["e2e_all_targets_conservative"]["correct"] == 0 and report["e2e_all_targets_conservative"]["total"] == 2
    assert report["processing_failures"] == {"candidate_leaves": 2, "reasons": {failure: 2}}
    assert report["failure_reasons"]["processing_failed"] == 2 and report["failure_reasons"]["missed_leaf"] == 1
    assert report["details"][2]["field_checks"] == {name: False for name in fields}
    assert report["details"][3]["detected"] is False


def test_matching_uses_primary_anchor_even_when_neighbor_crosses_boundary():
    truth = {"pages": [
        {"page_id": "first", "start_s": 0, "end_s": 2, "document_type": "unknown"},
        {"page_id": "second", "start_s": 2, "end_s": 4, "document_type": "unknown"},
    ]}
    second = _prediction(timestamp=3, page_id="second-prediction")
    second["frames"].insert(0, {"frame_id": "neighbor", "timestamp_s": 1})
    first = _prediction(timestamp=1, page_id="first-prediction")
    report = evaluate({"pages": [second, first]}, truth)
    assert report["detection"]["correct"] == 2
    assert report["unmatched_predictions"] == 0
    assert [detail["prediction_page_id"] for detail in report["details"]] == ["first-prediction", "second-prediction"]
    assert [detail["primary_timestamp_s"] for detail in report["details"]] == [1, 3]


def test_missing_declared_primary_does_not_match_through_neighbor():
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "unknown"}]}
    page = _prediction(selected_frame_id="absent")
    report = evaluate({"pages": [page]}, truth)
    assert report["detection"]["correct"] == 0
    assert report["unmatched_predictions"] == 1
    assert report["selected_frame_quality"]["missing_primary_frame"] == 1


def test_target_accuracy_and_e2e_annotation_coverage_are_separate():
    fields = _ttn_fields()
    truth = {"pages": [
        {"start_s": 0, "end_s": 2, "document_type": "ttn", "fields": fields},
        {"start_s": 2, "end_s": 4, "document_type": "ttn", "fields": {}},
        {"start_s": 4, "end_s": 6, "document_type": "unknown"},
    ]}
    report = evaluate({"pages": [_prediction("ttn", fields), _prediction("ttn", timestamp=3), _prediction(timestamp=5)]}, truth)
    assert report["type_accuracy_targets"]["correct"] == 2
    assert report["type_accuracy_targets"]["total"] == 2
    assert report["e2e_on_fully_labeled"]["rate"] == 1
    assert report["e2e_all_targets_conservative"]["rate"] == .5
    assert report["annotation_coverage"] == {"fully_labeled_targets": 1, "target_total": 2, "unlabeled_targets": 1}
    assert report["details"][1]["e2e_success"] is None


def test_negative_controls_do_not_inflate_target_accuracy():
    truth = {"pages": [
        {"start_s": 0, "end_s": 2, "document_type": "ttn", "fields": _ttn_fields()},
        {"start_s": 2, "end_s": 4, "document_type": "unknown"},
    ]}
    report = evaluate({"pages": [_prediction(), _prediction(timestamp=3)]}, truth)
    assert report["type_accuracy_all_expected"]["rate"] == .5
    assert report["type_accuracy_targets"]["rate"] == 0
    assert report["e2e_on_fully_labeled"]["rate"] == 0
    assert all(count["total"] == 1 and count["accuracy"] == 0 for count in report["field_accuracy"].values())


def test_explicitly_unlabeled_field_is_excluded_but_makes_e2e_unmeasured():
    fields = _ttn_fields()
    fields["consignee"] = {"value": "ООО Б", "labeled": False}
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "ttn", "fields": fields}]}
    report = evaluate({"pages": [_prediction("ttn", _ttn_fields())]}, truth)
    assert "ttn.consignee" not in report["field_accuracy"]
    assert report["e2e_on_fully_labeled"]["total"] == 0
    assert report["e2e_all_targets_conservative"]["unlabeled_targets"] == 1


def test_accepted_number_script_variant_is_targeted_to_annotated_field():
    fields = _ttn_fields()
    fields["document_number"] = {"value": "СТ-98616", "status": "recognized", "accepted_variants": ["CT-98616"]}
    truth = {"pages": [{"start_s": 0, "end_s": 2, "document_type": "ttn", "fields": fields}]}
    values = {**_ttn_fields(), "document_number": "CT-98616"}
    report = evaluate({"pages": [_prediction("ttn", values)]}, truth)
    assert report["field_accuracy"]["ttn.document_number"]["accuracy"] == 1
    assert report["e2e_on_fully_labeled"]["rate"] == 1
    fields["document_number"].pop("accepted_variants")
    assert evaluate({"pages": [_prediction("ttn", values)]}, truth)["field_accuracy"]["ttn.document_number"]["accuracy"] == 0
    assert comparison("document_number", "CT-98616") != comparison("document_number", "СТ-98616")


@pytest.mark.parametrize("field,left,right", [
    ("consignor", 'Общество с ограниченной ответственностью "Диета +"', 'ООО «Диета+»'),
    ("consignee", 'Акционерное общество "Тандер"', 'АО Тандер'),
    ("customer", 'ООО "УМНЫЙ РИТЕЙЛ"', 'ООО УМНЫЙ РИТЕЙЛ'),
    ("consignor_address", '192236, Санкт-Петербург, ул.Софийская, д.6, к.2, литер А', '192236 Санкт-Петербург ул Софийская д 6 к 2 литер А'),
    ("consignee_address", '628422, г. Сургут, Индустриальная ул., д. 21, соор. 16', '628422 г Сургут Индустриальная ул д 21 соор 16'),
    ("consignor_address", 'дом Nº 158, квартира 59', 'дом № 158 квартира 59'),
])
def test_formatting_equivalence_keeps_names_and_address_tokens(field, left, right):
    assert comparison(field, left) == comparison(field, right)


@pytest.mark.parametrize("field,left,right", [
    ("consignor", "ООО Диета +", "ООО Диета"),
    ("consignor", "ООО Север", "АО Север"),
    ("consignee", "ООО Ритейл 1", "ООО Ритейл 2"),
    ("consignee_address", "Сургут Индустриальная ул д21 соор16", "Сургут Индустриальная ул д21 соор17"),
    ("consignor_address", "ул Софийская д6 к2", "ул Гороховая д6 к2"),
    ("consignee_address", "д.16.1", "д.161"),
    ("document_number", "0001", "001"),
])
def test_normalization_does_not_accept_meaningful_differences(field, left, right):
    assert comparison(field, left) != comparison(field, right)


def test_selected_quality_summary_uses_primary_flags_and_keeps_unknowns():
    page = _prediction()
    page["frames"][0]["quality"] = {
        "limitations": ["single_sample_candidate_stability_unconfirmed"],
        "full_page_visible": False, "exposure": {"glare_confirmed": None},
        "possible_hand_occlusion": {"hand_occlusion_confirmed": None},
    }
    page["frames"].append({"frame_id": "neighbor", "timestamp_s": 1.2, "quality": {"limitations": ["neighbor_only"]}})
    report = evaluate({"pages": [page, _prediction(timestamp=3)]}, {"pages": []})
    quality = report["selected_frame_quality"]
    assert quality["candidate_leaves"] == 2
    assert quality["frames_with_quality"] == 1
    assert quality["missing_quality"] == 1
    assert quality["flags"] == {"single_sample_candidate_stability_unconfirmed": 1}
    assert quality["full_page_visible"] == {"false": 1}
    assert quality["glare_confirmed"] == {"unknown": 1}
    assert quality["hand_occlusion_confirmed"] == {"unknown": 1}


def test_candidate_coverage_does_not_invent_usable_frame_detection():
    truth = {'pages': [{'page_id':'truth','start_s':0,'end_s':2,'document_type':'unknown'}]}
    report = evaluate({'pages':[_prediction()]},truth)
    assert report['candidate_coverage']['rate'] == 1
    assert report['frame_document_detection_rate']['rate'] is None
    truth['selected_frame_usability']={'truth':{'selected_frame_id':'primary','usable':False,'independently_reviewed':True}}
    report = evaluate({'pages':[_prediction()]},truth)
    assert report['frame_document_detection_rate']['rate'] == 0
    truth['selected_frame_usability']['truth']['usable']=True
    assert evaluate({'pages':[_prediction()]},truth)['frame_document_detection_rate']['rate'] == 1
    truth['selected_frame_usability']['truth']['selected_frame_id']='different-source-frame'
    assert evaluate({'pages':[_prediction()]},truth)['frame_document_detection_rate']['rate'] is None


def test_ground_truth_source_hash_is_required_when_declared():
    with pytest.raises(ValueError, match="source hash"):
        evaluate({'source':{'sha256':'different'},'pages':[]},{'source_sha256':'expected','pages':[]})


def test_readable_non_target_headings_do_not_inflate_target_field_usability():
    truth = {"pages": [
        {"page_id": "other", "start_s": 0, "end_s": 2, "document_type": "unknown"},
        {"page_id": "target", "start_s": 2, "end_s": 4, "document_type": "ttn"},
    ], "selected_frame_usability": {
        "other": {"selected_frame_id": "primary", "usable": True, "independently_reviewed": True},
        "target": {"selected_frame_id": "primary", "usable": False, "independently_reviewed": True},
    }}
    report = evaluate({"pages": [_prediction(timestamp=1), _prediction(kind="ttn", timestamp=3)]}, truth)
    assert report["frame_document_detection_rate"]["rate"] == .5
    assert report["frame_document_detection_rate_targets"]["total"] == 1
    assert report["frame_document_detection_rate_targets"]["rate"] == 0


def test_form_rejection_only_review_cannot_establish_target_field_readability():
    truth = {"pages": [{"page_id": "target", "start_s": 0, "end_s": 2, "document_type": "ttn"}],
             "selected_frame_usability": {"target": {"selected_frame_id": "primary", "usable": True,
                "independently_reviewed": True, "assessment": "form_rejection"}}}
    report = evaluate({"pages": [_prediction()]}, truth)
    assert report["frame_document_detection_rate_targets"]["rate"] == 0
    truth["selected_frame_usability"]["target"]["usable"] = None
    metric = evaluate({"pages": [_prediction()]}, truth)["frame_document_detection_rate"]
    assert metric["uncertain"] == 1 and metric["confirmed_unusable"] == 0
