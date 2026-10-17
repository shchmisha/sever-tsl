from sever.aggregation import merge_documents, unresolved_fields
from sever.schema import DocumentResult, Evidence, FieldValue, FIELDS_BY_TYPE


def document(frame="a", **values):
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE["ttn"]}
    for name, value in values.items():
        fields[name] = FieldValue(raw_value=value, normalized_value=value, status="recognized",
                                 evidence=[Evidence(frame_id=frame, text=value, bbox=[.2, .2, .3, .04])])
    return DocumentResult(source_id="source", page_id="leaf", document_type="ttn", fields=fields)


def test_neighbor_recovers_missing_with_its_actual_frame_evidence():
    before = document(document_number="00012")
    after = document("b", document_number="00012", consignee='ООО "Получатель"')
    merged, report = merge_documents(before, after, "ttn")
    assert report["recovered_fields"] == ["consignee"]
    assert merged.fields["consignee"].evidence[0].frame_id == "b"
    assert {e.frame_id for e in merged.fields["document_number"].evidence} == {"a", "b"}
    assert before.fields["consignee"].status == "unreadable"


def test_disagreement_never_silently_replaces_recognized_value():
    merged, report = merge_documents(document(document_number="0012"), document("b", document_number="0013"), "ttn")
    value = merged.fields["document_number"]
    assert value.status == "conflict" and value.normalized_value is None
    assert {e.frame_id for e in value.evidence} == {"a", "b"}
    assert report["conflicting_fields"] == ["document_number"]


def test_blank_and_printed_value_disagreement_is_conflict():
    before = document()
    before.fields["document_number"] = FieldValue(status="not_present", evidence=[Evidence(frame_id="a", text="Номер:")])
    merged, _ = merge_documents(before, document("b", document_number="12"), "ttn")
    assert merged.fields["document_number"].status == "conflict"


def test_different_sources_cannot_fuse():
    import pytest
    other = document("b").model_copy(update={"source_id": "other"})
    with pytest.raises(ValueError):
        merge_documents(document(), other, "ttn")


def test_unknown_or_wrong_neighbor_cannot_change_primary_type():
    before = document(document_number="12")
    other = DocumentResult(source_id="source", page_id="leaf", document_type="unknown")
    merged, report = merge_documents(before, other, "ttn")
    assert merged == before and report["reason"] == "neighbor_reader_did_not_confirm_expected_type"


def test_needed_fields_excludes_confirmed_blank_and_known_values():
    before = document(document_number="12")
    before.fields["document_date"] = FieldValue(status="not_present")
    assert unresolved_fields(before, "ttn") == ["consignor", "consignee"]


def test_third_reading_cannot_erase_a_source_disagreement():
    disputed, _ = merge_documents(document(document_number="12"), document("b", document_number="13"), "ttn")
    merged, _ = merge_documents(disputed, document("c", document_number="12"), "ttn")
    assert merged.fields["document_number"].status == "conflict"
    assert merged.fields["document_number"].normalized_value is None
    assert {e.frame_id for e in merged.fields["document_number"].evidence} == {"a", "b", "c"}


def test_two_views_required_per_field_not_number_of_region_boxes():
    from sever.aggregation import fields_needing_views
    value = document(document_number="12")
    value.fields["document_number"].evidence.append(Evidence(frame_id="a", text="12", bbox=[.3, .2, .2, .04]))
    assert "document_number" in fields_needing_views(value, "ttn")
    agreed, _ = merge_documents(value, document("b", document_number="12"), "ttn")
    assert "document_number" not in fields_needing_views(agreed, "ttn")


def test_null_normalized_values_cannot_corroborate_different_raw_values():
    first, second = document(document_number="12"), document("b",document_number="13")
    first.fields["document_number"].normalized_value = None
    second.fields["document_number"].normalized_value = None
    result, _ = merge_documents(first, second, "ttn")
    assert result.fields["document_number"].status == "conflict"


def test_grounded_neighbor_conflict_is_not_discarded():
    second = document("b")
    second.fields["document_number"] = FieldValue(status="conflict",evidence=[Evidence(frame_id="b",text="12 / 13")])
    result, _ = merge_documents(document(document_number="12"),second,"ttn")
    assert result.fields["document_number"].status == "conflict"
    assert {e.frame_id for e in result.fields["document_number"].evidence} == {"a","b"}


def test_different_own_document_numbers_block_fusion_before_registration(monkeypatch):
    import sever.aggregation as aggregation
    monkeypatch.setattr(aggregation, "_identity", lambda ocr, kind: {"document_number": ocr["number"]})
    decision = aggregation.neighbor_eligibility({"frame_id": "a", "number": "12"}, {"frame_id": "b", "number": "13"}, "ttn", ["consignee"])
    assert not decision["eligible"] and decision["reason"] == "different_own_header_values"


def test_reference_to_other_form_is_not_own_heading():
    from sever.aggregation import neighbor_eligibility
    primary = {"frame_id": "a", "regions": []}
    neighbor = {"frame_id": "b", "regions": [{"text": "Приложение к транспортной накладной", "bbox": [.1, .1, .7, .04]}]}
    decision = neighbor_eligibility(primary, neighbor, "trn", ["consignee"])
    assert not decision["eligible"] and decision["reason"] == "neighbor_own_form_heading_not_confirmed"


def test_partial_address_with_continuation_marker_cannot_erase_full_source_value():
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignor_address"] = FieldValue(status="recognized", raw_value="454004, Street, dom № 158, apartment 59", normalized_value="454004, Street, dom № 158, apartment 59", evidence=[Evidence(frame_id="a",text="address")])
    before = DocumentResult(source_id="source",page_id="leaf",document_type="upd",fields=fields)
    second = before.model_copy(deep=True)
    second.fields["consignor_address"] = FieldValue(status="recognized",raw_value="454004, Street, dom No 158,",normalized_value="454004, Street, dom No 158,",evidence=[Evidence(frame_id="b",text="partial address")])
    merged, report = merge_documents(before, second, "upd")
    assert merged.fields["consignor_address"] == before.fields["consignor_address"]
    assert report["partial_observations"][0]["field"] == "consignor_address"
    # The partial view supplies no evidence for the apartment number.
    assert {e.frame_id for e in merged.fields["consignor_address"].evidence} == {"a"}
    second.fields["consignor_address"] = FieldValue(status="recognized",raw_value="454004, Street, dom № 158, apartment 60",normalized_value="454004, Street, dom № 158, apartment 60",evidence=[Evidence(frame_id="c",text="different unit")])
    merged, report = merge_documents(before, second, "upd")
    assert merged.fields["consignor_address"].status == "conflict"
