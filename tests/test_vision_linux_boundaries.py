"""Linux line segmentation regressions; no paid calls or expected source values."""
from copy import deepcopy

import pytest

from sever.aggregation import neighbor_eligibility
from sever.schema import FIELDS_BY_TYPE
from sever.vision import (VisionDecision, VisionField, _hint_bundle, _is_company, _role_sources,
                          _seller_address_literal, _validate_with_ocr, _wrapped_upd_titles)


def region(text, x, y, w=.35, h=.012, source_pass="full_image"):
    return {"text": text, "bbox": [x, y, w, h], "processing_pass": source_pass}


def seller_lookup(caption="ИННКПП продавца"):
    rows = [region("Адрес:", .05, .10, .07),
            region("123456, Город, Улица, дом 8,", .20, .10),
            region("квартира 18", .20, .12),
            region(caption, .05, .145, .13),
            region("1234567890/123456789", .20, .15)]
    return {f"r{i}": row for i, row in enumerate(rows)}


@pytest.mark.parametrize("caption", ["ИННКПП продавца", "ИНН/КПП продавца", "КПП",
                                      "Грузоотправитель и его адрес", "Грузополучатель и его адрес"])
def test_complete_requisite_and_party_captions_bound_literal_address(caption):
    lookup = seller_lookup(caption)
    raw, ids = _seller_address_literal({"ids": list(lookup)}, lookup)
    assert raw == "123456, Город, Улица, дом 8, квартира 18"
    assert ids == ["r0", "r1", "r2"]


def test_unreadable_requisite_caption_cannot_turn_tax_digits_into_address():
    lookup = seller_lookup("VYHH/KNN продавца")
    assert _seller_address_literal({"ids": list(lookup)}, lookup) is None


def reference_page():
    rows = [region("Продавец:", .05, .06, .10),
            region('ООО "Первая сторона"', .20, .06),
            region("Адрес:", .05, .09, .07),
            region("123456, Город, Улица, дом 8,", .20, .09),
            region("квартира 1В", .20, .11),
            region("ИННКПП продавца", .05, .14, .13),
            region("1234567890/123456789", .20, .14),
            region("Грузоотправитель и его адрес", .05, .18, .13),
            region("он же", .20, .18, .05),
            region('ООО "Вторая сторона"', .20, .21),
            region("Грузополучатель и его адрес", .05, .23, .13),
            region("654321, Другой город, дом 9", .20, .23)]
    page = {"frame_id": "actual-frame", "regions": rows}
    crop = {"frame_id": page["frame_id"], "crop_bbox_normalized_top_left": [0, 0, 1, .35],
            "source_full_page_visible": True, "role_panels": _role_sources(page, "upd")}
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignor_address"] = VisionField(raw_value="123456, Город, Улица, дом 8, квартира 18",
        status="recognized", panel_id="consignor", evidence_ids=["r7", "r3", "r4"],
        evidence_text="Грузоотправитель: он же. Адрес: 123456, Город, Улица, дом 8, квартира 18", confidence=.9)
    decision = VisionDecision(document_type="upd", heading_text="Универсальный передаточный документ", fields=fields)
    return page, crop, decision


def test_single_contradictory_ocr_suffix_does_not_replace_validated_pixel_address():
    page, crop, decision = reference_page()
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "recognized"
    assert field.raw_value == decision.fields["consignor_address"].raw_value
    assert field.confidence == .9
    assert any(e.text == "он же" and e.region_index == 8 for e in field.evidence)
    assert "1234567890" not in field.raw_value


def test_partial_ocr_prefix_does_not_replace_readable_pixel_address_suffix():
    page, crop, decision = reference_page()
    page["regions"][4]["text"] = ""
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "recognized" and field.raw_value.endswith("квартира 18")
    assert field.confidence == .9


def test_two_independent_literal_seller_address_disagreements_remain_conflict():
    page, crop, decision = reference_page()
    first_pass = deepcopy(page["regions"][:7])
    for row in first_pass:
        row["processing_pass"] = "header"
    page["regions"].extend(first_pass)
    crop["role_panels"] = _role_sources(page, "upd")
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "conflict" and field.raw_value is None
    assert "repeated literal OCR" in field.reason
    assert {4, 16} <= {e.region_index for e in field.evidence}


def test_two_partial_literal_prefixes_do_not_contradict_unseen_pixel_suffix():
    page, crop, decision = reference_page()
    page["regions"][4]["text"] = ""
    first_pass = deepcopy(page["regions"][:7])
    for row in first_pass:
        row["processing_pass"] = "header"
    page["regions"].extend(first_pass)
    crop["role_panels"] = _role_sources(page, "upd")
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "recognized" and field.raw_value.endswith("квартира 18")
    assert field.confidence == .9


def test_lone_corrupted_seller_name_does_not_replace_readable_pixel_name():
    page, crop, decision = reference_page()
    page["regions"][1]["text"] = 'ООО "Первая стор0на"'
    decision.fields["consignor"] = VisionField(raw_value='ООО "Первая сторона"', status="recognized", panel_id="consignor",
        confidence=.9, evidence_ids=["r7", "r8", "r1"], evidence_text='Грузоотправитель: он же. Продавец ООО "Первая сторона"')
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor"]
    assert field.status == "recognized" and field.raw_value == decision.fields["consignor"].raw_value
    assert field.confidence == .9
    assert any(e.text == "он же" for e in field.evidence)


def test_repeated_literal_seller_name_disagreement_remains_conflict():
    page, crop, decision = reference_page()
    page["regions"][1]["text"] = 'ООО "Первая стор0на"'
    page["regions"].append({**deepcopy(page["regions"][1]), "processing_pass": "header"})
    crop["role_panels"] = _role_sources(page, "upd")
    decision.fields["consignor"] = VisionField(raw_value='ООО "Первая сторона"', status="recognized", panel_id="consignor",
        confidence=.9, evidence_ids=["r7", "r8", "r1"], evidence_text='Грузоотправитель: он же. Продавец ООО "Первая сторона"')
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor"]
    assert field.status == "conflict" and field.raw_value is None
    assert "repeated literal OCR" in field.reason


def test_wrong_receiver_address_still_resolves_only_to_literal_seller_reference():
    page, crop, decision = reference_page()
    decision.fields["consignor_address"].raw_value = page["regions"][11]["text"]
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "recognized"
    assert field.raw_value == "123456, Город, Улица, дом 8, квартира 1В"
    assert "654321" not in field.raw_value
    assert field.confidence is None


def test_leading_punctuation_keeps_literal_company_inside_its_own_role_band():
    assert _is_company('`ООО "Получатель"; ИННКПП 1234567890/123456789')
    assert not _is_company('г. Город, офис ООО "Получатель"')
    page = {"frame_id": "f", "regions": [
        region('`ООО "Получатель"; ИННКПП 1234567890/123456789', .30, .26),
        region("Грузополучатель и его адрес", .05, .28, .20),
        region("123456, Город, улица, дом 1", .30, .28)]}
    panels = _role_sources(page, "upd")
    assert "r0" in panels["consignee"]["ids"]
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignee"] = VisionField(status="recognized", raw_value='ООО "Получатель"',
        panel_id="consignee", evidence_ids=["r1"], evidence_text='Грузополучатель ООО "Получатель"')
    decision = VisionDecision(document_type="upd", heading_text="Универсальный передаточный документ", fields=fields)
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .35],
            "source_full_page_visible": True, "role_panels": panels}
    field = _validate_with_ocr(decision, crop, "upd", "s", "p", "test", page).fields["consignee"]
    assert field.status == "recognized" and field.raw_value == 'ООО "Получатель"'


def test_fragmented_early_pass_cannot_exhaust_hints_before_actual_later_party_name():
    page = {"frame_id": "f", "regions": [region("Грузополучатель", .05, .20, .20)]}
    page["regions"] += [region(f"фрагмент {i}", .35, .20, .05, source_pass="full_image") for i in range(160)]
    page["regions"].append(region('`ООО"Получатель"', .35, .19, source_pass="left_roles"))
    panels = {"consignee": {"panel_id": "consignee", "bbox": [0, .18, .8, .06],
                            "caption_id": "r0", "ids": [f"r{i}" for i in range(len(page["regions"]))]}}
    bundle, _ = _hint_bundle(page, "upd", panels)
    retained = bundle["fields"]["consignee"]["allowed_ocr_ids"]
    assert "r0" in retained and "r161" in retained
    assert len(retained) < len(page["regions"])


def wrapped_page():
    return {"frame_id": "wrapped-frame", "regions": [
        region("Универсальный", .26, .145, .037, .009),
        region("передаточный", .26, .158, .034, .008),
        region("документ", .26, .169, .022, .007),
        region("Счет-фактура № 00017", .32, .15, .16, .01)]}


def test_wrapped_upd_title_preserves_actual_three_region_evidence():
    page = wrapped_page()
    title, = _wrapped_upd_titles(page)
    assert title["text"] == "Универсальный передаточный документ"
    assert title["source_region_indices"] == [0, 1, 2]
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    decision = VisionDecision(document_type="upd", heading_text="Счет-фактура № 00017", fields=fields)
    crop = {"frame_id": page["frame_id"], "crop_bbox_normalized_top_left": [0, 0, .58, .35],
            "source_full_page_visible": True}
    result = _validate_with_ocr(decision, crop, "upd", "s", "p", "test", page)
    assert result.document_type == "upd"
    assert {0, 1, 2} <= {e.region_index for e in result.classification_evidence}


@pytest.mark.parametrize("failure", ["different_pass", "different_column", "large_gap", "reference", "below_header"])
def test_wrapped_title_cannot_join_unrelated_words(failure):
    page = wrapped_page()
    if failure == "different_pass":
        page["regions"][1]["processing_pass"] = "header"
    elif failure == "different_column":
        page["regions"][1]["bbox"][0] = .75
    elif failure == "large_gap":
        page["regions"][1]["bbox"][1] = .30
    elif failure == "reference":
        page["regions"].append(region("Приложение к", .26, .132, .09, .008))
    elif failure == "below_header":
        for row in page["regions"]:
            row["bbox"][1] += .4
    assert _wrapped_upd_titles(page) == []


def test_adjacent_left_reference_is_not_an_own_title_or_neighbor_identity():
    primary = wrapped_page()
    neighbor = deepcopy(primary)
    neighbor["frame_id"] = "different-frame"
    neighbor["regions"].append(region("Приложение к", .05, .145, .20, .009))
    assert _wrapped_upd_titles(neighbor) == []
    eligibility = neighbor_eligibility(primary, neighbor, "upd", ["invoice_number"])
    assert not eligibility["eligible"]
    assert eligibility["reason"] == "neighbor_own_form_heading_not_confirmed"
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    decision = VisionDecision(document_type="upd", heading_text="Счет-фактура № 00017", fields=fields)
    crop = {"frame_id": neighbor["frame_id"], "crop_bbox_normalized_top_left": [0, 0, .58, .35],
            "source_full_page_visible": True}
    assert _validate_with_ocr(decision, crop, "upd", "s", "p", "test", neighbor).document_type == "unknown"


@pytest.mark.parametrize("qualifier", [
    region("Приложение к", .05, .145, .10, .009),
    region("Приложение к", .70, .145, .20, .009),
    region("Приложение к", .05, .145, .20, .009, "another_pass"),
])
def test_unrelated_column_or_pass_qualifier_cannot_erase_own_title(qualifier):
    page = wrapped_page()
    page["regions"].append(qualifier)
    assert len(_wrapped_upd_titles(page)) == 1


@pytest.mark.parametrize("other_heading,visual_heading", [
    ("CMR", "CMR"),
    ("Товарно-транспортная накладная", "Счет-фактура № 00017"),
])
def test_wrapped_upd_title_does_not_override_foreign_or_mixed_own_form_proof(other_heading, visual_heading):
    page = wrapped_page()
    page["regions"].append(region(other_heading, .62, .145, .25, .02))
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    decision = VisionDecision(document_type="upd", heading_text=visual_heading, fields=fields)
    crop = {"frame_id": page["frame_id"], "crop_bbox_normalized_top_left": [0, 0, .90, .35],
            "source_full_page_visible": True}
    result = _validate_with_ocr(decision, crop, "upd", "s", "p", "test", page)
    assert result.document_type == "unknown" and result.fields == {}


def test_wrapped_title_does_not_skip_neighbor_registration(tmp_path, monkeypatch):
    from PIL import Image
    image = tmp_path / "actual-paper.png"
    Image.new("RGB", (320, 200), "white").save(image)
    primary = wrapped_page()
    primary["processing"] = {"source_path": str(image)}
    neighbor = deepcopy(primary)
    neighbor["frame_id"] = "different-frame"
    calls = []
    def no_registration(*features):
        calls.append(features)
        return {"same_page": False, "reason": "insufficient_shared_paper_features"}
    monkeypatch.setattr("sever.video._pair_evidence", no_registration)
    result = neighbor_eligibility(primary, neighbor, "upd", ["invoice_number"])
    assert not result["eligible"]
    assert result["reason"] == "same_leaf_registration_not_confirmed"
    assert len(calls) == 1


def test_wrapped_title_does_not_override_incompatible_own_identity():
    primary, neighbor = wrapped_page(), wrapped_page()
    neighbor["frame_id"] = "different-frame"
    neighbor["regions"][3]["text"] = "Счет-фактура № 00018"
    result = neighbor_eligibility(primary, neighbor, "upd", ["invoice_number"])
    assert not result["eligible"]
    assert result["reason"] == "different_own_header_values"
