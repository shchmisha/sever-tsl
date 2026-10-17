"""Postal contradiction evidence must come from an own address, not tax digits."""
from copy import deepcopy

import pytest

from sever.schema import FIELDS_BY_TYPE
from sever.vision import (VisionDecision, VisionField, _own_address_postal_observations,
                          _role_sources, _validate_with_ocr)


def row(text, x, y, w=.25, h=.012, source_pass="full_image"):
    return {"text": text, "bbox": [x, y, w, h], "processing_pass": source_pass}


def document(tax_caption="ИНН/КПП продавца", *, repeated_address=False):
    # One pass reads the postal code intact. Independent passes split the KPP
    # into an apparent six-digit number underneath its actual tax caption.
    rows = [row("Продавец:", .05, .18, .10),
            row('ООО "Поставляющая сторона"', .25, .196),
            row("Адре‹", .05, .205, .07),
            row("135790, Город, Улица, дом 7, квартира 12", .25, .209, .30, .016),
            row(tax_caption, .05, .230, .14, .008),
            row("246802", .30, .238, .04, .006),
            row("Грузоотправитель и его адрес", .05, .247, .18, .008),
            row("OH же", .25, .254, .04, .005),
            row('ООО "Принимающая сторона"', .25, .265),
            row("Грузополучатель и его адрес", .05, .282, .18),
            row("864200, Иной город, Улица, дом 3", .25, .282, .30),
            row(tax_caption, .05, .230, .14, .008, "header"),
            row("98765/246802", .30, .237, .08, .007, "header")]
    if repeated_address:
        rows.append(row(rows[3]["text"], .25, .210, .30, .016, "header"))
    page = {"frame_id": "independent-source", "regions": rows}
    crop = {"frame_id": page["frame_id"], "crop_bbox_normalized_top_left": [0, .1, .6, .3],
            "source_full_page_visible": True, "role_panels": _role_sources(page, "upd")}
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignor_address"] = VisionField(status="recognized", raw_value=rows[3]["text"],
        panel_id="consignor", evidence_ids=["r3", "r6"], confidence=.9,
        evidence_text="Грузоотправитель и его адрес: он же. Адрес: " + rows[3]["text"])
    decision = VisionDecision(document_type="upd", heading_text="Универсальный передаточный документ", fields=fields)
    return page, crop, decision


@pytest.mark.parametrize("tax_caption", ["ИНН/КПП продавца", "ИННКПП продавца", "КПП продавца"])
def test_repeated_tax_fragments_do_not_conflict_with_a_readable_address(tax_caption):
    page, crop, decision = document(tax_caption)
    before = deepcopy((page, crop, decision.model_dump()))
    assert not crop["role_panels"]["consignor"]["explicit_seller_reference"]
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "recognized"
    assert field.raw_value == decision.fields["consignor_address"].raw_value
    assert field.confidence == .9
    assert "246802" not in field.raw_value
    assert before == (page, crop, decision.model_dump())
    assert page["regions"][7]["text"] == "OH же"  # No homoglyph/value repair.


def test_true_independent_address_postal_disagreement_remains_conflict_with_actual_proof():
    page, crop, decision = document(repeated_address=True)
    decision.fields["consignor_address"].raw_value = "975310, Город, Улица, дом 7, квартира 12"
    decision.fields["consignor_address"].evidence_text = "Грузоотправитель: " + decision.fields["consignor_address"].raw_value
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "conflict" and field.raw_value is None
    assert "postal code conflicts" in field.reason
    proof = {e.region_index: e.text for e in field.evidence if e.region_index is not None}
    assert proof[3] == proof[13] == page["regions"][3]["text"]
    assert 5 not in proof and 12 not in proof
    assert decision.fields["consignor_address"].raw_value.startswith("975310")


def test_omitted_repeated_own_postal_code_remains_unreadable():
    page, crop, decision = document(repeated_address=True)
    decision.fields["consignor_address"].raw_value = "Город, Улица, дом 7, квартира 12"
    decision.fields["consignor_address"].evidence_text = "Грузоотправитель: " + decision.fields["consignor_address"].raw_value
    field = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page).fields["consignor_address"]
    assert field.status == "unreadable" and "omits the postal code" in field.reason
    assert {3, 13} <= {e.region_index for e in field.evidence}


def observations(rows, *, ids=None, caption_id="r0", bounds=None):
    lookup = {f"r{i}": item for i, item in enumerate(rows)}
    panel = {"caption_id": caption_id, "bbox": bounds or [0, 0, 1, 1]}
    return _own_address_postal_observations(ids or list(lookup), panel, lookup)


def test_bare_postal_number_needs_an_actual_own_address_caption():
    rows = [row("Адрес:", .30, .2, .07), row("123890", .40, .2, .06)]
    assert observations(rows) == [("123890", "r1")]
    assert observations(rows[1:], caption_id=None) == []


def test_short_postal_number_in_each_own_address_pass_preserves_real_conflict():
    rows = [row("Грузополучатель", .10, .2, .20), row("Адрес:", .10, .22, .07),
            row("123890", .35, .22, .06), row("Адрес:", .10, .22, .07, source_pass="header"),
            row("123890", .35, .22, .06, source_pass="header")]
    page = {"frame_id": "f", "regions": rows}
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignee_address"] = VisionField(status="recognized", raw_value="890123, Город, дом 4",
        panel_id="consignee", evidence_ids=["r0", "r2"], evidence_text="Грузополучатель 890123, Город, дом 4")
    decision = VisionDecision(document_type="upd", heading_text="Универсальный передаточный документ", fields=fields)
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .4], "source_full_page_visible": True}
    result = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page)
    assert result.fields["consignee_address"].status == "conflict"


def test_nearest_own_address_caption_wins_over_another_columns_tax_caption():
    rows = [row("Адрес:", .45, .2, .07), row("123890", .55, .2, .06),
            row("ИНН/КПП", .20, .2, .10)]
    assert observations(rows) == [("123890", "r1")]


@pytest.mark.parametrize("inline_caption", [True, False])
def test_self_contained_address_in_wide_panel_preserves_true_postal_conflict(inline_caption):
    literal = ("Адрес: " if inline_caption else "") + "123890, Город, улица, дом 4"
    rows = [row("Грузополучатель", .50, .18, .20),
            row(literal, .55, .20, .30), row(literal, .55, .20, .30, source_pass="header"),
            row("ИНН/КПП", .20, .20, .10), row("ИНН/КПП", .20, .20, .10, source_pass="header")]
    panel = {"panel_id": "consignee", "caption_id": "r0", "bbox": [0, .15, 1, .15],
             "ids": ["r0", "r1", "r2"]}
    lookup = {f"r{i}": item for i, item in enumerate(rows)}
    assert _own_address_postal_observations(panel["ids"], panel, lookup) == [("123890", "r1"), ("123890", "r2")]
    page = {"frame_id": "f", "regions": rows}
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields["consignee_address"] = VisionField(status="recognized", raw_value="890123, Город, улица, дом 4",
        panel_id="consignee", evidence_ids=["r0", "r1"], evidence_text="Грузополучатель 890123, Город, улица, дом 4")
    decision = VisionDecision(document_type="upd", heading_text="Универсальный передаточный документ", fields=fields)
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .4], "source_full_page_visible": True,
            "role_panels": {"consignee": panel}}
    result = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page)
    assert result.fields["consignee_address"].status == "conflict"
    assert {1, 2} <= {e.region_index for e in result.fields["consignee_address"].evidence}


def test_aligned_address_caption_precedes_a_closer_tax_caption_from_another_row():
    rows = [row("Адрес:", .10, .20, .07), row("123890", .35, .20, .06),
            row("ИНН/КПП", .20, .207, .10)]
    assert observations(rows) == [("123890", "r1")]


@pytest.mark.parametrize("tax_row", [
    row("ИНН/КПП", .75, .2, .10),  # A caption to the right is another column.
    row("ИНН/КПП", .05, .3, .10),  # A different row cannot own this number.
    row("ИНН/КПП", .30, .2, .10, source_pass="header"),  # Another OCR pass is not this row.
])
def test_unrelated_tax_caption_cannot_remove_an_intact_own_address(tax_row):
    rows = [row("Грузополучатель", .10, .2, .20),
            row("123890, Город, улица, дом 4", .45, .2), tax_row]
    assert observations(rows) == [("123890", "r1")]


def test_tax_caption_outside_own_source_column_cannot_remove_short_postal_code():
    rows = [row("Грузополучатель", .50, .18, .20), row("123890, Город, улица, дом 4", .65, .2),
            row("ИНН/КПП", .40, .2, .07)]
    assert observations(rows, bounds=[.50, .15, .45, .15]) == [("123890", "r1")]


def test_one_pass_duplicated_postal_observation_cannot_create_independent_consensus():
    page, crop, decision = document()
    page["regions"].append(deepcopy(page["regions"][3]))
    crop["role_panels"] = _role_sources(page, "upd")
    decision.fields["consignor_address"].raw_value = "975310, Город, Улица, дом 7, квартира 12"
    decision.fields["consignor_address"].evidence_text = "Грузоотправитель: " + decision.fields["consignor_address"].raw_value
    result = _validate_with_ocr(decision, crop, "upd", "source", "leaf", "test", page)
    assert result.fields["consignor_address"].status == "recognized"
