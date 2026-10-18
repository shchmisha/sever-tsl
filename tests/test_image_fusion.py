"""Actual distinct frame identities, literal conflicts and own-role isolation."""
from copy import deepcopy

import pytest

from sever.image_fusion import _field_comparison, fuse_readings
from sever.schema import DocumentResult, Evidence, FIELDS_BY_TYPE, FieldValue


def reading(frame, *, kind="upd", values=None, absent=(), blank_proof=True):
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE[kind]}
    for name, raw in (values or {}).items():
        fields[name] = FieldValue(status="recognized", raw_value=raw, normalized_value=raw,
            evidence=[Evidence(frame_id=frame, text=raw, bbox=[.2, .2, .5, .03])])
    for name in absent:
        fields[name] = FieldValue(status="not_present", evidence=[Evidence(frame_id=frame,
            text="", bbox=[.2, .2, .5, .03] if blank_proof else None)])
    return DocumentResult(source_id="source", page_id="leaf", document_type=kind,
        classification_evidence=[Evidence(frame_id=frame, text="Own heading " + kind, bbox=[.1, .1, .7, .05])],
        fields=fields, notes=["Raw image reading " + frame])


def fuse(items):
    return fuse_readings(items, source_id="source", page_id="leaf")


def test_initial_unknown_does_not_veto_two_supported_views_or_change_raw_readings():
    items = [reading("f0", kind="unknown"), reading("f1", values={"consignor": 'ООО «Поставщик»'}),
             reading("f2", values={"consignor": 'ооо   «поставщик»'})]
    before = deepcopy([item.model_dump() for item in items])
    result, meta = fuse(items)
    assert result.document_type == "upd" and meta["type_confirmation"]["confirmed"]
    assert result.fields["consignor"].status == "recognized"
    assert result.fields["consignor"].raw_value == 'ООО «Поставщик»'
    assert result.fields["consignor"].confidence is None
    assert meta["coverage"]["confirmed_fields"] == 1
    assert "consignee" in meta["remaining_fields"]
    assert [item.model_dump() for item in items] == before
    assert meta["readings"][2]["fields"]["consignor"]["raw_value"] == items[2].fields["consignor"].raw_value


def test_repeated_same_frame_cannot_confirm_type():
    one = reading("same", values={"invoice_number": "1234"})
    result, meta = fuse([one, one.model_copy(deep=True)])
    assert result.document_type == "unknown" and result.fields == {}
    assert meta["type_confirmation"]["frame_ids_by_type"] == {"upd": ["same"]}
    assert meta["type_confirmation"]["status"] == "pending"


def test_type_confirmation_requires_actual_evidence_frame_ids():
    items = [reading("f1"), reading("f2")]
    items[1].classification_evidence = []
    result, meta = fuse(items)
    assert result.document_type == "unknown" and not meta["type_confirmation"]["confirmed"]


def test_type_confirmation_uses_source_frame_ids_without_heading_text_semantics():
    items = [reading(frame, values={"invoice_number": "123"}) for frame in ["f1", "f2"]]
    for item in items:
        item.classification_evidence[0].text = ""
    result, meta = fuse(items)
    assert result.document_type == "upd" and meta["type_confirmation"]["confirmed"]
    assert meta["type_confirmation"]["frame_ids_by_type"] == {"upd": ["f1", "f2"]}
    assert result.fields["invoice_number"].raw_value == "123"


@pytest.mark.parametrize("missing_evidence", [[], [Evidence(frame_id=" ", text="", bbox=[0, 0, 1, 1])]])
def test_ungrounded_competing_type_cannot_erase_four_grounded_readings(missing_evidence):
    items = [reading(f"f{index}", values={"invoice_number": "0042"}) for index in range(4)]
    invalid = reading("f4", kind="ttn", values={"document_number": "unproven type proposal"})
    invalid.classification_evidence = missing_evidence
    invalid.errors = ["Model document type has no supplied-panel heading evidence"]
    items.append(invalid)
    before = deepcopy([item.model_dump(mode="json") for item in items])

    result, meta = fuse(items)

    assert result.document_type == "upd"
    assert result.fields["invoice_number"].raw_value == "0042"
    assert result.fields["invoice_number"].status == "recognized"
    assert meta["type_confirmation"]["frame_ids_by_type"] == {"upd": ["f0", "f1", "f2", "f3"]}
    assert result.errors == invalid.errors
    assert meta["readings"] == before
    assert [item.model_dump(mode="json") for item in items] == before


def test_field_structural_error_does_not_discard_grounded_competing_type():
    items = [reading("f0"), reading("f1"), reading("f2", kind="ttn")]
    items[-1].errors = ["consignee: Field cites a panel that was not supplied for this frame"]
    result, meta = fuse(items)
    assert result.document_type == "unknown" and result.fields == {}
    assert meta["type_confirmation"]["status"] == "conflict"
    assert meta["type_confirmation"]["frame_ids_by_type"] == {"ttn": ["f2"], "upd": ["f0", "f1"]}
    assert result.errors == [*items[-1].errors, "image_fusion:conflicting_document_types"]


def test_field_needs_its_own_distinct_frame_evidence_not_only_type_consensus():
    items = [reading("f1", values={"invoice_number": "№ 1234"}), reading("f2")]
    result, meta = fuse(items)
    assert result.document_type == "upd"
    assert result.fields["invoice_number"].status == "unreadable"
    assert result.fields["invoice_number"].normalized_value is None
    assert meta["field_confirmations"]["invoice_number"]["value_groups"][0]["frame_ids"] == ["f1"]
    items.append(reading("f3", values={"invoice_number": "№  1234"}))
    refined, _ = fuse(items)
    assert refined.fields["invoice_number"].status == "recognized"
    assert refined.fields["invoice_number"].raw_value == "№ 1234"
    assert refined.fields["invoice_number"].normalized_value == "№ 1234"


@pytest.mark.parametrize("name,raw", [
    ("invoice_date", "27.08.26"),
    ("invoice_date", "31.02.2026"),
    ("invoice_date", "после отгрузки"),
    ("invoice_number", "№"),
    ("invoice_number", "0042 / б"),
    ("consignor_address", "ООО «Название в поле адреса»"),
    ("consignor", "он же"),
])
def test_matching_model_raw_values_are_never_rejected_by_field_semantics(name, raw):
    items = [reading(frame, values={name: raw}) for frame in ["f1", "f2"]]
    for item in items:
        item.fields[name].normalized_value = None
    before = deepcopy([item.model_dump() for item in items])
    result, meta = fuse(items)
    field = result.fields[name]
    assert field.status == "recognized" and field.raw_value == raw
    assert field.normalized_value is None
    assert meta["field_confirmations"][name]["confirmed"]
    assert meta["field_confirmations"][name]["value_groups"][0]["raw_values"] == [raw]
    assert [item.model_dump() for item in items] == before


@pytest.mark.parametrize("name,first,second", [
    ("invoice_date", "15.02.2020", "15 февраля 2020"),
    ("invoice_date", "15.02.2020", "15.02.2020 отгрузка 16.02.2020"),
    ("invoice_number", "№ 1234", "1234"),
    ("invoice_number", "12 / 3", "12/3"),
    ("consignor", 'ООО «Поставщик»', 'общество с ограниченной ответственностью «Поставщик»'),
    ("consignor", 'ООО «Поставщик»', 'ООО «Поставщик Е»'),
])
def test_different_model_raw_values_are_not_semantically_merged(name, first, second):
    items = [reading("f1", values={name: first}), reading("f2", values={name: second})]
    result, meta = fuse(items)
    field = result.fields[name]
    assert field.status == "conflict" and field.raw_value is None and field.normalized_value is None
    assert {group["raw_values"][0] for group in meta["field_confirmations"][name]["value_groups"]} == {first, second}
    assert [row["fields"][name]["raw_value"] for row in meta["readings"]] == [first, second]


def test_generic_unicode_case_whitespace_comparison_preserves_first_literal_value():
    first, second = 'ООО «Cafe\u0301»', '  ооо   «CAFÉ» '
    items = [reading("f1", values={"consignor": first}), reading("f2", values={"consignor": second})]
    items[0].fields["consignor"].normalized_value = "Existing original value"
    result, meta = fuse(items)
    assert result.fields["consignor"].status == "recognized"
    assert result.fields["consignor"].raw_value == first
    assert result.fields["consignor"].normalized_value == "Existing original value"
    assert meta["field_confirmations"]["consignor"]["value_groups"][0]["raw_values"] == [first, second]


@pytest.mark.parametrize("name,kind", [
    ("consignor", "upd"), ("consignee", "upd"),
    ("supplier", "torg12"), ("customer", "trn"),
])
@pytest.mark.parametrize("second", [
    'ООО «Поставщик-Север»', 'ооо „Поставщик-Север“',
    'ООО“Поставщик-Север”', ' ООО  " Поставщик-Север " ',
])
def test_organization_quote_presentation_agrees_without_rewriting_raw(name, kind, second):
    first = 'ООО "Поставщик-Север"'
    items = [reading("f1", kind=kind, values={name: first}),
             reading("f2", kind=kind, values={name: second})]
    items[0].fields[name].normalized_value = None
    before = deepcopy([item.model_dump() for item in items])
    result, meta = fuse(items)
    field = result.fields[name]
    assert field.status == "recognized" and field.raw_value == first
    assert field.normalized_value is None
    groups = meta["field_confirmations"][name]["value_groups"]
    assert len(groups) == 1 and groups[0]["frame_ids"] == ["f1", "f2"]
    assert groups[0]["raw_values"] == [first, second]
    assert [item.model_dump() for item in items] == before
    assert [row["fields"][name]["raw_value"] for row in meta["readings"]] == [first, second]
    assert [proof.text for proof in field.evidence] == [first, second]


@pytest.mark.parametrize("name", ["consignor_address", "consignee_address"])
@pytest.mark.parametrize("first,second", [
    ("454004, Челябинская обл, Челябинск г, Братьев Кашириных ул, дом № 158, квартира 59",
     "454004, Челябинская обл., Челябинск г., Братьев Кашириных ул., дом № 158, квартира 59"),
    ("ул. Озерная, д. 2", "ул Озерная д 2"),
    ("г. Москва; ул. Озерная, д. 2", "г Москва ул Озерная д 2"),
    ("ул. Озерная, д. 21, корп.16", "ул Озерная д 21,корп 16"),
    ("ООО «Склад» , г. Москва, д. 2", 'ООО "Склад",г Москва д 2'),
    ("д. 12 / 3, кв. 4", "д 12/3,кв 4"),
])
def test_address_abbreviation_and_component_layout_agree_with_literal_proofs(name, first, second):
    items = [reading("f1", values={name: first}), reading("f2", values={name: second})]
    result, meta = fuse(items)
    assert result.fields[name].status == "recognized"
    assert result.fields[name].raw_value == first
    assert result.fields[name].normalized_value == first
    group = meta["field_confirmations"][name]["value_groups"]
    assert len(group) == 1 and group[0]["raw_values"] == [first, second]
    assert [proof.text for proof in result.fields[name].evidence] == [first, second]


@pytest.mark.parametrize("name,first,second", [
    ("consignee_address", "д. 21, корп.16", "д. 21, соор.16"),
    ("consignor_address", "д. 12, кв. 3", "д. 13, кв. 3"),
    ("consignor_address", "д. 12, кв. 3", "д. 12, кв. 4"),
    ("consignor_address", "д. 12, 3", "д. 12 3"),
    ("consignor_address", "д. 12, А", "д. 12 А"),
    ("consignor_address", "д. 12А,3", "д. 12А 3"),
    ("consignor_address", "д. 12.3", "д. 12 3"),
    ("consignor_address", "д. 12/3", "д. 123"),
    ("consignor_address", "д. 12/3", "д. 12-3"),
    ("consignor_address", "д. 12-А", "д. 12А"),
    ("consignor_address", "корп.16", "корп16"),
    ("consignor", 'ООО «А.Б»', 'ООО «А Б»'),
    ("consignor", 'ООО «Север-Экспресс»', 'ООО «Север Экспресс»'),
    ("consignor", 'ООО «А, Б»', 'ООО «АБ»'),
    ("consignor", 'ООО «А, Б»', 'ООО «А Б»'),
    ("consignor_address", 'ООО «А, Б», г. Москва', 'ООО «А Б», г. Москва'),
    ("consignor", 'ООО «Север 1,2»', 'ООО «Север 1 2»'),
    ("consignor", 'ООО «Поставщик 1»', 'ООО «Поставщик 2»'),
    ("consignor", 'ООО «Поставщик»', 'АО «Поставщик»'),
    ("consignor", 'ООО «Поставщик»', 'ООО Поставщик'),
    ("invoice_number", "2560", "2561"),
    ("invoice_number", "12 / 3", "12/3"),
    ("invoice_number", "№ 2560", "2560"),
    ("invoice_date", "20.08.2026", "20 августа 2026 г."),
])
def test_presentation_equivalence_never_merges_different_words_or_numbering(name, first, second):
    # A repeated majority cannot hide one genuinely different model proposal.
    items = [reading("f1", values={name: first}), reading("f2", values={name: second}),
             reading("f3", values={name: first})]
    result, meta = fuse(items)
    assert result.fields[name].status == "conflict"
    assert result.fields[name].raw_value is None and result.fields[name].normalized_value is None
    assert len(meta["field_confirmations"][name]["value_groups"]) == 2
    assert [row["fields"][name]["raw_value"] for row in meta["readings"]] == [first, second, first]


def test_formatting_equivalence_still_requires_two_actual_field_frames():
    items = [reading("f1", values={"consignor": 'ООО "Поставщик"'}),
             reading("f2", values={"consignor": 'ООО «Поставщик»'})]
    items[1].fields["consignor"].evidence[0].frame_id = "f1"
    result, meta = fuse(items)
    assert result.document_type == "upd"
    assert result.fields["consignor"].status == "unreadable"
    assert meta["field_confirmations"]["consignor"]["value_groups"][0]["frame_ids"] == ["f1"]


def test_unsupported_field_names_keep_strict_literal_comparison():
    assert _field_comparison("future_field", 'ООО «Поставщик»') != _field_comparison(
        "future_field", 'ООО "Поставщик"')
    assert _field_comparison("document_number", "12, 3") != _field_comparison("document_number", "12 3")


def test_duplicate_field_frame_ids_cannot_supply_two_view_confirmation():
    items = [reading(frame, values={"invoice_number": "1234"}) for frame in ["f1", "f2"]]
    items[1].fields["invoice_number"].evidence[0].frame_id = "f1"
    result, _ = fuse(items)
    assert result.document_type == "upd"
    assert result.fields["invoice_number"].status == "unreadable"


def test_matching_model_normalization_cannot_hide_different_literal_digits():
    items = [reading("f1", values={"invoice_number": "1234"}),
             reading("f2", values={"invoice_number": "1235"})]
    for item in items:
        item.fields["invoice_number"].normalized_value = "1234"
    result, _ = fuse(items)
    assert result.fields["invoice_number"].status == "conflict"


@pytest.mark.parametrize("first,second", [("1234", "1235"), ("А1234", "A1234")])
def test_numeric_or_script_disagreement_is_not_repaired_or_majority_selected(first, second):
    items = [reading("f1", values={"invoice_number": first}),
             reading("f2", values={"invoice_number": second}),
             reading("f3", values={"invoice_number": first})]
    result, meta = fuse(items)
    field = result.fields["invoice_number"]
    assert field.status == "conflict" and field.raw_value is None and field.normalized_value is None
    assert {e.text for e in field.evidence} == {first, second}
    assert len(meta["field_confirmations"]["invoice_number"]["value_groups"]) == 2
    assert [item["fields"]["invoice_number"]["raw_value"] for item in meta["readings"]] == [first, second, first]


def test_cross_supported_type_disagreement_returns_unknown_without_forcing_fields():
    items = [reading("f1"), reading("f2"), reading("f3", kind="ttn")]
    result, meta = fuse(items)
    assert result.document_type == "unknown" and result.fields == {}
    assert "image_fusion:conflicting_document_types" in result.errors
    assert meta["type_confirmation"]["status"] == "conflict"
    assert {e.frame_id for e in result.classification_evidence} == {"f1", "f2", "f3"}


def test_other_role_value_never_fills_an_unreadable_party():
    items = [reading(frame, values={"consignee": 'ООО "Получатель"'}) for frame in ["f1", "f2"]]
    result, _ = fuse(items)
    assert result.fields["consignee"].status == "recognized"
    assert result.fields["consignor"].status == "unreadable" and result.fields["consignor"].raw_value is None


def test_absence_requires_two_located_blank_views_and_conflicts_with_a_read_value():
    one = reading("f1", absent=["invoice_number"])
    result, _ = fuse([one, reading("f2")])
    assert result.fields["invoice_number"].status == "unreadable"
    result, meta = fuse([one, reading("f2", absent=["invoice_number"])])
    assert result.fields["invoice_number"].status == "not_present"
    assert meta["field_confirmations"]["invoice_number"]["blank_frame_ids"] == ["f1", "f2"]
    result, _ = fuse([one, reading("f2", values={"invoice_number": "1234"})])
    assert result.fields["invoice_number"].status == "conflict"


def test_missing_blank_area_cannot_turn_unknown_fields_into_not_present():
    items = [reading(frame, absent=["invoice_number"], blank_proof=False) for frame in ["f1", "f2"]]
    result, _ = fuse(items)
    assert result.fields["invoice_number"].status == "unreadable"


def test_explicit_blank_evidence_preserves_its_actual_own_caption():
    items = [reading(frame, absent=["invoice_number"]) for frame in ["f1", "f2"]]
    for item in items:
        item.fields["invoice_number"].evidence[0].text = "Счёт-фактура №"
    result, _ = fuse(items)
    assert result.fields["invoice_number"].status == "not_present"
    assert [e.text for e in result.fields["invoice_number"].evidence] == ["Счёт-фактура №"] * 2


def test_all_unknown_preserves_actual_readings_and_heading_evidence():
    items = [reading("f1", kind="unknown"), reading("f2", kind="unknown")]
    result, meta = fuse(items)
    assert result.document_type == "unknown" and result.fields == {}
    assert len(result.classification_evidence) == 2
    assert all(item.notes[0] in result.notes for item in items)
    assert len(meta["readings"]) == 2 and meta["coverage"]["reading_count"] == 2
    assert meta["type_confirmation"]["status"] == "unconfirmed"


def test_empty_readings_and_foreign_leaf_are_explicit():
    result, meta = fuse([])
    assert result.document_type == "unknown" and meta["coverage"]["reading_count"] == 0
    other = reading("f1"); other.page_id = "another-leaf"
    with pytest.raises(ValueError, match="same source and visible leaf"):
        fuse([other])


def test_field_evidence_survives_missing_heading_on_same_declared_form():
    items = [reading("f0"), reading("f1", values={"invoice_number": "00042"}),
             reading("f2", values={"invoice_number": "00042"})]
    items[1].classification_evidence = []
    items[1].errors = ["Model document type has no supplied-panel heading evidence"]
    before = deepcopy([item.model_dump() for item in items])
    result, meta = fuse(items)
    assert result.document_type == "upd"
    assert result.fields["invoice_number"].raw_value == "00042"
    assert meta["field_confirmations"]["invoice_number"]["value_groups"][0]["frame_ids"] == ["f1", "f2"]
    assert meta["type_confirmation"]["frame_ids_by_type"] == {"upd": ["f0", "f2"]}
    assert [item.model_dump() for item in items] == before


def test_missing_heading_does_not_hide_contradictory_own_field():
    items = [reading("f0", values={"invoice_number": "00042"}),
             reading("f1", values={"invoice_number": "00042"}),
             reading("f2", values={"invoice_number": "00043"})]
    items[2].classification_evidence = []
    result, meta = fuse(items)
    assert result.document_type == "upd"
    assert result.fields["invoice_number"].status == "conflict"
    assert len(meta["field_confirmations"]["invoice_number"]["value_groups"]) == 2


def test_numeric_strong_agreement_accepts_many_identical_views_over_one_glyph_dissent():
    items = [reading(f"f{n}", values={"invoice_number": "2559"}) for n in range(3)]
    items.append(reading("f9", values={"invoice_number": "2660"}))
    result, meta = fuse(items)
    field = result.fields["invoice_number"]
    assert field.status == "recognized" and field.raw_value == "2559"
    assert "strong agreement" in field.reason.lower() and "preserved" in field.reason
    assert {e.text for e in field.evidence} == {"2559", "2660"}
    strong = meta["field_confirmations"]["invoice_number"]["strong_agreement"]
    assert strong["accepted_comparison"] == "2559" and strong["agreeing_frame_ids"] == ["f0", "f1", "f2"]
    assert strong["dissent"] == [{"comparison": "2660", "raw_values": ["2660"], "frame_ids": ["f9"]}]
    assert strong["policy"]["version"] == "numeric-strong-agreement-v1"
    # Every raw reading, including the dissent, remains in the record.
    assert [item["fields"]["invoice_number"]["raw_value"] for item in meta["readings"]] == ["2559"] * 3 + ["2660"]


@pytest.mark.parametrize("agreeing,dissent", [
    (2, ["2660"]),                 # fewer than three agreeing views
    (3, ["2660", "2650"]),         # two dissenting views
    (3, ["7781"]),                 # dissent too far from the leader (edit distance 4)
    (5, ["2660", "2560"]),         # ratio 5:2 is below 3:1
])
def test_numeric_strong_agreement_stays_a_conflict_outside_its_bounds(agreeing, dissent):
    items = [reading(f"f{n}", kind="ttn", values={"document_number": "2559"}) for n in range(agreeing)]
    items += [reading(f"d{n}", kind="ttn", values={"document_number": value}) for n, value in enumerate(dissent)]
    result, meta = fuse(items)
    assert result.fields["document_number"].status == "conflict"
    assert "strong_agreement" not in meta["field_confirmations"]["document_number"]


def test_strong_agreement_never_applies_to_names_addresses_or_blank_claims():
    names = [reading(f"f{n}", values={"consignee": 'ООО "Получатель"'}) for n in range(4)]
    names.append(reading("f9", values={"consignee": 'ООО "Получитель"'}))
    assert fuse(names)[0].fields["consignee"].status == "conflict"
    blank = [reading(f"f{n}", values={"invoice_number": "2559"}) for n in range(3)]
    blank.append(reading("f9", absent=["invoice_number"]))
    assert fuse(blank)[0].fields["invoice_number"].status == "conflict"
