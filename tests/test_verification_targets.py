from copy import deepcopy
import json

import pytest

from sever.schema import DocumentResult, Evidence, FieldValue, FIELDS_BY_TYPE
from sever.verification_targets import frames_for_verification, needs_detail_verification


FIELD = "consignor_address"


def recognized(value, frame_id="f1"):
    return {"status": "recognized", "raw_value": value,
            "evidence": [{"frame_id": frame_id, "text": "Грузоотправитель и его адрес"}]}


def direct(value, frame_id="f1", kind="upd"):
    return {"document_type": kind, "fields": {FIELD: recognized(value, frame_id)}}


def page(page_id="p1", value="дом 16, стр. 8", count=2, start=1, scores=None):
    result = {"page_id": page_id, "source_id": "video-identity", "document_type": "upd",
              "fields": {FIELD: recognized(value)}}
    frames, views = [], []
    for index in range(count):
        identity = start + index
        frame_id = f"f{identity}"
        path = f"/original/source-{identity}.png"
        timestamp = identity * .5
        frames.append({"frame_id": frame_id, "path": path, "timestamp_s": timestamp,
                       "quality": {"relative_selection_score": scores[index] if scores else 1.0,
                                   "sharpness_laplacian_variance": identity * 10}})
        document = {**direct(value, frame_id), "page_id": page_id, "source_id": "video-identity"}
        views.append({"frame_id": frame_id, "frame_path": path, "document_type": "upd", "document": document,
                      "pixel_proof": {"frame_id": frame_id, "source_path": path,
                                      "source_sha256": f"{identity:064x}", "timestamp_s": timestamp,
                                      "rotation_clockwise_degrees": 90}})
    return {"page_id": page_id, "result": result, "frames": frames, "image_recognition": {"views": views}}


def test_new_joint_suffix_is_verified_even_when_direct_addresses_differ_elsewhere():
    readings = [direct('ООО "Гринфилдс", дом 16, стр. 8', "f1"),
                direct('ООО "Тринфилдс", дом 15, стр. 8', "f2")]
    assert needs_detail_verification(recognized('ООО "Гринфилдс", дом 15, стр. В'), readings, FIELD, "upd")
    # A joint choice of an existing grounded value keeps the established policy.
    assert not needs_detail_verification(readings[0]["fields"][FIELD], readings, FIELD, "upd")


def test_supported_invoice_number_is_not_made_unreadable_by_an_unneeded_reread():
    name = "invoice_number"
    readings = [{"document_type": "upd", "fields": {name: recognized(value, str(index))}}
                for index, value in enumerate(["ГФЛ00056144", "ГФП00056144"])]
    assert not needs_detail_verification(recognized("ГФЛ00056144"), readings, name, "upd")
    assert needs_detail_verification(recognized("ГФБ00056144"), readings, name, "upd")


@pytest.mark.parametrize("value", ["дом 16, стр. В", "дом 16, стр. 8"])
def test_matching_literal_readings_are_not_repaired_or_targeted(value):
    assert not needs_detail_verification(recognized(value), [direct(value, "f1"), direct(value, "f2")], FIELD, "upd")


@pytest.mark.parametrize("status", ["unreadable", "not_present", "conflict"])
def test_missing_or_blank_proposal_is_not_a_verification_target(status):
    assert not needs_detail_verification({"status": status, "raw_value": None}, [direct("стр. 8")], FIELD, "upd")


def test_missing_blank_foreign_and_ungrounded_direct_readings_do_not_create_targets():
    readings = [direct("стр. 8", kind="trn"), {"document_type": "upd", "fields": {FIELD: {"status": "unreadable"}}},
                {"document_type": "upd", "fields": {FIELD: {"status": "not_present"}}},
                {"document_type": "upd", "fields": {FIELD: {"status": "recognized", "raw_value": "стр. 8", "evidence": []}}}]
    assert not needs_detail_verification(recognized("стр. В"), readings, FIELD, "upd")
    assert not needs_detail_verification(recognized("стр. В"), readings, "customer", "upd")


def test_pydantic_readings_and_proposals_are_supported():
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE["upd"]}
    fields[FIELD] = FieldValue(status="recognized", raw_value="стр. 8", evidence=[Evidence(frame_id="f1", text="стр.")])
    reading = DocumentResult(source_id="video", page_id="p1", document_type="upd", fields=fields)
    assert needs_detail_verification(FieldValue(status="recognized", raw_value="стр. В"), [reading], FIELD, "upd")


def test_sharp_original_views_keep_provenance_and_never_expose_candidate_strings():
    item = page(count=4, scores=[.1, .8, .9, .4])
    before = deepcopy(item)
    chosen = frames_for_verification([item], FIELD)
    assert [f["frame_id"] for f in chosen] == ["f3", "f2", "f4"]
    assert chosen[0] == {"frame_id": "f3", "path": "/original/source-3.png", "source_sha256": f"{3:064x}",
                         "timestamp_s": 1.5, "rotation_clockwise_degrees": 90, "source_page_id": "p1"}
    assert "стр." not in json.dumps(chosen, ensure_ascii=False) and "raw_value" not in json.dumps(chosen)
    assert item == before


def test_document_conflict_covers_both_final_candidates_despite_four_noisy_direct_values():
    left = page("p1", "дом 15, стр. В", count=2, start=1, scores=[.99, .98])
    right = page("p2", "дом 16, стр. 8", count=2, start=3, scores=[.1, .09])
    for index, view in enumerate([*left["image_recognition"]["views"], *right["image_recognition"]["views"]]):
        view["document"]["fields"][FIELD]["raw_value"] = f"noisy whole address {index}"
    chosen = frames_for_verification([left, right], FIELD, limit=2)
    assert [(f["source_page_id"], f["frame_id"]) for f in chosen] == [("p1", "f1"), ("p2", "f3")]
    assert len(frames_for_verification([left, right], FIELD)) == 3


def test_conflicting_leaf_covers_original_reading_sides():
    item = page(count=3, scores=[.99, .98, .1])
    item["result"]["fields"][FIELD] = {"status": "conflict", "raw_value": None}
    item["image_recognition"]["views"][2]["document"]["fields"][FIELD]["raw_value"] = "стр. В"
    chosen = frames_for_verification([item], FIELD, limit=2)
    assert {frame["frame_id"] for frame in chosen} == {"f1", "f3"}


def test_root_leaf_input_without_top_level_page_id_uses_result_association():
    item = page()
    del item["page_id"]
    assert {frame["source_page_id"] for frame in frames_for_verification([item], FIELD)} == {"p1"}


@pytest.mark.parametrize("duplicate_key", ["source_sha256", "frame_id", "timestamp_s"])
def test_duplicate_pixels_ids_or_timestamps_never_count_as_independent(duplicate_key):
    item = page()
    views = item["image_recognition"]["views"]
    views[1]["pixel_proof"][duplicate_key] = views[0]["pixel_proof"][duplicate_key]
    assert frames_for_verification([item], FIELD) == []


def test_selection_can_avoid_sharp_duplicate_pixels_to_cover_the_other_side():
    left = page("p1", "стр. В", count=2, start=1, scores=[.9, .8])
    right = page("p2", "стр. 8", count=1, start=3)
    right["image_recognition"]["views"][0]["pixel_proof"]["source_sha256"] = f"{1:064x}"
    chosen = frames_for_verification([left, right], FIELD, limit=2)
    assert {frame["frame_id"] for frame in chosen} == {"f2", "f3"}


def test_single_group_backtracks_when_sharpest_view_has_no_independent_partner():
    item = page(count=3, scores=[.9, .8, .7])
    views = item["image_recognition"]["views"]
    views[1]["pixel_proof"]["source_sha256"] = views[0]["pixel_proof"]["source_sha256"]
    views[2]["pixel_proof"]["timestamp_s"] = views[0]["pixel_proof"]["timestamp_s"]
    item["frames"][2]["timestamp_s"] = item["frames"][0]["timestamp_s"]
    chosen = frames_for_verification([item], FIELD, limit=2)
    assert {frame["frame_id"] for frame in chosen} == {"f2", "f3"}


@pytest.mark.parametrize("property_name,bad", [("source_sha256", "not-a-sha"), ("source_path", ""),
                                              ("rotation_clockwise_degrees", True), ("timestamp_s", float("nan")),
                                              ("frame_id", "foreign")])
def test_invalid_pixel_proof_prevents_false_corroboration(property_name, bad):
    item = page()
    item["image_recognition"]["views"][0]["pixel_proof"][property_name] = bad
    assert frames_for_verification([item], FIELD) == []


def test_missing_candidate_side_is_not_silently_skipped():
    left, right = page("p1", "стр. В"), page("p2", "стр. 8", start=3)
    right["image_recognition"]["views"] = []
    assert frames_for_verification([left, right], FIELD) == []


def test_more_dispute_sides_than_limit_remain_unresolved():
    items = [page(f"p{n}", f"стр. {n}", start=n * 2 + 1) for n in range(4)]
    assert frames_for_verification(items, FIELD) == []


@pytest.mark.parametrize("status", ["unreadable", "not_present"])
def test_genuinely_missing_and_printed_blank_fields_do_not_select_frames(status):
    item = page()
    item["result"]["fields"][FIELD] = {"status": status, "raw_value": None}
    assert frames_for_verification([item], FIELD) == []


@pytest.mark.parametrize("mismatch", ["page_id", "source_id"])
def test_foreign_view_document_identity_is_rejected(mismatch):
    item = page()
    item["image_recognition"]["views"][0]["document"][mismatch] = "foreign"
    assert frames_for_verification([item], FIELD) == []


def test_cross_video_or_cross_form_pages_are_not_mixed():
    left, right = page(), page("p2", start=3)
    right["result"]["source_id"] = "foreign-video"
    assert frames_for_verification([left, right], FIELD) == []
    right["result"]["source_id"] = left["result"]["source_id"]
    right["result"]["document_type"] = "trn"
    assert frames_for_verification([left, right], FIELD) == []


@pytest.mark.parametrize("limit", [0, 1, 4, True])
def test_invalid_frame_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="two or three"):
        frames_for_verification([page()], FIELD, limit=limit)
