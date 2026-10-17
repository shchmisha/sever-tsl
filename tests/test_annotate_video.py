"""Offline source-view selection and coordinate grounding, no video/model calls."""
from copy import deepcopy

import pytest

from tools.annotate_video import anchors, select_reference


def region(index, text="ООО Печатный продавец", box=None):
    return {"region_index": index, "text": text, "bbox": box or [.1, .2, .3, .05]}


def evidence(frame_id, index=0, box=None, text="ООО Печатный продавец"):
    return {"frame_id": frame_id, "region_index": index, "text": text,
            "bbox": box or [.1, .2, .3, .05]}


def field(*items, status="recognized", value="ООО Печатный продавец"):
    return {"status": status, "raw_value": value if status == "recognized" else None,
            "normalized_value": None, "evidence": list(items)}


def observation(frame_id, *regions):
    return {"frame_id": frame_id, "timestamp_s": 1., "regions": list(regions),
            "processing": {"source_path": f"/source/{frame_id}.jpg",
                           "ocr_image_path": f"/ocr/{frame_id}.png"}}


def page(fields, *frame_ids):
    return {"selected_frame_id": frame_ids[0], "result": {"fields": fields},
            "frames": [{"frame_id": frame_id, "path": f"/source/{frame_id}.jpg", "timestamp_s": index}
                       for index, frame_id in enumerate(frame_ids)]}


def test_neighbor_regions_with_same_indices_never_become_reference_anchors():
    """A same-leaf neighbor box is still in a different image coordinate system."""
    source = field(evidence("primary"), evidence("neighbor", box=[.5, .6, .3, .05]))
    result = anchors("consignor", source, [region(0)], "primary")
    assert len(result) == 1
    assert result[0]["frame_id"] == "primary"
    assert result[0]["bbox"] == [.1, .2, .3, .05]
    only_neighbor = field(evidence("neighbor"))
    assert anchors("consignor", only_neighbor, [region(0)], "primary") == []


def test_foreign_pixel_panel_does_not_search_matching_reference_ocr():
    """Matching words in the reference cannot repair a foreign evidence frame."""
    source = field(evidence("neighbor", index=None))
    assert anchors("consignor", source, [region(0)], "primary") == []
    local = field(evidence("primary", index=None))
    result = anchors("consignor", local, [region(0)], "primary")
    assert result[0]["frame_id"] == "primary"
    assert result[0]["kind"] == "literal_ocr_inside_recorded_pixel_panel"


def test_stale_or_missing_ocr_region_cannot_be_replaced_by_another_box():
    assert anchors("consignor", field(evidence("primary", index=7)), [region(0)], "primary") == []
    stale = field(evidence("primary", box=[.7, .6, .2, .05]))
    assert anchors("consignor", stale, [region(0)], "primary") == []
    # Absent recorded geometry can safely use the actual saved region geometry.
    item = evidence("primary")
    item["bbox"] = None
    assert anchors("consignor", field(item), [region(0)], "primary")[0]["bbox"] == region(0)["bbox"]


@pytest.mark.parametrize("box", [None, [0, 0, 0, .1], [0, 0, .1], [-.1, 0, .3, .2],
                                      [.9, .9, .2, .2], [float("nan"), 0, .1, .1]])
def test_invalid_pixel_panel_is_not_drawable(box):
    item = evidence("primary", index=None)
    item["bbox"] = box
    assert anchors("consignor", field(item), [region(0)], "primary") == []


def test_saved_pixel_panel_without_literal_ocr_remains_explicit_panel_evidence():
    source = field(evidence("primary", index=None), value="Визуально прочитанная компания")
    result = anchors("consignor", source, [region(0, text="Иная строка")], "primary")
    assert result[0]["kind"] == "recorded_pixel_reading_panel"
    assert result[0]["frame_id"] == "primary"
    assert result[0]["region_index"] is None


def test_reference_selection_prefers_full_neighbor_over_partial_primary_without_mutation():
    fields = {"consignor": field(evidence("primary"), evidence("neighbor")),
              "consignee": field(evidence("neighbor", index=1, text="ООО Получатель"), value="ООО Получатель")}
    source_page = page(fields, "primary", "neighbor")
    views = [observation("primary", region(0)),
             observation("neighbor", region(0), region(1, text="ООО Получатель"))]
    original = deepcopy((source_page, views))
    chosen = select_reference(source_page, views, require_all=True)
    assert chosen["frame"]["frame_id"] == chosen["observation"]["frame_id"] == "neighbor"
    assert [name for name, _ in chosen["fields"]] == ["consignor", "consignee"]
    assert chosen["all_recognized_fields_grounded"] is True
    assert chosen["excluded_fields"] == {}
    assert all(anchor["frame_id"] == "neighbor" for items in chosen["evidence"].values() for anchor in items)
    assert (source_page, views) == original


def test_split_source_views_render_only_supported_fields_or_explicitly_fail_strict_mode():
    source_page = page({"consignor": field(evidence("primary")),
                        "consignee": field(evidence("neighbor"))}, "primary", "neighbor")
    views = [observation("neighbor", region(0)), observation("primary", region(0))]
    chosen = select_reference(source_page, views)
    assert chosen["frame"]["frame_id"] == "primary"  # Stable selected-frame tie break.
    assert [name for name, _ in chosen["fields"]] == ["consignor"]
    assert set(chosen["evidence"]) == {"consignor"}
    assert chosen["excluded_fields"] == {
        "consignee": {"reason": "no_evidence_on_reference_frame", "evidence_frame_ids": ["neighbor"]}}
    assert chosen["all_recognized_fields_grounded"] is False
    with pytest.raises(ValueError, match="No single OCR frame grounds all recognized fields.*consignee"):
        select_reference(source_page, views, require_all=True)


def test_observation_must_belong_to_actual_page_frame_and_source_path():
    source_page = page({"consignor": field(evidence("primary")),
                        "consignee": field(evidence("foreign"))}, "primary")
    mismatched = observation("primary", region(0))
    mismatched["processing"]["source_path"] = "/source/another-image.jpg"
    with pytest.raises(ValueError, match="No OCR observation is bound"):
        select_reference(source_page, [mismatched, observation("foreign", region(0))])


def test_unresolved_and_missing_geometry_are_not_counted_as_supported_fields():
    source_page = page({"consignor": field(evidence("primary")),
                        "consignee": field(evidence("primary", index=99)),
                        "invoice_number": field(evidence("primary"), status="conflict")}, "primary")
    chosen = select_reference(source_page, [observation("primary", region(0))])
    assert [name for name, _ in chosen["fields"]] == ["consignor"]
    assert chosen["excluded_fields"] == {
        "consignee": {"reason": "invalid_or_unresolved_reference_geometry", "evidence_frame_ids": ["primary"]}}
    assert chosen["candidate_support"] == [{"frame_id": "primary", "supported_fields": ["consignor"]}]


def test_zero_anchorable_fields_fail_before_starting_video_encoding():
    source_page = page({"consignor": field(evidence("neighbor"))}, "primary")
    with pytest.raises(ValueError, match="No recognized field has drawable evidence"):
        select_reference(source_page, [observation("primary", region(0))])
