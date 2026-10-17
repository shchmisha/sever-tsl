import asyncio

from sever.agents import BatchBudgetExceeded
from sever.schema import DocumentResult, Evidence, FieldValue, FIELDS_BY_TYPE


def reading(frame, complete=False):
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE["ttn"]}
    if complete:
        for name in fields:
            value = "12" if name.endswith("number") else "18.08.2026" if name.endswith("date") else 'ООО "Источник"'
            fields[name] = FieldValue(status="recognized", raw_value=value, normalized_value=value,
                                     evidence=[Evidence(frame_id=frame, text=value)])
    return DocumentResult(source_id="source", page_id="leaf", document_type="ttn", fields=fields)


def setup(monkeypatch):
    import sever.multiview as multi
    import sever.quality as quality
    monkeypatch.setattr(multi, "neighbor_eligibility", lambda *args: {"eligible": True})
    monkeypatch.setattr(quality, "assess_frame_quality", lambda frame: {"accepted": True})
    return multi, {"page_id": "leaf", "selected_frame_id": "a", "frames": [{"frame_id": "a", "timestamp_s": 0}],
                   "quality_gate": {"accepted_frame_ids": ["a"]}}


def test_more_than_eight_frames_can_be_read_and_missing_fields_recovered(monkeypatch, tmp_path):
    multi, page = setup(monkeypatch)
    called = []
    records = []
    closed = []

    def pool(*args):
        try:
            for index in range(1, 30):
                yield {"frame_id": str(index), "timestamp_s": index / 4}
        finally:
            closed.append(True)

    async def reader(views):
        frame = views[0]["frame_id"]
        called.append(frame)
        return reading(frame, int(frame) >= 10)

    result = asyncio.run(multi.compare_page_views(tmp_path/"source.mp4", page, [{"frame_id": "a"}], reading("a"),
        "ttn", out=tmp_path, local_ocr=lambda frame, _: {"frame_id": frame["frame_id"]}, reader=reader,
        record=records.append, additional_frames=pool))
    assert len(called) == 11 and len(records) == 11
    assert all(value.status == "recognized" for value in result.fields.values())
    assert all({e.frame_id for e in value.evidence} == {"10", "11"} for value in result.fields.values())
    assert page["neighbor_comparison"]["status"] == "all_fields_corroborated"
    assert page["neighbor_comparison"]["frame_count_limit"] is None
    assert closed == [True]


def test_agreement_in_retained_frames_does_not_materialize_extra_pool(monkeypatch, tmp_path):
    multi, page = setup(monkeypatch)
    page["frames"].append({"frame_id": "b", "timestamp_s": .5})
    page["quality_gate"]["accepted_frame_ids"].append("b")
    async def reader(views):
        return reading(views[0]["frame_id"], True)
    def forbidden_pool(*args):
        raise AssertionError("Resolved source readings need no extra video decoding")
    result = asyncio.run(multi.compare_page_views(tmp_path/"source.mp4", page, [{"frame_id": "a"}], reading("a", True),
        "ttn", out=tmp_path, local_ocr=lambda frame, _: {"frame_id": frame["frame_id"]}, reader=reader,
        record=lambda _: None, additional_frames=forbidden_pool))
    assert all({e.frame_id for e in value.evidence} == {"a", "b"} for value in result.fields.values())
    assert page["neighbor_comparison"]["status"] == "all_fields_corroborated"


def test_shared_cost_budget_is_explicit_and_preserves_primary_answer(monkeypatch, tmp_path):
    multi, page = setup(monkeypatch)
    before = reading("a", True)
    async def reader(views):
        raise BatchBudgetExceeded("shared budget")
    result = asyncio.run(multi.compare_page_views(tmp_path/"source.mp4", page, [{"frame_id": "a"}], before,
        "ttn", out=tmp_path, local_ocr=lambda frame, _: {"frame_id": frame["frame_id"]}, reader=reader,
        record=lambda _: None, additional_frames=lambda *args: iter([{"frame_id": "b", "timestamp_s": .5}])))
    assert result.fields == before.fields
    assert page["neighbor_comparison"]["status"] == "shared_model_budget_exhausted"
    assert len(page["neighbor_comparison"]["remaining_fields"]) == 4


def test_pool_failure_exposes_unconfirmed_fields_and_halt(monkeypatch, tmp_path):
    multi, page = setup(monkeypatch)
    recorded = []
    def pool(*args):
        raise OSError("fixture decode failure")
    async def forbidden(views):
        raise AssertionError("Failed frame extraction must not call the model")
    result = asyncio.run(multi.compare_page_views(tmp_path/"source.mp4", page, [{"frame_id": "a"}], reading("a",True),
        "ttn", out=tmp_path,local_ocr=lambda *args: None,reader=forbidden,record=recorded.append,additional_frames=pool))
    assert page["neighbor_comparison"]["status"] == "additional_frame_pool_failed"
    assert len(page["neighbor_comparison"]["remaining_fields"]) == 4
    assert recorded == [{"status": "additional_frame_pool_failed", "error_type": "OSError"}]
    assert any(error.startswith("multiview: additional_frame_pool_failed") for error in result.errors)


def test_grounded_conflict_is_reported_without_buying_useless_majority_votes(monkeypatch, tmp_path):
    multi, page = setup(monkeypatch)
    page["frames"].append({"frame_id":"b","timestamp_s":.5})
    page["quality_gate"]["accepted_frame_ids"].append("b")
    async def reader(views):
        result = reading("b",True)
        result.fields["document_number"] = FieldValue(status="recognized",raw_value="13",normalized_value="13",evidence=[Evidence(frame_id="b",text="13")])
        return result
    def forbidden(*args):
        raise AssertionError("A reported source dispute cannot be resolved by extra model voting")
    result = asyncio.run(multi.compare_page_views(tmp_path/"source.mp4",page,[{"frame_id":"a"}],reading("a",True),
        "ttn",out=tmp_path,local_ocr=lambda frame,_:{"frame_id":frame["frame_id"]},reader=reader,
        record=lambda _:None,additional_frames=forbidden))
    assert result.fields["document_number"].status == "conflict"
    assert page["neighbor_comparison"]["status"] == "grounded_conflicts_preserved"
    assert page["neighbor_comparison"]["remaining_fields"] == ["document_number"]
