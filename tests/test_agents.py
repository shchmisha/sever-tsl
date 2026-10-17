import asyncio
import json

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai.exceptions import ModelHTTPError

from sever.agents import (AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError,
                          ExtractionDecision, FieldProposal, _regions, _model_input,
                          _complete_literal_evidence, analyze_page, validate_decision)
from sever.schema import FIELDS_BY_TYPE


def regions(*texts):
    return _regions([{"frame_id": "frame-1", "regions": [
        {"text": t, "visionIndex": i, "bbox": [0.1, 0.1 + i * 0.05, 0.7, 0.04]}
        for i, t in enumerate(texts)]}])


def proposal(doc_type="torg12", **fields):
    defaults = {name: FieldProposal(status="unreadable") for name in FIELDS_BY_TYPE[doc_type]}
    defaults.update(fields)
    return ExtractionDecision(document_type=doc_type, type_confidence=0.9,
                              classification_evidence_ids=["v0:r0"], fields=defaults)


def validate(decision, data, **kwargs):
    return validate_decision(decision, data, source_id="source", page_id="page-1", **kwargs)


def test_invoice_is_not_upd():
    data = regions("Счет-фактура № 25 от 18.08.2026", "Грузоотправитель ООО Север")
    result = validate(proposal("upd"), data)
    assert result.document_type == "unknown"
    assert result.fields == {}
    assert any("invoice is not UPD" in e for e in result.errors)


@pytest.mark.parametrize("title", ["CMR Международная товарно-транспортная накладная", "ТН-2 Товарная накладная"])
def test_explicit_out_of_scope_forms_are_unknown(title):
    result = validate(proposal("ttn"), regions(title))
    assert result.document_type == "unknown"
    assert not result.fields


def test_upd_fields_keep_raw_ocr_and_normalise_date():
    data = regions("Универсальный передаточный документ", "Счет-фактура № А-025 от 18.08.2026")
    decision = proposal("upd", invoice_number=FieldProposal(raw_value="А-025", status="recognized", evidence_ids=["v0:r1"]),
                        invoice_date=FieldProposal(raw_value="18.08.2026", status="recognized", evidence_ids=["v0:r1"]))
    result = validate(decision, data)
    assert result.document_type == "upd"
    assert set(result.fields) == set(FIELDS_BY_TYPE["upd"])
    assert result.fields["invoice_number"].raw_value == "А-025"
    assert result.fields["invoice_date"].normalized_value == "2026-08-18"
    assert result.fields["invoice_date"].evidence[0].frame_id == "frame-1"


def test_invented_ocr_reference_and_value_are_rejected():
    data = regions("Форма ТОРГ-12", "Поставщик ООО Север")
    decision = proposal(supplier=FieldProposal(raw_value="ООО Юг", status="recognized", evidence_ids=["v0:r1", "v0:r999"]))
    result = validate(decision, data)
    value = result.fields["supplier"]
    assert value.status == "unreadable"
    assert value.normalized_value is None
    assert value.evidence[0].text == "Поставщик ООО Север"
    assert any("invalid OCR evidence" in e for e in result.errors)


def test_transport_order_is_not_transport_consignment_number():
    data = regions("Транспортная накладная №", "Заказ (заявка) № 11844")
    decision = proposal("trn", document_number=FieldProposal(raw_value="11844", status="recognized",
                                                             evidence_ids=["v0:r0", "v0:r1"]))
    result = validate(decision, data)
    assert result.fields["document_number"].status == "unreadable"
    assert result.fields["document_number"].normalized_value is None


def test_unlabeled_transport_reference_is_not_accepted():
    data = regions("Транспортная накладная", "11844")
    result = validate(proposal("trn", document_number=FieldProposal(raw_value="11844", status="recognized",
                                                                    evidence_ids=["v0:r1"])), data)
    assert result.fields["document_number"].status == "unreadable"


def test_absence_on_incomplete_view_is_unreadable_and_invalid_date_stays_unresolved():
    data = regions("Форма ТОРГ-12", "Дата составления 31.02.2026")
    result = validate(proposal(consignee=FieldProposal(status="not_present"),
                               document_date=FieldProposal(raw_value="31.02.2026", status="recognized", evidence_ids=["v0:r1"])), data)
    assert result.fields["consignee"].status == "unreadable"
    assert result.fields["document_date"].normalized_value is None
    own_blank_label = regions("Форма ТОРГ-12", "Грузополучатель:")
    complete = validate(proposal(consignee=FieldProposal(status="not_present", evidence_ids=["v0:r1"])), own_blank_label, full_page_visible=True)
    assert complete.fields["consignee"].status == "not_present"


def test_bbox_is_reconstructed_in_top_left_ocr_space():
    page = {"frame_id": "f", "regions": [{"text": "label", "boundingBoxNormalizedBottomLeft":
                                          {"x": 0.1, "y": 0.6, "width": 0.3, "height": 0.2}}]}
    evidence = _regions([page])["v0:r0"]
    assert evidence.bbox == pytest.approx([0.1, 0.2, 0.3, 0.2])


def test_strict_type_and_positive_limits():
    with pytest.raises(ValidationError):
        ExtractionDecision(document_type="invoice")
    with pytest.raises(ValueError):
        AgentConfig(max_requests=0)


def _assert_native_json(info):
    assert info.output_tools == []
    assert info.model_request_parameters.output_mode == "native"
    output = info.model_request_parameters.output_object
    assert output is not None
    assert output.strict is False
    assert "document_type" in output.json_schema["properties"]


def _model_for(decision, counter, *, fail=False, use_tool=False):
    def respond(messages, info):
        counter.append(1)
        if fail:
            raise RuntimeError("provider body might contain private-key-marker")
        if use_tool and len(counter) % 2 == 1:
            return ModelResponse(parts=[ToolCallPart("read_ocr", {"region_ids": ["v0:r0"]})],
                                 usage=RequestUsage(input_tokens=15, output_tokens=5))
        _assert_native_json(info)
        return ModelResponse(parts=[TextPart(decision.model_dump_json())],
                             usage=RequestUsage(input_tokens=50, output_tokens=20))
    return FunctionModel(respond, model_name="unit-test-model")


def test_structured_agent_cache_and_shared_usage(tmp_path):
    calls = []
    config = AgentConfig(max_requests=2)
    model = _model_for(ExtractionDecision(document_type="unknown", observation="CMR title"), calls)
    pages = [{"frame_id": "f", "regions": [{"text": "CMR", "bbox": [0, 0, 1, 0.1]}]}]
    kwargs = dict(source_id="source", page_id="p", model=model, config=config,
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    first = asyncio.run(analyze_page(pages, **kwargs))
    second = asyncio.run(analyze_page(pages, **kwargs))
    assert first == second
    assert len(calls) == 1
    assert config.usage.requests == 1
    assert config.usage.attempted_requests == 1
    assert config.usage.input_tokens == 50
    assert config.usage.model_dump()["output_tokens"] == 20
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert {row["event"] for row in trace} >= {"ocr_observation", "model_decision", "validation", "cache_hit"}
    assert trace[-1]["usage_delta"]["requests"] == 0


def test_model_tool_footprints_and_per_page_tool_budget(tmp_path):
    calls = []
    config = AgentConfig(max_requests=4)
    model = _model_for(ExtractionDecision(document_type="unknown"), calls, use_tool=True)
    pages = [{"frame_id": "f", "regions": [{"text": "unknown"}]}]
    for page in ("p1", "p2"):
        asyncio.run(analyze_page(pages, source_id="s", page_id=page, config=config, model=model,
                                 cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))
    assert config.usage.requests == 4
    assert config.usage.tool_calls == 2
    assert "tool_read_ocr" in (tmp_path / "trace.jsonl").read_text()


def test_failure_is_cached_without_secret_or_automatic_retry(tmp_path):
    calls = []
    config = AgentConfig(api_key="private-key-marker")
    model = _model_for(ExtractionDecision(document_type="unknown"), calls, fail=True)
    kwargs = dict(source_id="s", page_id="p", config=config, model=model,
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    pages = [{"regions": [{"text": "document"}]}]
    for _ in range(2):
        with pytest.raises(AnalysisError):
            asyncio.run(analyze_page(pages, **kwargs))
    assert len(calls) == 1
    artifacts = (tmp_path / "trace.jsonl").read_text() + "".join(p.read_text() for p in (tmp_path / "cache").glob("*.json"))
    assert "private-key-marker" not in artifacts


def test_empty_ocr_and_exhausted_budget_make_no_model_request(tmp_path):
    calls = []
    model = _model_for(ExtractionDecision(document_type="unknown"), calls)
    config = AgentConfig(max_requests=1, max_tokens=1)
    kwargs = dict(source_id="s", config=config, model=model,
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    result = asyncio.run(analyze_page([], page_id="empty", **kwargs))
    assert result.document_type == "unknown"
    config.usage.input_tokens = 1
    with pytest.raises(AnalysisError):
        asyncio.run(analyze_page([{"regions": [{"text": "document"}]}], page_id="overbudget", **kwargs))
    assert not calls


def test_compact_input_deduplicates_views_and_bounded_coverage():
    pages = [{"frame_id": "f1", "regions": [{"text": "Счет-фактура № 123"}, {"text": "Поставщик ООО Север"}]},
             {"frame_id": "f2", "regions": [{"text": "Счет-фактура № 123"}, {"text": "Поставщик ООО Юг"}]}]
    lookup = _regions(pages)
    compact = _model_input(lookup, 40000, True)
    assert [r["id"] for r in compact["regions"]] == ["v0:r0", "v0:r1", "v1:r1"]
    assert "frame_id" not in compact["regions"][0]
    large = _regions([{"regions": [{"text": "Поставщик ООО Север"}] +
                                  [{"text": f"table row {i} " + "a" * 500} for i in range(50)]}])
    bounded = _model_input(large, 1200, True)
    assert bounded["ocr_coverage"].startswith("bounded")
    assert not bounded["full_page_visible"]
    assert bounded["regions"][0]["text"] == "Поставщик ООО Север"


def test_failed_request_uses_shared_budget_and_fatal_errors_stop_batch(tmp_path):
    calls = []
    def fatal(messages, info):
        calls.append(1)
        raise ModelHTTPError(status_code=402, model_name="test", body={"secret": "private-marker"})
    config = AgentConfig(max_requests=1)
    model = FunctionModel(fatal)
    kwargs = dict(source_id="s", config=config, model=model, cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    pages = [{"regions": [{"text": "document"}]}]
    with pytest.raises(FatalProviderError):
        asyncio.run(analyze_page(pages, page_id="p1", **kwargs))
    with pytest.raises(FatalProviderError):
        asyncio.run(analyze_page(pages, page_id="p1", **kwargs))
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(analyze_page(pages, page_id="p2", **kwargs))
    assert len(calls) == 1
    assert config.usage.requests == 1
    assert config.usage.attempted_requests == 1
    assert "private-marker" not in (tmp_path / "trace.jsonl").read_text()


def test_routerai_actual_cost_is_kept_as_rubles(tmp_path):
    def respond(messages, info):
        _assert_native_json(info)
        return ModelResponse(parts=[TextPart('{"document_type": "unknown"}')],
                             usage=RequestUsage(input_tokens=30, output_tokens=15),
                             provider_details={"routerai_cost_rub": 0.0123})
    config = AgentConfig()
    asyncio.run(analyze_page([{"regions": [{"text": "CMR"}]}], source_id="s", page_id="p",
                             config=config, model=FunctionModel(respond),
                             cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))
    usage = config.usage.model_dump()
    assert usage["actual_cost_rub"] == 0.0123
    assert usage["estimated_cost_rub"] == 0
    assert usage["provider_cost_currency"] == "RUB"
    assert usage["costed_requests"] == 1


def test_same_page_consignor_reference_requires_resolved_name():
    lookup = regions("Универсальный передаточный документ", "Продавец ООО Север", "Грузоотправитель он же")
    unresolved = validate(proposal("upd", consignor=FieldProposal(raw_value="он же", status="recognized", evidence_ids=["v0:r2"])), lookup)
    assert unresolved.fields["consignor"].status == "unreadable"
    resolved = validate(proposal("upd", consignor=FieldProposal(raw_value="ООО Север", status="recognized",
                                                                 evidence_ids=["v0:r1", "v0:r2"], reason="Explicit seller reference resolved")), lookup)
    assert resolved.fields["consignor"].normalized_value == "ООО Север"


def test_russian_date_words_use_shared_normalizer():
    lookup = regions("Универсальный передаточный документ", "Счет-фактура № 77 от 3 июня 2026")
    result = validate(proposal("upd", invoice_date=FieldProposal(raw_value="3 июня 2026", status="recognized",
                                                                 evidence_ids=["v0:r1"])), lookup)
    assert result.fields["invoice_date"].normalized_value == "2026-06-03"


@pytest.mark.parametrize("content", ['not JSON', '{"document_type":"invoice"}', '{"document_type":"unknown",'])
def test_invalid_content_json_spends_once_is_cached_and_keeps_cost(tmp_path, content):
    calls = []
    def respond(messages, info):
        calls.append(1)
        _assert_native_json(info)
        return ModelResponse(parts=[TextPart(content)], usage=RequestUsage(input_tokens=90, output_tokens=12),
                             provider_details={"routerai_cost_rub": 0.04})
    config = AgentConfig(max_requests=2)
    kwargs = dict(source_id="s", page_id="p", config=config, model=FunctionModel(respond),
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    pages = [{"regions": [{"text": "CMR"}]}]
    for _ in range(2):
        with pytest.raises(AnalysisError):
            asyncio.run(analyze_page(pages, **kwargs))
    assert len(calls) == 1
    assert config.usage.attempted_requests == config.usage.requests == 1
    assert config.usage.input_tokens == 90
    assert config.usage.output_tokens == 12
    assert config.usage.actual_cost_rub == 0.04
    assert config.usage.costed_requests == 1


def test_json_protocol_preserves_optional_tool_steps(tmp_path):
    calls = []
    target = proposal("upd", invoice_number=FieldProposal(raw_value="123", status="recognized", evidence_ids=["v0:r1"]))
    model = _model_for(target, calls, use_tool=True)
    config = AgentConfig(max_requests=2)
    pages = [{"regions": [{"text": "Универсальный передаточный документ"}, {"text": "Счет-фактура № 123"}]}]
    result = asyncio.run(analyze_page(pages, source_id="s", page_id="p", config=config, model=model,
                                      cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))
    assert result.document_type == "upd"
    assert result.fields["invoice_number"].normalized_value == "123"
    assert len(calls) == 2
    assert config.usage.requests == 2


def test_json_request_preflight_prevents_an_oversized_request_before_payment(tmp_path):
    calls = []
    model = _model_for(ExtractionDecision(document_type="unknown"), calls)
    config = AgentConfig(max_requests=5, max_tokens=60)
    kwargs = dict(source_id="s", config=config, model=model,
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    pages = [{"regions": [{"text": "CMR"}]}]
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(analyze_page(pages, page_id="p1", **kwargs))
    assert config.usage.requests == config.usage.attempted_requests == 0
    assert config.usage.input_tokens + config.usage.output_tokens == 0
    assert config.usage.actual_cost_rub == 0
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(analyze_page(pages, page_id="p2", **kwargs))
    assert not calls


def test_provider_reports_excess_usage_after_a_sufficient_preflight(tmp_path):
    calls = []
    def respond(messages, info):
        calls.append(1)
        _assert_native_json(info)
        # Simulate a provider count above the local preflight estimate. The
        # response was already paid and must remain in the usage ledger.
        return ModelResponse(parts=[TextPart('{"document_type":"unknown"}')],
                             usage=RequestUsage(input_tokens=25000, output_tokens=75),
                             provider_details={"routerai_cost_rub": .31})
    config = AgentConfig(max_requests=5, max_tokens=20000)
    kwargs = dict(source_id="s", config=config, model=FunctionModel(respond),
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl")
    pages = [{"regions": [{"text": "CMR"}]}]
    with pytest.raises(AnalysisError):
        asyncio.run(analyze_page(pages, page_id="p1", **kwargs))
    assert config.usage.requests == config.usage.attempted_requests == 1
    assert config.usage.input_tokens + config.usage.output_tokens == 25075
    assert config.usage.actual_cost_rub == .31
    assert config.usage.costed_requests == 1
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(analyze_page(pages, page_id="p2", **kwargs))
    assert len(calls) == 1


def test_literal_completion_never_stitches_partial_values_across_frames():
    lookup = _regions([
        {"frame_id": "a", "regions": [{"text": "ул Северная", "bbox": [.1, .2, .4, .03]}]},
        {"frame_id": "b", "regions": [{"text": "дом 12", "bbox": [.1, .24, .4, .03]}]},
    ])
    seeds = [lookup["v0:r0"], lookup["v1:r0"]]
    _, grounded = _complete_literal_evidence("ул Северная дом 12", seeds, lookup)
    assert not grounded
    same_frame = _regions([{"frame_id": "a", "regions": [
        {"text": "ул Северная", "bbox": [.1, .2, .4, .03]},
        {"text": "дом 12", "bbox": [.1, .24, .4, .03]},
    ]}])
    completed, grounded = _complete_literal_evidence("ул Северная дом 12", [same_frame["v0:r0"]], same_frame)
    assert grounded
    assert {e.region_index for e in completed} == {0, 1}
    assert {e.frame_id for e in completed} == {"a"}


def test_field_grounding_does_not_join_name_fragments_from_different_frames():
    lookup = _regions([
        {"frame_id": "a", "regions": [{"text": "Форма ТОРГ-12"}, {"text": "ООО Север"}]},
        {"frame_id": "b", "regions": [{"text": "Снабжение"}]},
    ])
    result = validate(proposal(supplier=FieldProposal(raw_value="ООО Север Снабжение", status="recognized",
                                                       evidence_ids=["v0:r1", "v1:r0"])), lookup)
    assert result.fields["supplier"].status == "unreadable"
    assert result.fields["supplier"].normalized_value is None


def test_consignor_reference_and_resolved_entity_are_read_from_same_frame():
    lookup = _regions([
        {"frame_id": "a", "regions": [{"text": "Универсальный передаточный документ"},
                                          {"text": "Грузоотправитель"}, {"text": "он же"}]},
        {"frame_id": "b", "regions": [{"text": "Продавец ООО Север"}]},
    ])
    result = validate(proposal("upd", consignor=FieldProposal(raw_value="ООО Север", status="recognized",
                                                               evidence_ids=["v0:r2", "v1:r0"])), lookup)
    assert result.fields["consignor"].status == "unreadable"
    assert result.fields["consignor"].normalized_value is None
    same = _regions([{"frame_id": "a", "regions": [{"text": "Универсальный передаточный документ"},
                                                       {"text": "он же"}, {"text": "Продавец ООО Север"}]}])
    result = validate(proposal("upd", consignor=FieldProposal(raw_value="ООО Север", status="recognized",
                                                               evidence_ids=["v0:r1", "v0:r2"])), same)
    assert result.fields["consignor"].status == "recognized"
    assert {e.frame_id for e in result.fields["consignor"].evidence} == {"a"}


def test_not_present_requires_evidence_of_the_specific_blank_field():
    data = regions("Форма ТОРГ-12", "Дата составления 31.02.2026")
    result = validate(proposal(consignee=FieldProposal(status="not_present", evidence_ids=["v0:r0"])), data, full_page_visible=True)
    assert result.fields["consignee"].status == "unreadable"


def test_transport_blank_number_uses_its_own_header_and_null_raw_value():
    data = regions("Транспортная накладная", "№", "Заказ (заявка) № 123", "Сопроводительная ТТН № 456")
    own = validate(proposal("trn", document_number=FieldProposal(status="not_present", evidence_ids=["v0:r0", "v0:r1"])), data, full_page_visible=True)
    assert own.fields["document_number"].status == "not_present"
    for ids, raw in [(["v0:r2"], None), (["v0:r0", "v0:r3"], None), (["v0:r0", "v0:r1"], "123")]:
        foreign = validate(proposal("trn", document_number=FieldProposal(status="not_present", raw_value=raw, evidence_ids=ids)), data, full_page_visible=True)
        assert foreign.fields["document_number"].status == "unreadable"
