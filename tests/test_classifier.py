import asyncio
import json

import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError, _regions
import sever.classifier as classifier
from sever.classifier import (MAX_BATCH_SIZE, MAX_PAGE_CHARS, _header_regions, classify_pages,
                              _validated_result, PageDecision, revalidate_cached_classifications)
from sever.schema import FIELDS_BY_TYPE


def page(page_id, title, *, more=None):
    items = [title] + list(more or [])
    return {"page_id": page_id, "ocr_pages": [{"frame_id": page_id + "-frame", "regions": [
        {"text": text, "region_index": index, "bbox": [0.1, 0.02 + .04 * index, 0.8, .03]}
        for index, text in enumerate(items)]}]}


def batch_model(calls, decide=None, *, failure=None):
    def respond(messages, info):
        assert info.output_tools == []
        assert info.function_tools == []
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_request_parameters.output_object.strict is False
        prompt = [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)][-1]
        data = json.loads(prompt.split("\n", 1)[1])
        calls.append(data)
        if failure:
            raise failure
        decisions = []
        for item in data["pages"]:
            if decide:
                decisions.append(decide(item, data))
            else:
                decisions.append({"page_id": item["page_id"], "document_type": "unknown",
                                  "classification_evidence_ids": [item["regions"][0]["id"]] if item["regions"] else [],
                                  "observation": "Unsupported visible form"})
        return ModelResponse(parts=[TextPart(json.dumps({"pages": decisions}, ensure_ascii=False))],
                             usage=RequestUsage(input_tokens=100, output_tokens=60),
                             provider_details={"routerai_cost_rub": .03})
    return FunctionModel(respond, model_name="unit-batch-classifier")


def run(pages, tmp_path, config, model):
    return asyncio.run(classify_pages(pages, source_id="source", config=config, model=model,
                                     cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl"))


def test_all_leaves_are_classified_in_chunks_of_eight_and_cached(tmp_path):
    calls = []
    config = AgentConfig(max_requests=2)
    model = batch_model(calls)
    pages = [page(f"p{i}", "Сертификат качества") for i in range(9)]
    first = run(pages, tmp_path, config, model)
    assert list(first) == [p["page_id"] for p in pages]
    assert [len(c["pages"]) for c in calls] == [8, 1]
    assert all(r.document_type == "unknown" and r.fields == {} for r in first.values())
    assert config.usage.requests == config.usage.attempted_requests == 2
    assert config.usage.actual_cost_rub == pytest.approx(.06)
    assert config.usage.input_tokens == 200
    assert config.usage.output_tokens == 120
    second = run(pages, tmp_path, config, model)
    assert first == second
    assert len(calls) == 2
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "classifier_cache_hit" for row in trace) == 2
    assert trace[-1]["usage_delta"]["requests"] == 0


def test_supported_classes_have_ocr_evidence_and_pending_field_placeholders(tmp_path):
    expected = {"upd": "Универсальный передаточный документ", "torg12": "ТОРГ-12 Форма по ОКУД 0330212",
                "ttn": "ТОВАРНО-ТРАНСПОРТНАЯ НАКЛАДНАЯ", "trn": "Транспортная накладная"}
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": item["page_id"], "type_confidence": .9,
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([page(kind, title) for kind, title in expected.items()], tmp_path, AgentConfig(), batch_model([], decide))
    for kind, value in result.items():
        assert value.document_type == kind
        assert set(value.fields) == set(FIELDS_BY_TYPE[kind])
        assert all(f.status == "unreadable" and f.normalized_value is None for f in value.fields.values())
        assert not value.errors
        assert value.classification_evidence[0].frame_id == kind + "-frame"
        assert value.classification_evidence[0].text == expected[kind]


@pytest.mark.parametrize(("claimed", "title"), [("upd", "Счет-фактура № 123"),
                                                  ("ttn", "CMR Международная товарно-транспортная накладная"),
                                                  ("torg12", "ТН-2 ТОРГ-12 товарная накладная")])
def test_foreign_forms_and_invoice_alone_are_unknown(tmp_path, claimed, title):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": claimed,
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([page("p", title)], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "unknown"
    assert result.fields == {}
    assert result.errors


def test_cross_page_evidence_is_rejected(tmp_path):
    def decide(item, data):
        if item["page_id"] == "certificate":
            return {"page_id": "certificate", "document_type": "upd",
                    "classification_evidence_ids": [data["pages"][1]["regions"][0]["id"]]}
        return {"page_id": "actual-upd", "document_type": "upd",
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([page("certificate", "Сертификат"), page("actual-upd", "Универсальный передаточный документ")],
                 tmp_path, AgentConfig(), batch_model([], decide))
    assert result["certificate"].document_type == "unknown"
    assert result["certificate"].classification_evidence == []
    assert "foreign" in result["certificate"].errors[0]
    assert result["actual-upd"].document_type == "upd"


def test_conflicting_own_titles_are_unknown(tmp_path):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": [r["id"] for r in item["regions"]]}
    result = run([page("p", "Товарно-транспортная накладная", more=["Транспортная накладная"])],
                 tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "unknown"
    assert "conflicting" in result.errors[0]


def test_header_input_has_hard_page_character_bound_and_skips_goods():
    item = page("p", "Транспортная накладная")
    item["ocr_pages"][0]["regions"] += [{"text": "goods " + str(i) + "x" * 100, "bbox": [.1, .6, .8, .03]}
                                          for i in range(100)]
    selected, lookup, bounded = _header_regions(_regions(item["ocr_pages"]), page_index=3)
    assert len(json.dumps(selected, ensure_ascii=False, sort_keys=True)) <= MAX_PAGE_CHARS
    assert bounded
    assert len(selected) == 1
    assert list(lookup) == ["p3:v0:r0"]
    assert selected[0]["text"] == "Транспортная накладная"


def test_native_schema_failure_is_counted_and_cached_without_retry(tmp_path):
    calls = []
    def invalid(messages, info):
        calls.append(1)
        return ModelResponse(parts=[TextPart('{"pages":[{"page_id":"p","document_type":"invoice"}]}')],
                             usage=RequestUsage(input_tokens=50, output_tokens=20),
                             provider_details={"routerai_cost_rub": .02})
    config = AgentConfig()
    model = FunctionModel(invalid, model_name="invalid-test-batch")
    for _ in range(2):
        with pytest.raises(AnalysisError):
            run([page("p", "Счет-фактура")], tmp_path, config, model)
    assert len(calls) == 1
    assert config.usage.requests == config.usage.attempted_requests == 1
    assert config.usage.input_tokens == 50
    assert config.usage.actual_cost_rub == .02


def test_missing_or_duplicate_output_pages_stop_after_one_request(tmp_path):
    calls = []
    def decide(item, data):
        return {"page_id": "p1", "document_type": "unknown"}
    with pytest.raises(AnalysisError):
        run([page("p1", "Certificate"), page("p2", "Certificate")], tmp_path, AgentConfig(), batch_model(calls, decide))
    assert len(calls) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


@pytest.mark.parametrize("status", [401, 402, 403])
def test_fatal_provider_failure_stops_all_later_chunks_and_does_not_leak_secret(tmp_path, status):
    calls = []
    config = AgentConfig(api_key="private-test-marker")
    model = batch_model(calls, failure=ModelHTTPError(status_code=status, model_name="test", body={"secret": "private-test-marker"}))
    pages = [page(f"p{i}", "Certificate") for i in range(9)]
    for _ in range(2):
        with pytest.raises(FatalProviderError):
            run(pages, tmp_path, config, model)
    assert len(calls) == 1
    assert config.usage.requests == config.usage.attempted_requests == 1
    artifacts = (tmp_path / "trace.jsonl").read_text() + "".join(p.read_text() for p in (tmp_path / "cache").glob("*.json"))
    assert "private-test-marker" not in artifacts


def test_shared_budget_stops_next_chunk_and_preflight_stops_before_any_request(tmp_path):
    calls = []
    config = AgentConfig(max_requests=1)
    with pytest.raises(BatchBudgetExceeded):
        run([page(f"p{i}", "Certificate") for i in range(9)], tmp_path, config, batch_model(calls))
    assert len(calls) == 1
    assert config.usage.requests == config.usage.attempted_requests == 1
    calls = []
    with pytest.raises(BatchBudgetExceeded):
        run([page("p", "Certificate")], tmp_path / "preflight", AgentConfig(max_tokens=1), batch_model(calls))
    assert not calls


def test_changed_ocr_invalidates_only_changed_chunk(tmp_path):
    calls = []
    config = AgentConfig(max_requests=3)
    model = batch_model(calls)
    pages = [page(f"p{i}", "Certificate") for i in range(9)]
    run(pages, tmp_path, config, model)
    pages[-1]["ocr_pages"][0]["regions"][0]["text"] = "CMR"
    run(pages, tmp_path, config, model)
    assert len(calls) == 3
    assert len(calls[-1]["pages"]) == 1


def test_empty_ocr_is_still_included_in_ml_batch_and_duplicate_input_ids_fail_locally(tmp_path):
    calls = []
    config = AgentConfig()
    model = batch_model(calls)
    result = run([{"page_id": "empty", "ocr_pages": []}], tmp_path, config, model)
    assert result["empty"].document_type == "unknown"
    assert calls[0]["pages"][0]["regions"] == []
    assert run([], tmp_path, config, model) == {}
    with pytest.raises(ValueError):
        run([page("same", "CMR"), page("same", "CMR")], tmp_path, config, model)
    assert len(calls) == 1


def test_ttn_reverse_form_and_trn_header_ignore_accompanying_ttn_reference(tmp_path):
    pages = [page("ttn-back", "Оборотная сторона формы № 1-Т"),
             page("trn", "Транспортная накладная", more=["Сопроводительная ТТН № 456"])]
    def decide(item, data):
        title = next(region for region in item["regions"] if region["id"].endswith(":r0"))
        return {"page_id": item["page_id"], "document_type": "ttn" if item["page_id"] == "ttn-back" else "trn",
                "classification_evidence_ids": [title["id"]]}
    result = run(pages, tmp_path, AgentConfig(), batch_model([], decide))
    assert result["ttn-back"].document_type == "ttn"
    assert result["trn"].document_type == "trn"


def test_form_title_has_priority_over_earlier_footer_role_regions():
    item = page("p", "Грузополучатель " + "x" * 100)
    item["ocr_pages"][0]["regions"] = [{"text": "Грузополучатель " + str(i) + "x" * 100,
                                          "bbox": [.1, .7, .8, .03]} for i in range(40)]
    item["ocr_pages"][0]["regions"].append({"text": "ТН-2 Товарная накладная", "bbox": [.1, .01, .8, .03]})
    selected, _, _ = _header_regions(_regions(item["ocr_pages"]), page_index=0)
    assert selected[0]["text"] == "ТН-2 Товарная накладная"


@pytest.mark.parametrize("title", ["Оборотная сторона формы Nº 1-T", "Форма № 1-T", "Типовая межотраслевая форма № 1-Т"])
def test_latin_and_cyrillic_ttn_form_codes_are_prioritized_and_supported(tmp_path, title):
    item = page("p", "Грузополучатель " + "x" * 200)
    item["ocr_pages"][0]["regions"] += [{"text": "Грузополучатель " + "x" * 200,
                                         "bbox": [.1, .7, .8, .03]} for _ in range(30)]
    item["ocr_pages"][0]["regions"].append({"text": title, "bbox": [.1, .85, .8, .03]})
    def decide(item, data):
        assert item["regions"][0]["text"] == title
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([item], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "ttn"
    assert result.classification_evidence[0].text == title


def test_missing_ttn_proof_is_restored_only_from_own_same_page_form_code(tmp_path):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": ["p9:v0:r0"]}
    result = run([page("p", "1АОН", more=["Оборотная сторона формы Nº 1-T"])],
                 tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "ttn"
    assert result.classification_evidence[0].text == "Оборотная сторона формы Nº 1-T"
    assert result.classification_evidence[0].frame_id == "p-frame"
    assert not result.errors
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    repaired = next(row for row in trace if row["event"] == "classifier_evidence_restored")
    assert repaired["original_evidence_ids"] == ["p9:v0:r0"]
    assert repaired["restored_evidence_id"] == "p0:v0:r1"


@pytest.mark.parametrize("title", ["ГОВАРНО-ТРАНСПОРТНАЯ НАКЛАДНА", "Сопроводительная форма № 1-T"])
def test_broken_heading_and_supporting_form_reference_cannot_restore_a_type(tmp_path, title):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": ["missing"]}
    result = run([page("p", title)], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "unknown"
    assert result.fields == {}
    assert "classifier_evidence_restored" not in (tmp_path / "trace.jsonl").read_text()


def test_citing_a_supporting_form_code_does_not_identify_the_own_form(tmp_path):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([page("p", "Сопроводительная форма № 1-T")], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "unknown"


@pytest.mark.parametrize(("title", "y", "claimed", "expected"), [
    ("ерсальный передаточный документ Nº 0012 от 3 мая 2026 г.", .04, "upd", "upd"),
    ("зерсальный передаточный документ Nº 12 от 3 мая 2026 г.", .04, "upd", "upd"),
    ("ерсальный передаточный документ Nº 12 от 3 мая 2026 г.", .15, "upd", "upd"),
    ("ерсальный передаточный документ Nº 12 от 3 мая 2026 г.", .21, "upd", "unknown"),
    ("Приложен ерсальный передаточный документ Nº 12 от 3 мая 2026 г.", .04, "upd", "unknown"),
    ("передаточный документ Nº 12 от 3 мая 2026 г.", .04, "upd", "unknown"),
    ("ерсальный передаточный документ Nº 12 от 3 мая 2026 г.", .04, "unknown", "unknown"),
])
def test_clipped_upd_prefix_only_supports_an_existing_upd_claim_in_the_own_top_header(tmp_path, title, y, claimed, expected):
    item = page("p", title)
    item["ocr_pages"][0]["regions"][0]["bbox"][1] = y
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": claimed,
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([item], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == expected
    assert result.classification_evidence[0].text == title
    if expected == "upd":
        assert any("clipped prefix" in note for note in result.notes)


def test_an_unknown_ml_decision_is_not_promoted_by_the_form_code(tmp_path):
    result = run([page("p", "Оборотная сторона формы Nº 1-T")], tmp_path, AgentConfig(), batch_model([]))["p"]
    assert result.document_type == "unknown"
    assert result.fields == {}


@pytest.mark.parametrize("title", ["Международная товарно-транспортная накладная",
                                   "МЕЖДУНАРОДНАЯ ТРАНСПОРТНАЯ НАКЛАДНАЯ"])
def test_international_consignment_title_without_cmr_literal_is_unknown(tmp_path, title):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": [item["regions"][0]["id"]]}
    result = run([page("p", title, more=["Оборотная сторона формы № 1-Т"])],
                 tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "unknown"
    assert not result.fields


def test_new_cache_preserves_raw_decisions_separately_from_validated_results(tmp_path):
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "upd", "classification_evidence_ids": ["invalid"]}
    run([page("p", "Сертификат")], tmp_path, AgentConfig(), batch_model([], decide))
    saved = json.loads(next((tmp_path / "cache").glob("classifier-*.json")).read_text())
    assert saved["decisions"][0]["document_type"] == "upd"
    assert saved["results"]["p"]["document_type"] == "unknown"
    assert saved["key_context"]["source_id"] == "source"
    assert saved["validation_version"] == classifier.CLASSIFIER_VALIDATION_VERSION
    assert saved["full_ocr_sha256"]


def test_legacy_trace_revalidates_old_header_selection_and_raw_type_without_payment(tmp_path, monkeypatch):
    calls = []
    config = AgentConfig(max_requests=1)
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn", "classification_evidence_ids": ["p0:v0:r0"]}
    model = batch_model(calls, decide)
    item = page("p", "1АОН")
    item["ocr_pages"][0]["regions"].append({"text": "Оборотная сторона формы Nº 1-T", "bbox": [.1, .8, .8, .03]})
    import re
    with monkeypatch.context() as old:
        old.setattr(classifier, "CLASSIFIER_VERSION", "sever-batch-classifier-v1-native-json")
        old.setattr(classifier, "_TTN_OWN_CODE", re.compile(r"never-match-own-code"))
        old.setattr(classifier, "_ANCHORS", re.compile(r"накладная"))
        old.setattr(classifier, "_FORM_ANCHORS", re.compile(r"накладная"))
        first = run([item], tmp_path, config, model)
    assert first["p"].document_type == "unknown"
    assert all(r["text"] != "Оборотная сторона формы Nº 1-T" for r in calls[0]["pages"][0]["regions"])
    cache_path = next((tmp_path / "cache").glob("classifier-*.json"))
    saved = json.loads(cache_path.read_text())
    # Simulate the legacy payload, which had only final results; raw type/proof
    # remain in the original paid validation trace.
    for field in ("decisions", "key_context", "full_ocr_sha256", "validation_version"):
        saved.pop(field)
    cache_path.write_text(json.dumps(saved, ensure_ascii=False))
    before = config.usage.model_dump()
    free = revalidate_cached_classifications([item], source_id="source", config=config,
                                             cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl",
                                             model_name=model.model_name)
    assert free["p"].document_type == "ttn"
    assert free["p"].classification_evidence[0].text == "Оборотная сторона формы Nº 1-T"
    assert config.usage.model_dump() == before
    second = run([item], tmp_path, config, model)
    assert second == free
    assert len(calls) == 1
    updated = json.loads(cache_path.read_text())
    assert updated["usage_delta"] == saved["usage_delta"]
    assert updated["decisions"][0]["document_type"] == "ttn"
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert next(row for row in trace if row["event"] == "classifier_cache_revalidated")["usage_delta"]["actual_cost_rub"] == 0


@pytest.mark.parametrize("change", ["text", "frame_id", "bbox"])
def test_free_revalidation_cannot_reuse_a_different_ocr_source(tmp_path, change):
    item = page("p", "Сертификат")
    config = AgentConfig()
    model = batch_model([])
    run([item], tmp_path, config, model)
    if change == "frame_id":
        item["ocr_pages"][0][change] = "a-different-frame"
    elif change == "text":
        item["ocr_pages"][0]["regions"][0][change] = "Декларация"
    else:
        item["ocr_pages"][0]["regions"][0][change] = [.1, .03, .8, .03]
    before = config.usage.model_dump()
    assert revalidate_cached_classifications([item], source_id="source", config=config,
                                             cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl",
                                             model_name=model.model_name) == {}
    assert config.usage.model_dump() == before


def overlay_page():
    item = page("p", "Универсальный передаточный документ № 0007 от 3 мая 2026 г.",
                more=["ТОВАРНО-ТРАНСПОРТНАЯ НАКЛАДНАЯ"])
    item["ocr_pages"][0]["regions"][0]["bbox"] = [.1166, .1423, .2203, .0816]
    item["ocr_pages"][0]["regions"][1]["bbox"] = [.5331, -.0045, .0594, .2389]
    return item


def test_vertical_overlay_heading_is_not_sent_as_the_main_leaf_form_anchor():
    item = overlay_page()
    selected, lookup, bounded = _header_regions(_regions(item["ocr_pages"]), page_index=0)
    assert [r["id"] for r in selected] == ["p0:v0:r0"]
    assert selected[0]["text"] == item["ocr_pages"][0]["regions"][0]["text"]
    assert set(lookup) == {"p0:v0:r0"}
    assert bounded


def test_old_cited_vertical_overlay_is_excluded_from_joined_and_individual_form_proof(tmp_path):
    item = overlay_page()
    original = {f"p0:{i}": e for i, e in _regions(item["ocr_pages"]).items()}
    decision = PageDecision(page_id="p", document_type="upd",
                            classification_evidence_ids=["p0:v0:r0", "p0:v0:r1"])
    result = _validated_result(decision, original, source_id="source", source_lookup=original,
                               trace_path=tmp_path / "trace.jsonl")
    assert result.document_type == "upd"
    assert [e.text for e in result.classification_evidence] == [item["ocr_pages"][0]["regions"][0]["text"]]
    assert not result.errors
    ignored = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert ignored["event"] == "classifier_vertical_evidence_ignored"
    assert ignored["evidence"]["p0:v0:r1"]["text"] == item["ocr_pages"][0]["regions"][1]["text"]


@pytest.mark.parametrize("title", ["ТОВАРНО-ТРАНСПОРТНАЯ НАКЛАДНАЯ", "Оборотная сторона формы № 1-Т"])
def test_a_vertical_heading_cannot_supply_own_type_or_restored_own_form_code(title):
    item = page("p", title)
    item["ocr_pages"][0]["regions"][0]["bbox"] = [.5, 0., .05, .24]
    original = {f"p0:{i}": e for i, e in _regions(item["ocr_pages"]).items()}
    decision = PageDecision(page_id="p", document_type="ttn", classification_evidence_ids=list(original))
    result = _validated_result(decision, original, source_id="source", source_lookup=original)
    assert result.document_type == "unknown"
    assert result.classification_evidence == []
    assert result.fields == {}


def test_vertical_single_words_are_retained_for_a_split_main_upd_title(tmp_path):
    item = page("p", "Универсальный", more=["передаточный", "документ"])
    for region in item["ocr_pages"][0]["regions"]:
        region["bbox"] = [.1, .05, .02, .05]
    selected, _, _ = _header_regions(_regions(item["ocr_pages"]), page_index=0)
    assert {r["text"] for r in selected} == {"Универсальный", "передаточный", "документ"}
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "upd",
                "classification_evidence_ids": [r["id"] for r in item["regions"]]}
    result = run([item], tmp_path, AgentConfig(), batch_model([], decide))["p"]
    assert result.document_type == "upd"


def test_horizontal_alternate_heading_is_not_deduplicated_against_an_excluded_vertical_view():
    item = page("p", "Товарно-транспортная накладная")
    item["ocr_pages"][0]["regions"][0]["bbox"] = [.5, 0., .05, .24]
    item["ocr_pages"].append({"frame_id": "p-alt-frame", "regions": [
        {"text": "Товарно-транспортная накладная", "bbox": [.1, .05, .5, .04]}]})
    selected, lookup, _ = _header_regions(_regions(item["ocr_pages"]), page_index=0)
    assert [r["id"] for r in selected] == ["p0:v1:r0"]
    assert lookup["p0:v1:r0"].frame_id == "p-alt-frame"


def test_paid_cached_vertical_type_is_rejected_freely_without_reusing_overlay_proof(tmp_path, monkeypatch):
    item = overlay_page()
    calls = []
    config = AgentConfig(max_requests=1)
    def decide(item, data):
        return {"page_id": item["page_id"], "document_type": "ttn",
                "classification_evidence_ids": ["p0:v0:r1"]}
    model = batch_model(calls, decide)
    with monkeypatch.context() as old:
        old.setattr(classifier, "_vertical_form_anchor", lambda evidence: False)
        first = run([item], tmp_path, config, model)
    assert first["p"].document_type == "ttn"
    before = config.usage.model_dump()
    second = run([item], tmp_path, config, model)
    assert second["p"].document_type == "unknown"
    assert second["p"].classification_evidence == []
    assert config.usage.model_dump() == before
    assert len(calls) == 1
