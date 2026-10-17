import asyncio
from copy import deepcopy
import json
from urllib.parse import parse_qs

import httpx
import pytest

from sever.inn import FNS_ORIGIN, FnsRegistry, extract_printed_inns, validate_inn, verify_document_inns
from sever.schema import FIELDS_BY_TYPE


def region(text, x, y, width=.25, *, source="full"):
    return {"text": text, "bbox": [x, y, width, .012], "processing_pass": source}


def upd(frame_id="source", *, seller="7707083893", recipient="7453297257"):
    return {"frame_id": frame_id, "timestamp_s": 1.2, "regions": [
        region("Продавец:", .10, .10, .08),
        region('ООО "Продавец"', .25, .10),
        region("Адрес:", .10, .12, .08),
        region("100001, Город, ул. Первая, д. 1", .25, .12),
        region("ИНН/КПП продавца:", .10, .14, .13),
        region(seller + "/770701001", .25, .14),
        region("Грузоотправитель и его адрес:", .10, .18, .20),
        region("он же", .32, .18, .05),
        region('ООО "Получатель"; ИНН/КПП ' + recipient + "/745301001", .10, .215, .40),
        region("Грузополучатель и его адрес:", .10, .23, .20),
        region("Покупатель:", .60, .10, .08),
        region('ООО "Другой покупатель", ИНН 7725382984', .60, .12, .35),
        # More repeated consignee OCR must never win over «он же» seller.
        region('ООО "Получатель"; ИНН/КПП ' + recipient + "/745301001", .10, .216, .40, source="header"),
    ]}


def document(kind="upd", *, frame_id=None):
    fields = {key: {"status": "unreadable", "raw_value": None, "normalized_value": None, "evidence": []}
              for key in FIELDS_BY_TYPE[kind]}
    for role, value in (("consignor", 'ООО "Продавец"'), ("consignee", 'ООО "Получатель"')):
        if role in fields and frame_id:
            fields[role] = {"status": "recognized", "raw_value": value, "normalized_value": value,
                            "evidence": [{"frame_id": frame_id, "text": value, "bbox": [.1, .1, .3, .03]}]}
    return {"source_id": "video", "page_id": "leaf", "document_type": kind, "fields": fields}


def record(inn="7707083893", *, name='ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "Продавец"', kind="ul"):
    return {"k": kind, "i": inn, "n": name, "c": 'ООО "Продавец"' if kind == "ul" else None,
            "o": "1027700132195", "r": "01.01.2000", "e": None, "v": None,
            "a": "registry-address-must-not-copy", "t": "session-token-must-not-cache", "cnt": "1"}


def service(tmp_path, rows=None, *, home="var CAPTCHA_REQUIRED = false", query=None, result=None, budget=10):
    calls = []
    def respond(request):
        calls.append(request)
        assert request.url.host == "egrul.nalog.ru"
        if request.url.path == "/index.html":
            return httpx.Response(200, text=home, headers={"Set-Cookie": "session-token-must-not-cache"})
        if request.method == "POST":
            assert request.url.path == "/"
            assert parse_qs(request.content.decode())["query"]
            assert request.headers["X-Requested-With"] == "XMLHttpRequest"
            return httpx.Response(200, json=query if query is not None else {"t": "search_token"})
        assert request.url.path == "/search-result/search_token"
        return httpx.Response(200, json=result if result is not None else {"rows": rows if rows is not None else [record()]})
    return FnsRegistry(tmp_path, transport=httpx.MockTransport(respond), max_unique_lookups=budget), calls


@pytest.mark.parametrize("value", ["7707083893", "7453297257", "7725382984", "500100732259"])
def test_literal_valid_format_and_checksum_are_not_registration(value):
    result = validate_inn(value)
    assert result["format_valid"] and result["checksum_valid"]
    assert result["registration_verified"] is False


@pytest.mark.parametrize("value", [None, "", "００００００００００", "0077083893", "74S3297257", "7453297257/745301001", "74532972571745301001"])
def test_format_validation_never_repairs_or_truncates_digits(value):
    result = validate_inn(value)
    assert not result["format_valid"] and result["checksum_valid"] is None


def test_wrong_control_digit_remains_literal():
    assert validate_inn("7707083894")["checksum_valid"] is False
    page = upd(seller="7707083894")
    reading = extract_printed_inns([page], document())["roles"]["consignor"]
    assert reading["printed_inn"] == "7707083894"
    assert reading["local_validation"]["checksum_valid"] is False


def test_on_zhe_resolves_same_frame_seller_not_more_repeated_consignee_or_buyer():
    page, original = upd(), document()
    snapshot = deepcopy(original)
    roles = extract_printed_inns([page], original)["roles"]
    assert roles["consignor"]["printed_inn"] == "7707083893"
    assert roles["consignor"]["resolved_role"] == "seller"
    assert roles["consignee"]["printed_inn"] == "7453297257"
    assert roles["consignee"]["printed_company_name"] == 'ООО "Получатель"'
    assert all(proof["frame_id"] == "source" for proof in roles["consignor"]["evidence"])
    assert any(proof["text"] == "он же" for proof in roles["consignor"]["evidence"])
    assert not any("Другой покупатель" in proof["text"] for proof in roles["consignee"]["evidence"])
    assert original == snapshot


def test_on_zhe_cannot_borrow_seller_from_neighbor_frame():
    partial = upd("partial")
    partial["regions"] = partial["regions"][6:]
    roles = extract_printed_inns([partial, upd("neighbor")], document(frame_id="partial"))["roles"]
    assert roles["consignor"]["printed_inn"] is None
    assert roles["consignor"]["source_frame_ids"] == ["partial"]


def test_field_frame_evidence_selects_exact_frame_from_unlimited_pool():
    views = [upd("wrong-" + str(i), seller="7725382984") for i in range(5)] + [upd("correct")]
    roles = extract_printed_inns(views, document(frame_id="correct"))["roles"]
    assert roles["consignor"]["printed_inn"] == "7707083893"
    assert roles["consignor"]["source_frame_ids"] == ["correct"]
    assert roles["consignor"]["frame_observations"][0]["timestamp_s"] == 1.2


def test_missing_field_frame_does_not_fall_back_to_first_view():
    reading = extract_printed_inns([upd()], document(frame_id="missing"))["roles"]["consignor"]
    assert reading["printed_status"] == "unavailable" and reading["printed_inn"] is None
    assert reading["source_frame_ids"] == []


def test_two_literal_inns_are_ambiguous_even_if_one_has_more_votes_or_bad_checksum(tmp_path):
    views = [upd("many-" + str(i)) for i in range(4)] + [upd("other", seller="7707083894")]
    registry, calls = service(tmp_path)
    roles = asyncio.run(verify_document_inns(views, document(), registry=registry))["roles"]
    assert roles["consignor"]["printed_status"] == "ambiguous"
    assert roles["consignor"]["candidates"] == ["7707083893", "7707083894"]
    assert roles["consignor"]["registry"]["status"] == "ambiguous"
    # The unambiguous recipient is still checked; sender ambiguity makes no query.
    assert registry.stats()["unique_lookups"] == 1
    assert parse_qs(calls[1].content.decode())["query"] == ["7453297257"]


def test_trn_customer_is_not_carrier():
    page = {"frame_id": "trn", "regions": [
        region("1. Грузоотправитель", .08, .12, .25),
        region('ООО "Отправитель", ИНН 7707083893', .08, .15, .32),
        region("1а. Заказчик услуг, связанных с перевозкой груза", .55, .12, .4),
        region('ООО "Заказчик", ИНН 7453297257', .55, .15, .35),
        region("2. Грузополучатель", .08, .27, .3),
        region('ООО "Получатель", ИНН 7725382984', .08, .30, .35),
        region("10. Перевозчик", .55, .60, .3),
        region('ООО "Перевозчик", ИНН 7707083894', .55, .63, .35),
    ]}
    roles = extract_printed_inns([page], document("trn"))["roles"]
    assert roles["customer"]["printed_inn"] == "7453297257"
    assert not any("Перевозчик" in e["text"] for e in roles["customer"]["evidence"])


def test_registry_real_http_contract_cache_and_targeted_name_normalization(tmp_path):
    registry, calls = service(tmp_path)
    async def run():
        matched = await registry.lookup("7707083893", "ООО «Продавец»")
        mismatch = await registry.lookup("7707083893", 'ООО "Иная организация"')
        return matched, mismatch
    matched, mismatch = asyncio.run(run())
    assert matched["status"] == "matched" and not matched["cache_hit"]
    assert mismatch["status"] == "mismatch" and mismatch["cache_hit"]
    assert len(calls) == 3 and registry.stats()["unique_lookups"] == 1
    saved = ''.join(p.read_text() for p in tmp_path.glob("*.json"))
    assert "session-token-must-not-cache" not in saved
    assert "registry-address-must-not-copy" not in saved
    assert FNS_ORIGIN in saved
    fresh, fresh_calls = service(tmp_path)
    assert asyncio.run(fresh.lookup("7707083893", 'ООО "Продавец"'))["status"] == "matched"
    assert not fresh_calls


@pytest.mark.parametrize("rows,status", [
    ([], "not_present"), ([{"cnt": "0", "t": "session-secret"}], "not_present"),
    ([record(name='ООО "Другой"') | {"c": 'ООО "Другой"'}], "mismatch"),
    ([record("7453297257")], "mismatch"), ([record(), record()], "ambiguous"),
    ([{"cnt": "1", "i": "7707083893", "n": "not a supported official record"}], "unavailable"),
])
def test_registry_outcomes_do_not_invent_verified_records(tmp_path, rows, status):
    registry, _ = service(tmp_path, rows)
    assert asyncio.run(registry.lookup("7707083893", 'ООО "Продавец"'))["status"] == status


def test_checksum_invalid_and_lookup_budget_prevent_network(tmp_path):
    registry, calls = service(tmp_path, budget=1)
    async def run():
        invalid = await registry.lookup("7707083894", 'ООО "Продавец"')
        await registry.lookup("7707083893", 'ООО "Продавец"')
        over = await registry.lookup("7453297257", 'ООО "Другой"')
        return invalid, over
    invalid, over = asyncio.run(run())
    assert invalid["failure_class"] == "invalid_inn" and over["failure_class"] == "lookup_budget"
    assert len(calls) == 3


@pytest.mark.parametrize("kwargs,expected_calls", [
    ({"home": "var CAPTCHA_REQUIRED = true"}, 1),
    ({"query": {"errors": {"captchaSearch": "solve challenge"}}}, 2),
    ({"result": {"captchaRequired": True, "rows": [record()]}}, 3),
])
def test_captcha_is_honest_unavailable_and_failure_is_cached(tmp_path, kwargs, expected_calls):
    registry, calls = service(tmp_path, **kwargs)
    async def run():
        first = await registry.lookup("7707083893", 'ООО "Продавец"')
        second = await registry.lookup("7707083893", 'ООО "Продавец"')
        return first, second
    first, second = asyncio.run(run())
    assert first["status"] == "unavailable" and first["failure_class"] == "captcha_required"
    assert second["cache_hit"] and len(calls) == expected_calls


def test_public_failure_class_never_logs_exception_text_or_credentials(tmp_path):
    def unavailable(request):
        raise httpx.ConnectError("private-key-marker/session-token", request=request)
    registry = FnsRegistry(tmp_path, transport=httpx.MockTransport(unavailable))
    result = asyncio.run(registry.lookup("7707083893", 'ООО "Продавец"'))
    assert result["status"] == "unavailable" and result["failure_class"] == "ConnectError"
    assert "private-key-marker" not in json.dumps(result) + ''.join(p.read_text() for p in tmp_path.glob("*.json"))


def test_pending_polling_is_bounded(tmp_path):
    registry, calls = service(tmp_path, result={"status": "wait"})
    result = asyncio.run(registry.lookup("7707083893", 'ООО "Продавец"'))
    assert result["failure_class"] == "result_pending" and len(calls) == 4


def test_individual_entrepreneur_prefix_matches_only_egrip_person(tmp_path):
    registry, _ = service(tmp_path, rows=[record("500100732259", name="ИВАНОВ ИВАН ИВАНОВИЧ", kind="fl")])
    result = asyncio.run(registry.lookup("500100732259", "ИП Иванов Иван Иванович"))
    assert result["status"] == "matched"


def test_unreadable_company_is_not_registration_match_and_terminated_record_is_explicit(tmp_path):
    registry, _ = service(tmp_path, rows=[record() | {"e": "01.01.2020"}])
    result = asyncio.run(registry.lookup("7707083893", None))
    assert result["status"] == "unavailable" and result["failure_class"] == "printed_name_unreadable"
    assert result["records"][0]["termination_date"] == "01.01.2020"


def test_disabled_registry_and_unknown_document_do_not_imply_registration():
    result = asyncio.run(verify_document_inns([upd()], document()))
    assert result["roles"]["consignor"]["registry"]["status"] == "unavailable"
    assert result["roles"]["consignor"]["registry"]["failure_class"] == "lookup_disabled"
    assert extract_printed_inns([upd()], document("unknown"))["roles"] == {}


def test_multiple_legal_names_in_same_own_inn_row_are_ambiguous_without_field_overwrite(tmp_path):
    page = upd(recipient="7725382984")
    # This represents a composite recipient line with a separate distribution
    # center entity; use synthetic names rather than actual expected values.
    for row in page["regions"]:
        if row["text"].startswith('ООО "Получатель"; ИНН'):
            row["text"] += ', РЦ ООО "Складской оператор"'
    original = document(frame_id="source");snapshot = deepcopy(original)
    registry, calls = service(tmp_path, rows=[record("7725382984", name='ООО "Складской оператор"') | {"c": 'ООО "Складской оператор"'}])
    metadata = asyncio.run(verify_document_inns([page], original, registry=registry))
    reading = metadata["roles"]["consignee"]
    assert reading["printed_inn"] == "7725382984"
    assert reading["printed_company_name"] == 'ООО "Получатель"'
    assert reading["company_name_binding"]["status"] == "ambiguous"
    assert set(reading["company_name_binding"]["legal_name_candidates"]) == {'ООО "Получатель"', 'ООО "Складской оператор"'}
    assert any('РЦ ООО "Складской оператор"' in proof["text"] for proof in reading["company_name_binding"]["evidence"])
    assert reading["registry"]["status"] == "ambiguous"
    assert reading["registry"]["records"][0]["name"] == 'ООО "Складской оператор"'
    assert original == snapshot
    assert len(calls) == 6  # both literal party INNs, shared bounded registry


def test_one_inline_legal_name_remains_unique_and_other_role_does_not_create_binding_ambiguity(tmp_path):
    page = upd()
    page["regions"][-2]["text"] += '; РЦ ООО "Другой покупательский склад"'
    registry, _ = service(tmp_path, rows=[record("7453297257", name='ООО "Получатель"') | {"c": 'ООО "Получатель"'}])
    reading = asyncio.run(verify_document_inns([page], document(), registry=registry))["roles"]["consignee"]
    assert reading["company_name_binding"]["status"] == "unique"
    assert reading["registry"]["status"] == "matched"
    assert not any("покупательский" in proof["text"] for proof in reading["company_name_binding"]["evidence"])


def test_explicit_buyer_inn_qualifier_is_rejected_even_inside_wide_consignee_geometry():
    page = upd()
    # A clipped buyer caption is not available for column partitioning. The
    # explicit INN/КПП покупателя label itself must still block role confusion.
    page["regions"] = [row for row in page["regions"] if not row["text"].startswith(("Покупатель:", 'ООО "Другой покупатель"'))]
    for row in page["regions"]:
        if row["text"].startswith('ООО "Получатель"; ИНН'):
            row["text"] = 'ООО "Получатель"'
    page["regions"].extend([region("ИНН/КПП покупателя:", .10, .25, .16), region("7725382984/773001001", .30, .25, .15)])
    reading = extract_printed_inns([page], document())["roles"]["consignee"]
    assert reading["printed_inn"] is None
    assert not any("покупателя" in proof["text"] for proof in reading["evidence"])
