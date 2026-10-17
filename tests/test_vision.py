import asyncio
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, BatchBudgetExceeded
from sever.schema import FIELDS_BY_TYPE
from sever.vision import (VisionDecision, VisionField, _crop_bounds, _prepare_crop,
                          _validated, read_document)


def frame(tmp_path, *, full=False, kind="landscape"):
    image = tmp_path / "oriented.png"
    Image.new("RGB", (3840, 2160) if kind == "landscape" else (2160, 3840), "white").save(image)
    return {"frame_id": "best-frame", "full_page_visible": full,
            "processing": {"ocr_image_path": str(image), "ocr_to_source_matrix": [[0, 1, 0], [-1, 0, 3840], [0, 0, 1]]},
            "regions": [{"text": "Универсальный передаточный документ", "bbox": [.10, .04, .50, .03]},
                        {"text": "Грузополучатель", "bbox": [.10, .30, .70, .03]}]}


def decision(kind="upd", **overrides):
    fields = {name: VisionField(status="unreadable") for name in FIELDS_BY_TYPE[kind]}
    fields.update(overrides)
    heading = {"upd": "Универсальный передаточный документ", "trn": "Транспортная накладная",
               "ttn": "Товарно-транспортная накладная", "torg12": "Товарная накладная ТОРГ-12"}[kind]
    return VisionDecision(document_type=kind, heading_text=heading, fields=fields, type_confidence=.9)


def model_for(output, calls, *, fail=False):
    def respond(messages, info):
        calls.append(messages)
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_request_parameters.output_object is not None
        assert not info.function_tools and not info.output_tools
        contents = [part.content for message in messages for part in message.parts if hasattr(part, "content")]
        assert any(isinstance(item, BinaryContent) for content in contents if isinstance(content, list) for item in content)
        if fail:
            raise RuntimeError("private-key-marker must never be logged")
        return ModelResponse(parts=[TextPart(output.model_dump_json())],
                             usage=RequestUsage(input_tokens=550, output_tokens=160),
                             provider_details={"routerai_cost_rub": .01})
    return FunctionModel(respond, model_name="offline-vision-test")


def test_crop_is_bounded_saved_and_reversible(tmp_path):
    page = frame(tmp_path)
    cache = tmp_path / "cache";cache.mkdir()
    data, metadata = _prepare_crop(page, "upd", cache)
    assert data[:2] == b"\xff\xd8"
    assert max(metadata["upload_pixel_size"]) <= 1700
    assert metadata["crop_xyxy"][3] >= .38 * 2160
    assert metadata["crop_bbox_normalized_top_left"] != [0, 0, 1, 1]
    assert Path(metadata["crop_path"]).is_file()
    assert metadata["oriented_to_source_matrix"] == page["processing"]["ocr_to_source_matrix"]


def test_wrapped_low_customer_label_expands_portrait_crop_within_limit(tmp_path):
    page = frame(tmp_path, kind="portrait")
    page["regions"].append({"text": "Заказчик услуг, связанных с перевозкой груза", "bbox": [.1, .66, .6, .02]})
    box = _crop_bounds(page, (2160, 3840), "trn")
    assert .70 * 3840 <= box[3] <= .75 * 3840


def test_one_call_shared_usage_cache_and_source_panel_evidence(tmp_path):
    page = frame(tmp_path)
    calls = []
    output = decision(invoice_number=VisionField(raw_value="2560", status="recognized", confidence=.95,
                                                 evidence_text="Счет-фактура № 2560"),
                      invoice_date=VisionField(raw_value="20 августа 2026 г.", status="recognized",
                                               evidence_text="Счет-фактура № 2560 от 20 августа 2026 г."))
    config = AgentConfig(max_requests=2)
    config.usage.requests = 1
    config.usage.attempted_requests = 1
    kwargs = dict(source_id="source", page_id="page", document_type="upd", config=config,
                  cache_dir=tmp_path / "cache", trace_path=tmp_path / "trace.jsonl", model=model_for(output, calls))
    result = asyncio.run(read_document([page, {**page, "frame_id": "neighbor"}], **kwargs))
    again = asyncio.run(read_document([page], **kwargs))
    assert result == again
    assert len(calls) == 1
    assert config.usage.requests == config.usage.attempted_requests == 2
    assert config.usage.actual_cost_rub == .01
    assert config.usage.input_tokens == 550
    assert result.fields["invoice_date"].normalized_value == "2026-08-20"
    assert result.fields["invoice_number"].evidence[0].frame_id == "best-frame"
    assert result.fields["invoice_number"].evidence[0].region_index is None
    trace = [json.loads(s) for s in (tmp_path / "trace.jsonl").read_text().splitlines()]
    crop = next(s["crop"] for s in trace if s["event"] == "vision_crop_observation")
    panel = crop["role_panels"]["document_header"]
    assert result.fields["invoice_number"].evidence[0].bbox == panel["bbox"]
    assert panel["source_pixel_xyxy"][0] == 0
    assert Path(panel["crop_path"]).is_file()
    assert panel["upload_to_oriented_matrix"][0][2] == panel["source_pixel_xyxy"][0]
    assert trace[-1]["event"] == "vision_cache_hit" and trace[-1]["usage_delta"]["requests"] == 0


def test_own_heading_disagreement_does_not_force_requested_type(tmp_path):
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .6], "source_full_page_visible": True}
    wrong = decision("ttn")
    result = _validated(wrong, crop, "trn", "s", "p", "test")
    assert result.document_type == "unknown" and result.fields == {}
    false_upd = decision()
    false_upd.heading_text = "Счет-фактура № 2560"
    assert _validated(false_upd, crop, "upd", "s", "p", "test").document_type == "unknown"


def test_transport_blank_number_and_roles_do_not_fall_back_to_order_or_buyer():
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .75], "source_full_page_visible": True}
    output = decision("trn", document_number=VisionField(raw_value="11844", status="recognized", evidence_text="Заказ (заявка) № 11844"),
                      customer=VisionField(raw_value="ООО Перевозчик", status="recognized", evidence_text="Перевозчик ООО Перевозчик"),
                      consignee=VisionField(raw_value="ООО Покупатель", status="recognized", evidence_text="Покупатель ООО Покупатель"))
    result = _validated(output, crop, "trn", "s", "p", "test")
    assert all(result.fields[n].status == "unreadable" for n in ("document_number", "customer", "consignee"))
    output.fields["document_number"] = VisionField(status="not_present", explicit_blank=True, evidence_text="Транспортная накладная № [пусто]")
    assert _validated(output, crop, "trn", "s", "p", "test").fields["document_number"].status == "not_present"
    crop["source_full_page_visible"] = False
    assert _validated(output, crop, "trn", "s", "p", "test").fields["document_number"].status == "unreadable"


def test_explicit_on_zhe_resolution_requires_actual_company_output():
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .65], "source_full_page_visible": True}
    output = decision(consignor=VisionField(raw_value='ООО "Продавец"', status="recognized",
                       evidence_text='Грузоотправитель: он же; Продавец: ООО "Продавец"'))
    assert _validated(output, crop, "upd", "s", "p", "test").fields["consignor"].raw_value == 'ООО "Продавец"'
    output.fields["consignor"].raw_value = "он же"
    assert _validated(output, crop, "upd", "s", "p", "test").fields["consignor"].status == "unreadable"


def test_failure_is_cached_without_retry_or_credentials_in_artifacts(tmp_path):
    calls=[]
    config = AgentConfig(api_key="private-key-marker")
    kwargs=dict(source_id="s",page_id="p",document_type="upd",config=config,
                cache_dir=tmp_path/'cache',trace_path=tmp_path/'trace.jsonl',model=model_for(decision(),calls,fail=True))
    page=frame(tmp_path)
    with pytest.raises(AnalysisError):
        asyncio.run(read_document([page],**kwargs))
    with pytest.raises(AnalysisError):
        asyncio.run(read_document([page],**kwargs))
    assert len(calls)==1
    assert config.usage.requests == config.usage.attempted_requests == 1
    artifacts=(tmp_path/'trace.jsonl').read_text()+''.join(p.read_text() for p in (tmp_path/'cache').glob('*.json'))
    assert 'private-key-marker' not in artifacts


def test_budget_and_non_target_never_call_model(tmp_path):
    calls=[]
    config=AgentConfig(max_requests=1)
    config.usage.requests=1
    kwargs=dict(source_id="s",page_id="p",config=config,cache_dir=tmp_path/'cache',trace_path=tmp_path/'trace.jsonl',model=model_for(decision(),calls))
    page=frame(tmp_path)
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(read_document([page],document_type="upd",**kwargs))
    result=asyncio.run(read_document([page],document_type="unknown",**kwargs))
    assert result.document_type == 'unknown' and not calls


def test_upd_crop_excludes_unrelated_buyer_column(tmp_path):
    page = frame(tmp_path)
    page["regions"] = [{"text": "Продавец:", "bbox": [.18, .1, .04, .015]},
                       {"text": "Грузополучатель и его адрес:", "bbox": [.18, .22, .1, .02]},
                       {"text": "Покупатель:", "bbox": [.55, .14, .06, .02]}]
    box = _crop_bounds(page, (3840, 2160), "upd")
    assert box[2] < .57 * 3840
    assert box[3] > .28 * 2160


def test_organization_keeps_raw_requisites_but_normalizes_only_legal_name():
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .5], "source_full_page_visible": True}
    output = decision("ttn", consignor=VisionField(raw_value='ООО "Север", ИНН 1234567890, адрес', status="recognized",
                       evidence_text='Грузоотправитель: ООО "Север", ИНН 1234567890, адрес'))
    field = _validated(output, crop, "ttn", "s", "p", "test").fields["consignor"]
    assert field.status == "recognized" and field.normalized_value == 'ООО "Север"'
    assert "ИНН" in field.raw_value


def test_invoice_own_row_uses_local_number_before_internal_record():
    from sever.vision import _invoice_header_candidate, _validate_with_ocr
    page = {"frame_id": "f", "regions": [
        {"text": "Счет-фактура №", "bbox": [.18, .07, .05, .02]},
        {"text": "УР-99999999", "bbox": [.36, .065, .06, .02]},
        {"text": "00517", "bbox": [.29, .08, .03, .02], "processing_pass":"full"},
        {"text": "00517", "bbox": [.29, .081, .03, .02], "processing_pass":"header"}]}
    assert _invoice_header_candidate(page)[0] == "00517"
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0,0,.55,.3], "source_full_page_visible": True}
    output = decision(invoice_number=VisionField(raw_value="УР-99999999", status="recognized", evidence_text="УР-99999999"))
    result = _validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["invoice_number"].normalized_value == "00517"
    assert result.fields["invoice_number"].evidence[0].region_index == 0


def test_empty_echoed_schema_metadata_is_safe_other_extra_properties_rejected():
    from pydantic import ValidationError
    payload = decision().model_dump()
    payload["$defs"] = {}
    assert VisionDecision.model_validate(payload).document_type == "upd"
    payload["unrequested"] = "data"
    with pytest.raises(ValidationError): VisionDecision.model_validate(payload)


def test_role_band_consensus_excludes_bank_and_upstream_organization():
    from sever.vision import _role_sources, _validate_with_ocr
    page = {"frame_id": "f", "regions": [
        {"text": 'ООО "Вышестоящая компания"', "bbox": [.15,.07,.65,.015], "processing_pass": "full"},
        {"text": 'ООО "Собственный поставщик"', "bbox": [.18,.17,.60,.015], "processing_pass": "full"},
        {"text": 'ООО "Собственный поставщик"', "bbox": [.18,.171,.60,.015], "processing_pass": "header"},
        {"text": "Адрес, банк АО Банк, БИК 123456789", "bbox": [.18,.189,.65,.012], "processing_pass": "full"},
        {"text": "Постващик", "bbox": [.07,.185,.10,.012], "processing_pass": "full"},
        {"text": 'ООО "Плательщик"', "bbox": [.18,.215,.60,.015], "processing_pass": "full"},
        {"text": "Плательщик", "bbox": [.07,.23,.1,.015], "processing_pass": "full"}]}
    panel = _role_sources(page,"torg12")["supplier"]
    assert panel["bbox"][1] < .17 and panel["bbox"][1] + panel["bbox"][3] < .215
    crop = {"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.45],"source_full_page_visible":True}
    output = decision("torg12", supplier=VisionField(raw_value='ООО "Вышестоящая компания"',status="recognized", evidence_text='Поставщик ООО "Вышестоящая компания"'))
    result = _validate_with_ocr(output,crop,"torg12","s","p","test",page)
    assert result.fields["supplier"].raw_value == 'ООО "Собственный поставщик"'
    assert result.fields["supplier"].confidence is None
    assert {e.region_index for e in result.fields["supplier"].evidence} == {1,2,4}


def test_foreign_role_ids_and_repeated_wrong_postal_code_rejected():
    from sever.vision import _validate_with_ocr
    page = {"frame_id":"f", "regions":[
        {"text":"Грузополучатель", "bbox":[.10,.22,.2,.015], "processing_pass":"full"},
        {"text":"628300, город, дом 6", "bbox":[.35,.22,.22,.015], "processing_pass":"full"},
        {"text":"628300, город, дом 6", "bbox":[.35,.221,.22,.015], "processing_pass":"header"},
        {"text":"Покупатель", "bbox":[.60,.14,.2,.015], "processing_pass":"full"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.32],"source_full_page_visible":True}
    output=decision(consignee_address=VisionField(raw_value="623000, город, дом 6", status="recognized", panel_id="consignee",
        evidence_ids=["r1","r2"], evidence_text="Грузополучатель 623000, город, дом 6"))
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["consignee_address"].status == "conflict"
    output.fields["consignee_address"].raw_value="628300, город, дом 6"
    output.fields["consignee_address"].evidence_ids=["r3"]
    assert _validate_with_ocr(output,crop,"upd","s","p","test",page).fields["consignee_address"].status == "unreadable"


def test_transport_blank_own_number_cannot_use_blank_order_proof():
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.52],"source_full_page_visible":True}
    output=decision("trn",document_number=VisionField(status="not_present",explicit_blank=True,evidence_text="Заказ (заявка) № [пусто]"))
    assert _validated(output,crop,"trn","s","p","test").fields["document_number"].status == "unreadable"


def test_clipped_distinct_upd_own_body_heading_is_supported():
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.3],"source_full_page_visible":False}
    output=decision();output.heading_text="ерсальный передаточный документ № 2359 от 20.08.2026"
    assert _validated(output,crop,"upd","s","p","test").document_type == "upd"
    output.heading_text="Приложение к универсальному передаточному документу № 2359"
    assert _validated(output,crop,"upd","s","p","test").document_type == "unknown"


def test_own_number_and_date_consensus_keeps_leading_zeros_and_excludes_reference():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[
        {"text":"Номер документа","bbox":[.40,.25,.09,.012],"processing_pass":"full"},
        {"text":"Дата составления","bbox":[.51,.25,.10,.012],"processing_pass":"full"},
        {"text":"ПЕ0006980","bbox":[.42,.267,.07,.012],"processing_pass":"full"},
        {"text":"ПЕ0006980","bbox":[.42,.268,.07,.012],"processing_pass":"header"},
        {"text":"15.03.2025","bbox":[.52,.267,.08,.012],"processing_pass":"full"},
        {"text":"15.03.2025","bbox":[.52,.268,.08,.012],"processing_pass":"header"},
        {"text":"Транспортная накладная","bbox":[.76,.25,.22,.012],"processing_pass":"full"},
        {"text":"11844","bbox":[.94,.27,.04,.012],"processing_pass":"full"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.45],"source_full_page_visible":True}
    output=decision("torg12",document_number=VisionField(raw_value="ПЕ0006890",status="recognized",evidence_text="Номер документа ПЕ0006890"),
        document_date=VisionField(raw_value="01.01.2025",status="recognized",evidence_text="Дата составления 01.01.2025"))
    result=_validate_with_ocr(output,crop,"torg12","s","p","test",page)
    assert result.fields["document_number"].raw_value == "ПЕ0006980"
    assert result.fields["document_date"].normalized_value == "2025-03-15"
    assert {e.region_index for e in result.fields["document_number"].evidence} == {0,2,3}


def test_single_or_ambiguous_ocr_number_does_not_overwrite_pixel_reading():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[
        {"text":"Счет-фактура №","bbox":[.10,.07,.1,.012]},
        {"text":"00360","bbox":[.24,.071,.06,.012],"processing_pass":"full"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,.6,.32],"source_full_page_visible":True}
    output=decision(invoice_number=VisionField(raw_value="00300",status="recognized",evidence_text="Счет-фактура № 00300"))
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["invoice_number"].raw_value == "00300"


def test_own_number_digit_guard_does_not_repair_ambiguous_letter_script():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[
        {"text":"Номер документа","bbox":[.40,.25,.09,.012]},
        {"text":"ПЕ0006980","bbox":[.42,.267,.07,.012],"processing_pass":"full"},
        {"text":"ПE0006980","bbox":[.42,.268,.07,.012],"processing_pass":"header"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.45],"source_full_page_visible":True}
    output=decision("torg12",document_number=VisionField(raw_value="ПЕ0006890",status="recognized",evidence_text="Номер документа ПЕ0006890"))
    field=_validate_with_ocr(output,crop,"torg12","s","p","test",page).fields["document_number"]
    assert field.status=="conflict" and field.raw_value is None
    output.fields["document_number"].raw_value="ПЕ0006980"
    assert _validate_with_ocr(output,crop,"torg12","s","p","test",page).fields["document_number"].raw_value=="ПЕ0006980"


def test_bank_inside_role_address_is_not_company_consensus():
    from sever.vision import _consensus_organization
    lookup={"r0":{"text":'Адрес, в банке АО "Банк"',"processing_pass":"full"},
            "r1":{"text":'Адрес, в банке АО "Банк"',"processing_pass":"header"}}
    assert _consensus_organization(["r0","r1"],lookup) is None


def test_same_frame_own_upd_proof_supports_invoice_panel_without_promoting_references():
    from sever.vision import _validate_with_ocr
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,.6,.32],"source_full_page_visible":True}
    output=decision();output.heading_text="Счет-фактура"
    page={"frame_id":"f","regions":[{"text":"УПД № 00125 от 20 августа 2026 г.","bbox":[.2,.25,.4,.015]}]}
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.document_type=="upd" and result.classification_evidence[-1].region_index==0
    assert all(field.status=="unreadable" for field in result.fields.values())
    page["regions"][0]["text"]="К накладной (УПД) № 00125 от 20 августа 2026 г."
    assert _validate_with_ocr(output,crop,"upd","s","p","test",page).document_type=="unknown"
    page["regions"][0]["text"]="Универсальный передаточный документ"
    page["regions"][0]["bbox"]=[.01,.1,.02,.5]
    assert _validate_with_ocr(output,crop,"upd","s","p","test",page).document_type=="unknown"


def test_paid_native_parse_failure_recovery_requires_exact_input_and_makes_no_call(tmp_path):
    page=frame(tmp_path);calls=[]
    output=decision();output.fields["consignee"].evidence_ids=[f"r{i}" for i in range(20)]
    config=AgentConfig(max_requests=1)
    kwargs=dict(source_id="s",page_id="p",document_type="upd",config=config,cache_dir=tmp_path/"cache",
                trace_path=tmp_path/"trace.jsonl",model=model_for(output,calls))
    original=asyncio.run(read_document([page],**kwargs))
    cache=next((tmp_path/"cache").glob("vision-*.json"))
    cache.write_text(json.dumps({"status":"failed","error_type":"UnexpectedModelBehavior"}))
    restored=asyncio.run(read_document([page],**kwargs))
    assert restored==original and len(calls)==1 and config.usage.requests==1
    saved=json.loads(cache.read_text())
    assert saved["recovered_from_parse_failure"]
    assert saved["historical_paid_response_usage"]["input_tokens"]==550
    assert saved["usage_delta"]["requests"]==0
    # A changed actual OCR input cannot reuse the paid response or its failure.
    page["regions"][0]["text"]="Универсальный передаточный документ № 00999"
    with pytest.raises(BatchBudgetExceeded):asyncio.run(read_document([page],**kwargs))
    assert len(calls)==1


def test_own_date_space_separator_proof_is_local_and_order_date_is_excluded():
    from sever.vision import _role_sources,_hint_bundle,_validate_with_ocr
    page={"frame_id":"f","regions":[
        {"text":"Дата","bbox":[.09,.12,.04,.012],"processing_pass":"full"},
        {"text":"04.09 2025","bbox":[.14,.12,.08,.012],"processing_pass":"full"},
        {"text":"04.09.2025","bbox":[.14,.121,.08,.012],"processing_pass":"header"},
        {"text":"Дата","bbox":[.52,.12,.04,.012],"processing_pass":"full"},
        {"text":"01.09.2025","bbox":[.57,.12,.08,.012],"processing_pass":"full"},
        {"text":"Транспортная накладная","bbox":[.23,.09,.25,.012],"processing_pass":"full"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.52],"source_full_page_visible":True}
    panels=_role_sources(page,"trn");before=_hint_bundle(page,"trn",panels)[0]
    assert "r1" not in before["fields"]["document_date"]["allowed_ocr_ids"]
    output=decision("trn",document_date=VisionField(raw_value="04.09.2025",status="recognized",panel_id="document_header",
        evidence_text="Дата 04.09.2025",evidence_ids=["r0","r1","r2"]))
    result=_validate_with_ocr(output,crop,"trn","s","p","test",page)
    assert result.fields["document_date"].normalized_value=="2025-09-04"
    assert _hint_bundle(page,"trn",panels)[0]==before
    output.fields["document_date"].raw_value="01.09.2025";output.fields["document_date"].evidence_ids=["r3","r4"]
    assert _validate_with_ocr(output,crop,"trn","s","p","test",page).fields["document_date"].status=="unreadable"


def test_explicit_same_page_reference_uses_only_literal_seller_not_more_consignee_votes():
    from sever.vision import _validate_with_ocr
    texts=[("Продавец:",[.05,.05,.12,.01],"full"),('ООО "Мой продавец"',[.2,.05,.5,.01],"full"),
           ("Адрес:",[.05,.08,.06,.01],"full"),("111111, Город, улица, дом 1,",[.2,.08,.6,.01],"full"),
           ("квартира 59",[.2,.10,.2,.01],"full"),("ИНН/КПП:",[.05,.12,.1,.01],"full"),
           ("1234567890",[.2,.12,.2,.01],"full"),("Грузоотправитель",[.05,.15,.15,.01],"full"),
           ("он же",[.2,.15,.06,.01],"full"),('ООО "Другой получатель"',[.2,.17,.6,.01],"full"),
           ('ООО "Другой получатель"',[.2,.171,.6,.01],"header"),
           ("999999, Другой город, дом 2",[.2,.19,.6,.01],"full"),
           ("Грузополучатель",[.05,.18,.15,.01],"full")]
    page={"frame_id":"f","regions":[{"text":t,"bbox":b,"processing_pass":p} for t,b,p in texts]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.32],"source_full_page_visible":True,
          "role_panels":{
              "seller":{"panel_id":"seller","bbox":[0,.04,.9,.09],"caption_id":"r0","ids":[f"r{i}" for i in range(7)]},
              "consignor":{"panel_id":"consignor","bbox":[0,.14,.9,.07],"caption_id":"r7","ids":[f"r{i}" for i in range(7,12)],"explicit_seller_reference":True},
              "consignee":{"panel_id":"consignee","bbox":[0,.16,.9,.06],"caption_id":"r12","ids":["r9","r10","r11","r12"]},
              "document_header":{"panel_id":"document_header","bbox":[0,0,.6,.04],"ids":[]}}}
    output=decision(consignor=VisionField(raw_value='ООО "Другой получатель"',status="recognized",panel_id="consignor",evidence_ids=["r9","r10"],evidence_text='Грузоотправитель ООО "Другой получатель"'),
        consignor_address=VisionField(raw_value="999999, Другой город, дом 2",status="recognized",evidence_text="Грузоотправитель 999999, Другой город, дом 2",panel_id="consignor",evidence_ids=["r11"]))
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["consignor"].raw_value=='ООО "Мой продавец"'
    address=result.fields["consignor_address"]
    assert address.raw_value=="111111, Город, улица, дом 1, квартира 59" and address.confidence is None
    assert {e.region_index for e in address.evidence}=={0,2,3,4,8}
    assert all("999999" not in e.text and "Другой" not in e.text for e in address.evidence)


def test_seller_address_stops_before_following_reference_when_role_caption_is_missing():
    from sever.vision import _seller_address_literal
    rows=[('Адрес:',[.05,.10,.06,.01]),('123456, Город, улица, дом 7,',[.20,.10,.60,.01]),
          ('квартира 18',[.20,.12,.20,.01]),('он же',[.20,.15,.10,.01]),
          ('ООО "Следующая сторона"',[.20,.17,.50,.01])]
    lookup={f'r{i}':{'text':text,'bbox':bbox,'processing_pass':'full'} for i,(text,bbox) in enumerate(rows)}
    literal,ids=_seller_address_literal({'ids':list(lookup)},lookup)
    assert literal=='123456, Город, улица, дом 7, квартира 18'
    assert ids==['r0','r1','r2']
    # If the suffix was not read, the source prefix stays incomplete. The
    # following reference cannot be substituted for an address continuation.
    del lookup['r2']
    literal,ids=_seller_address_literal({'ids':list(lookup)},lookup)
    assert literal=='123456, Город, улица, дом 7,'
    assert ids==['r0','r1']


def test_seller_address_can_use_unambiguous_value_pass_when_caption_only_read_in_other_pass():
    from sever.vision import _seller_address_literal
    lookup={
        'r0':{'text':'Адрес:','bbox':[.05,.10,.06,.01],'processing_pass':'header'},
        'r1':{'text':'123456, Город, улица, дом 7,','bbox':[.20,.10,.60,.01],'processing_pass':'full'},
        'r2':{'text':'квартира 18','bbox':[.20,.12,.20,.01],'processing_pass':'full'},
        'r3':{'text':'ИНН/КПП продавца:','bbox':[.05,.14,.10,.01],'processing_pass':'full'},
        'r4':{'text':'он же','bbox':[.20,.16,.10,.01],'processing_pass':'header'},
    }
    literal,ids=_seller_address_literal({'ids':list(lookup)},lookup)
    assert literal=='123456, Город, улица, дом 7, квартира 18'
    assert ids==['r0','r1','r2']
    # Conflicting literal value passes remain ambiguous, with no digit repair.
    lookup['r5']={'text':'123456, Город, улица, дом 8,','bbox':[.20,.101,.60,.01],'processing_pass':'left_roles'}
    assert _seller_address_literal({'ids':list(lookup)},lookup) is None


def test_partial_address_missing_repeated_source_postal_code_is_unreadable():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[{"text":"Грузополучатель","bbox":[.1,.22,.2,.015],"processing_pass":"full"},
        {"text":"111111, Город, улица, дом 1","bbox":[.35,.22,.5,.015],"processing_pass":"full"},
        {"text":"111111, Город, улица, дом 1","bbox":[.35,.221,.5,.015],"processing_pass":"header"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.32],"source_full_page_visible":True}
    output=decision(consignee_address=VisionField(raw_value="улица, дом 1",status="recognized",evidence_text="Грузополучатель улица, дом 1"))
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["consignee_address"].status=="unreadable"
    assert "incomplete" in result.fields["consignee_address"].reason
    assert output.fields["consignee_address"].raw_value=="улица, дом 1"


def test_field_stage_section_heading_preserves_source_supported_type_without_inferred_fields():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[{"text":"Оборотная сторона формы № 1-Т","bbox":[.3,.07,.5,.015]}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,1,.38],"source_full_page_visible":True}
    output=decision("ttn");output.heading_text="ТРАНСПОРТНЫЙ РАЗДЕЛ"
    assert _validate_with_ocr(output,crop,"ttn","s","p","test",page).document_type=="ttn"
    unknown=VisionDecision(document_type="unknown",heading_text="II ТРАНСПОРТНЫЙ РАЗДЕЛ")
    result=_validate_with_ocr(unknown,crop,"ttn","s","p","test",page)
    assert result.document_type=="ttn" and all(v.status=="unreadable" for v in result.fields.values())
    unknown.heading_text="Международная накладная CMR"
    assert _validate_with_ocr(unknown,crop,"ttn","s","p","test",page).document_type=="unknown"


def test_nearest_own_invoice_value_excludes_more_votes_for_farther_internal_record():
    from sever.vision import _validate_with_ocr
    page={"frame_id":"f","regions":[
        {"text":"Счет-фактура №","bbox":[.31,.15,.05,.012],"processing_pass":"full"},
        {"text":"00555","bbox":[.399,.155,.025,.012],"processing_pass":"header"},
        {"text":"УР-99988877","bbox":[.455,.141,.06,.012],"processing_pass":"full"},
        {"text":"УР-99988877","bbox":[.455,.142,.06,.012],"processing_pass":"header-number"},
        {"text":"УПД № 00555 от 20 августа 2026 г.","bbox":[.39,.26,.2,.015],"processing_pass":"full"}]}
    crop={"frame_id":"f","crop_bbox_normalized_top_left":[0,0,.58,.32],"source_full_page_visible":True}
    output=decision(invoice_number=VisionField(raw_value="00555",status="recognized",evidence_text="00555",panel_id="document_header",evidence_ids=["r1","r4"]))
    result=_validate_with_ocr(output,crop,"upd","s","p","test",page)
    assert result.fields["invoice_number"].normalized_value=="00555"
    assert all(e.region_index not in {2,3,4} for e in result.fields["invoice_number"].evidence)
