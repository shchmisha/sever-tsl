"""Local native output must request literal evidence without generating defaults."""
import asyncio
import hashlib
import json

from PIL import Image
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.agents import AgentConfig, AnalysisError, _dump
from sever.schema import FIELDS_BY_TYPE
from sever.vision import (LOCAL_VISION_SCHEMA_VERSION, LocalVisionDecision, LocalVisionField,
                          VisionDecision, VisionField, _validated, read_document)


def decision(*, evidence='Грузополучатель: ООО "Получатель"', omit=None):
    fields = {name: {"status": "unreadable", "evidence_text": "", "panel_id": None, "evidence_ids": []}
              for name in FIELDS_BY_TYPE["upd"]}
    fields["consignee"] = {"status": "recognized", "raw_value": 'ООО "Получатель"',
        "evidence_text": evidence, "panel_id": "consignee", "evidence_ids": ["r1", "r2"], "confidence": .9}
    if omit:
        fields["consignee"].pop(omit)
    return {"document_type": "upd", "heading_text": "Универсальный передаточный документ",
            "heading_evidence_ids": ["r0"], "fields": fields}


def frame(tmp_path):
    path = tmp_path / "actual-pixels.png"
    Image.new("RGB", (960, 640), "white").save(path)
    return {"frame_id": "source-frame", "full_page_visible": True,
            "processing": {"ocr_image_path": str(path)}, "regions": [
                {"text": "Универсальный передаточный документ", "bbox": [.05, .04, .5, .02], "processing_pass": "full_image"},
                {"text": "Грузополучатель", "bbox": [.05, .10, .2, .015], "processing_pass": "full_image"},
                {"text": 'ООО "Получатель"', "bbox": [.30, .10, .3, .015], "processing_pass": "full_image"}]}


def model_for(payload, calls, *, local=True):
    def respond(messages, info):
        params = info.model_request_parameters
        assert params.output_mode == "native" and params.output_object is not None
        assert not info.function_tools and not info.output_tools
        schema = params.output_object.json_schema
        field = schema["$defs"]["LocalVisionField" if local else "VisionField"]
        if local:
            assert {"evidence_text", "panel_id", "evidence_ids"} <= set(field["required"])
            assert "default" not in field["properties"]["evidence_text"]
        else:
            assert field["properties"]["evidence_text"]["default"] == ""
        contents = [p.content for message in messages for p in message.parts if hasattr(p, "content")]
        assert any(isinstance(item, BinaryContent) for content in contents if isinstance(content, list) for item in content)
        calls.append(schema)
        return ModelResponse(parts=[TextPart(json.dumps(payload, ensure_ascii=False))],
                             usage=RequestUsage(input_tokens=100, output_tokens=50))
    return FunctionModel(respond, model_name="offline-local-schema-test")


def kwargs(tmp_path, calls, payload, *, local=True):
    config = AgentConfig(provider_kind="local" if local else "remote", base_url="http://127.0.0.1:8090/v1")
    return {"source_id": "source", "page_id": "leaf", "document_type": "upd", "config": config,
            "cache_dir": tmp_path / "cache", "trace_path": tmp_path / "trace.jsonl", "model": model_for(payload, calls, local=local)}


def test_local_schema_requires_explicit_evidence_without_changing_remote_schema():
    schema = LocalVisionDecision.model_json_schema()
    field = schema["$defs"]["LocalVisionField"]
    assert {"heading_text", "heading_evidence_ids", "fields"} <= set(schema["required"])
    assert {"evidence_text", "panel_id", "evidence_ids"} <= set(field["required"])
    for name in ("evidence_text", "panel_id", "evidence_ids"):
        assert "default" not in field["properties"][name]
    assert "Literal transcription" in field["properties"]["evidence_text"]["description"]
    assert "он же" in field["properties"]["evidence_text"]["description"]
    assert {"type": "null"} in field["properties"]["panel_id"]["anyOf"]
    remote = VisionDecision.model_json_schema()["$defs"]["VisionField"]
    assert remote["properties"]["evidence_text"]["default"] == ""
    assert "evidence_text" not in remote["required"]


@pytest.mark.parametrize("name", ["evidence_text", "panel_id", "evidence_ids"])
def test_local_parser_does_not_supply_omitted_evidence_properties(name):
    with pytest.raises(ValidationError):
        LocalVisionDecision.model_validate(decision(omit=name))


def test_explicit_null_panel_and_empty_evidence_are_allowed_only_as_unresolved_values():
    field = LocalVisionField(status="unreadable", evidence_text="", panel_id=None, evidence_ids=[])
    assert field.raw_value is None and field.panel_id is None
    output = LocalVisionDecision.model_validate(decision(evidence=""))
    crop = {"frame_id": "f", "crop_bbox_normalized_top_left": [0, 0, 1, .5], "source_full_page_visible": True}
    result = _validated(output, crop, "upd", "s", "p", "test")
    assert result.fields["consignee"].status == "unreadable"
    assert result.fields["consignee"].raw_value is None


@pytest.mark.parametrize("evidence,expected", [('Грузополучатель: ООО "Получатель"', "recognized"), ("", "unreadable")])
def test_native_local_protocol_retains_literal_evidence_and_rejects_empty_text(tmp_path, evidence, expected):
    page = frame(tmp_path)
    calls = []
    options = kwargs(tmp_path, calls, decision(evidence=evidence))
    result = asyncio.run(read_document([page], **options))
    field = result.fields["consignee"]
    assert field.status == expected and len(calls) == 1
    assert options["config"].usage.actual_cost_rub == 0
    if expected == "recognized":
        assert field.raw_value == 'ООО "Получатель"' and field.evidence[0].text == evidence
    else:
        assert field.raw_value is None
        assert all(not item.text for item in field.evidence)
    cached = next(options["cache_dir"].glob("vision-*.json"))
    cache = json.loads(cached.read_text())
    assert cache["response_schema_version"] == LOCAL_VISION_SCHEMA_VERSION
    assert cache["response_schema_sha256"] == hashlib.sha256(_dump(LocalVisionDecision.model_json_schema()).encode()).hexdigest()
    again = asyncio.run(read_document([page], **options))
    assert again == result and len(calls) == 1


def test_local_schema_change_cannot_reuse_legacy_default_response_cache(tmp_path):
    page = frame(tmp_path)
    calls = []
    options = kwargs(tmp_path, calls, decision())
    asyncio.run(read_document([page], **options))
    cache_path = next(options["cache_dir"].glob("vision-*.json"))
    old = json.loads(cache_path.read_text())
    old.pop("response_schema_version")
    old.pop("response_schema_sha256")
    old["decision"]["fields"]["consignee"]["evidence_text"] = ""
    (options["cache_dir"] / "vision-old-local-default-schema.json").write_text(json.dumps(old))
    cache_path.unlink()
    result = asyncio.run(read_document([page], **options))
    assert len(calls) == 2 and result.fields["consignee"].status == "recognized"


def test_failed_local_required_property_cannot_recover_through_legacy_defaults(tmp_path):
    page = frame(tmp_path)
    calls = []
    options = kwargs(tmp_path, calls, decision(omit="evidence_text"))
    with pytest.raises(AnalysisError):
        asyncio.run(read_document([page], **options))
    with pytest.raises(AnalysisError, match="Cached vision failure"):
        asyncio.run(read_document([page], **options))
    assert len(calls) == 1
    rows = [json.loads(line) for line in options["trace_path"].read_text().splitlines()]
    assert not any(row["event"] == "vision_paid_parse_recovered" for row in rows)


def test_remote_native_output_keeps_legacy_schema_and_cache_metadata(tmp_path):
    page = frame(tmp_path)
    calls = []
    options = kwargs(tmp_path, calls, decision(), local=False)
    result = asyncio.run(read_document([page], **options))
    assert result.fields["consignee"].status == "recognized"
    cache = json.loads(next(options["cache_dir"].glob("vision-*.json")).read_text())
    assert "response_schema_version" not in cache and "response_schema_sha256" not in cache
    assert len(calls) == 1


def test_remote_cannot_revalidate_local_response_with_same_alias_url_and_pixels(tmp_path):
    page = frame(tmp_path)
    calls = []
    local = kwargs(tmp_path, calls, decision(), local=True)
    remote = kwargs(tmp_path, calls, decision(), local=False)
    assert local["config"].base_url == remote["config"].base_url
    assert local["model"].model_name == remote["model"].model_name
    assert local["cache_dir"] == remote["cache_dir"]
    asyncio.run(read_document([page], **local))
    result = asyncio.run(read_document([page], **remote))
    assert result.fields["consignee"].status == "recognized" and len(calls) == 2
    assert "LocalVisionField" in calls[0]["$defs"]
    assert "VisionField" in calls[1]["$defs"]
    rows = [json.loads(line) for line in remote["trace_path"].read_text().splitlines()]
    assert not any(row["event"] == "vision_cache_revalidated" for row in rows)
