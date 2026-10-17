"""Prove fresh orientation cannot supply document answers or extra evidence."""
import asyncio
import hashlib
import json

from PIL import Image
import pytest
from pydantic_ai import BinaryContent, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever import adaptive_image_reader as adaptive
from sever.agents import AnalysisError, BatchBudgetExceeded, FatalProviderError
from test_image_reader import config, field, model_for, reading, source_frame


def orientation_model(calls, value=90, *, malformed=False, fail=None, mutate=None):
    def respond(messages, info):
        calls.append(messages)
        if mutate:
            mutate()
        if fail:
            raise fail
        return ModelResponse(parts=[TextPart('{}' if malformed else json.dumps({"rotation_clockwise": value}))],
                             usage=RequestUsage(input_tokens=11, output_tokens=3))
    return FunctionModel(respond)


def run(frame, tmp_path, *, settings=None, **models):
    return asyncio.run(adaptive.read_frame_adaptive(frame, source_id="video-source", page_id="one-leaf",
        config=settings or config(), cache_dir=tmp_path / "proof", trace_path=tmp_path / "trace.jsonl", **models))


def actual_content(messages):
    return [item for message in messages for part in message.parts
            if isinstance(getattr(part, "content", None), list) for item in part.content]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_two_fresh_calls_but_one_evidence_frame_no_orientation_answer_in_field_prompt(source_frame, tmp_path, rotation):
    ocalls, calls = [], []
    payload = reading("upd", fields={"invoice_number": field("Н-15", status="recognized",
        text="Счет-фактура № Н-15", panels=[f"detail_0_0_r{rotation}"])})
    payload["heading_panel_ids"] = [f"context_r{rotation}"]
    settings = config()
    result = run(source_frame, tmp_path, settings=settings,
        orientation_model=orientation_model(ocalls, rotation), model=model_for(payload, calls))
    assert len(calls) == len(ocalls) == 1 and settings.usage.requests == 2
    assert settings.usage.input_tokens == 61
    content = actual_content(calls[0][0])
    assert len([item for item in content if isinstance(item, BinaryContent)]) == 5
    prompt = '\n'.join(item for item in content if isinstance(item, str))
    assert 'raw_decision' not in prompt and 'rotation_clockwise"' not in prompt
    assert source_frame['path'] not in prompt and 'video-source' not in prompt
    proof = result['recognition']['pixel_proof']
    assert all(p['rotation_clockwise_degrees'] == rotation for p in proof['panels'].values())
    assert proof['source_sha256'] == hashlib.sha256(open(source_frame['path'], 'rb').read()).hexdigest()
    doc = result['document']
    assert doc.fields['invoice_number'].raw_value == 'Н-15'
    assert {e.frame_id for e in doc.fields['invoice_number'].evidence} == {source_frame['frame_id']}
    assert result['recognition']['image_preparation']['independent_evidence_frames'] == 1


@pytest.mark.parametrize("malformed", [False, True])
def test_unknown_or_malformed_orientation_still_reads_all_rotations(source_frame, tmp_path, malformed):
    ocalls, calls = [], []
    result = run(source_frame, tmp_path, orientation_model=orientation_model(ocalls, None, malformed=malformed),
                 model=model_for(reading(), calls))
    assert len(ocalls) == len(calls) == 1
    assert len([item for item in actual_content(calls[0][0]) if isinstance(item, BinaryContent)]) == 12
    assert result['recognition']['image_preparation']['fallback'] == (
        'orientation_response_schema_failed' if malformed else 'orientation_unknown')


@pytest.mark.parametrize("error", [FatalProviderError('stop'), BatchBudgetExceeded('stop'), AnalysisError('storage')])
def test_stop_condition_never_causes_field_call(source_frame, tmp_path, monkeypatch, error):
    calls = []
    async def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(adaptive, 'choose_reading_rotation', fail)
    with pytest.raises(type(error)):
        run(source_frame, tmp_path, model=model_for(reading(), calls))
    assert calls == []


@pytest.mark.parametrize("malformed", [False, True])
def test_changed_source_during_orientation_cannot_reach_field_reading(source_frame, tmp_path, malformed):
    calls = []
    def mutate():
        Image.new('RGB', (400, 600), 'red').save(source_frame['path'])
    with pytest.raises(AnalysisError, match='changed after orientation'):
        run(source_frame, tmp_path, orientation_model=orientation_model([], malformed=malformed, mutate=mutate),
            model=model_for(reading(), calls))
    assert calls == []


def test_shared_request_limit_prevents_field_call_after_orientation(source_frame, tmp_path):
    calls = []
    with pytest.raises(BatchBudgetExceeded):
        run(source_frame, tmp_path, settings=config(max_requests=1),
            orientation_model=orientation_model([]), model=model_for(reading(), calls))
    assert calls == []
