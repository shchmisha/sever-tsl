import asyncio
import json

from PIL import Image
import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever import zoom_image_reader as wrapper
from sever.agents import AnalysisError, BatchBudgetExceeded, FatalProviderError
from test_image_agent import frame, config, zoom_call, response, reading, last_panel
from test_image_reader import model_for, reading as native_reading


def orientation(value=90, *, bad=False, mutate=None):
    def respond(messages, info):
        if mutate:
            mutate()
        return ModelResponse(parts=[TextPart('{}' if bad else json.dumps({'rotation_clockwise': value}))],
                             usage=RequestUsage(input_tokens=12, output_tokens=3))
    return FunctionModel(respond)


def run(frame, tmp_path, **options):
    return asyncio.run(wrapper.read_frame_zoom(frame, source_id='source', page_id='leaf',
        cache_dir=tmp_path/'proof', trace_path=tmp_path/'trace.jsonl', **options))


def test_orientation_and_fresh_tool_reads_share_one_source_and_budget(frame, tmp_path):
    calls, settings = [], config()
    def respond(messages, info):
        calls.append(messages)
        return zoom_call() if len(calls) == 1 else response(reading(last_panel(messages)))
    result = run(frame, tmp_path, config=settings, orientation_model=orientation(), model=FunctionModel(respond))
    assert len(calls) == 2 and settings.usage.requests == 3
    assert settings.usage.input_tokens == 92 and settings.usage.tool_calls == 1
    assert result['recognition']['image_preparation']['independent_evidence_frames'] == 1
    assert result['recognition']['pixel_proof']['source_sha256'] == frame['source_sha256']
    assert {e.frame_id for e in result['document'].classification_evidence} == {frame['frame_id']}


@pytest.mark.parametrize('bad', [False, True])
def test_unknown_and_bad_orientation_fallback_has_one_actual_field_read(frame, tmp_path, bad):
    calls, settings = [], config()
    result = run(frame, tmp_path, config=settings, orientation_model=orientation(None, bad=bad),
                 model=model_for(native_reading(), calls))
    assert len(calls) == 1 and settings.usage.requests == 2
    assert len(result['recognition']['pixel_proof']['panels']) == 12
    assert result['recognition']['image_preparation']['fallback']


@pytest.mark.parametrize('bad', [False, True])
def test_source_changed_during_orientation_blocks_every_field_call(frame, tmp_path, bad):
    calls = []
    def mutate():
        Image.new('RGB', (1600, 1000), 'red').save(frame['path'])
    with pytest.raises(AnalysisError, match='changed after orientation'):
        run(frame, tmp_path, config=config(), orientation_model=orientation(bad=bad, mutate=mutate),
            model=model_for(native_reading(), calls))
    assert not calls


@pytest.mark.parametrize('failure', [FatalProviderError('stop'), BatchBudgetExceeded('stop'), AnalysisError('storage')])
def test_orientation_stop_never_falls_through_to_field_model(frame, tmp_path, monkeypatch, failure):
    calls = []
    async def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(wrapper, 'choose_reading_rotation', fail)
    with pytest.raises(type(failure)):
        run(frame, tmp_path, config=config(), model=model_for(native_reading(), calls))
    assert not calls
