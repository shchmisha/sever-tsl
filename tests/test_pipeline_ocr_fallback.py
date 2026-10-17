"""Operational OCR failures must not become a paid, successful unknown result."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sever import classifier, ocr, pipeline, quality, report, video
from sever.agents import AgentConfig
from sever.schema import DocumentResult


def setup(monkeypatch, tmp_path, reader, *, rejected=()):
    source=tmp_path/'input.mp4';source.write_bytes(b'independent pipeline fixture')
    frames=[{'frame_id':f'f{i}', 'path':str(tmp_path/f'f{i}.png'), 'timestamp_s':i/10,
             'quality':{'relative_selection_score':4-i}} for i in range(4)]
    for frame in frames:
        Path(frame['path']).write_bytes(frame['frame_id'].encode())
    page={'page_id':'leaf','start_s':0,'end_s':1,'selected_frame_id':'f0','frames':frames}
    monkeypatch.setattr(video,'analyze_video',lambda *a,**k:[deepcopy(page)])
    accepted=[f['frame_id'] for f in frames if f['frame_id'] not in rejected]
    monkeypatch.setattr(quality,'evaluate_page_frames',lambda p:{'status':'accepted',
                         'selected_frame_id':'f0','accepted_frame_ids':accepted})
    monkeypatch.setattr(ocr,'read_image',reader)
    classified=[]
    async def classify(prepared, *, source_id, **kwargs):
        classified.extend(deepcopy(prepared))
        return {p['page_id']:DocumentResult(source_id=source_id,page_id=p['page_id'],document_type='unknown')
                for p in prepared}
    monkeypatch.setattr(classifier,'classify_pages',classify)
    monkeypatch.setattr(report,'write_report',lambda out,result:(out/'report.html').write_text('fixture'))
    return source,classified


def run(source,tmp_path):
    return asyncio.run(pipeline.process_video(source,tmp_path/'out',config=AgentConfig(verify_inn=False)))


def observation():
    return {'engine':'controlled-local-ocr','regions':[{'text':'QUALITY CERTIFICATE','bbox':[.1,.1,.5,.1]}],
            'processing':{}}


def test_primary_error_tries_other_accepted_frames_and_binds_actual_successful_frame(monkeypatch,tmp_path):
    calls=[]
    def read(path,*a,**k):
        calls.append(path.stem)
        if path.stem=='f0': raise RuntimeError('cold native compiler failed')
        return observation()
    source,classified=setup(monkeypatch,tmp_path,read,rejected=('f1',))
    result=run(source,tmp_path)
    assert calls==['f0','f2']
    assert classified[0]['ocr_pages'][0]['frame_id']=='f2'
    page=result['pages'][0]
    assert page['selected_frame_id']==page['quality_gate']['selected_frame_id']=='f2'
    assert page['original_selected_frame_id']=='f0'
    assert page['ocr_attempts'][0]['error_type']=='RuntimeError'
    rows=[json.loads(l) for l in (tmp_path/'out/trace.jsonl').read_text().splitlines()]
    assert any(r['event']=='ocr_fallback_selected' and r['previous_frame_id']=='f0' for r in rows)


def test_empty_reading_keeps_searching_without_sending_empty_text_when_neighbor_readable(monkeypatch,tmp_path):
    calls=[]
    def read(path,*a,**k):
        calls.append(path.stem)
        return {'engine':'fixture','regions':[]} if path.stem=='f0' else observation()
    source,classified=setup(monkeypatch,tmp_path,read)
    result=run(source,tmp_path)
    assert calls==['f0','f1']
    assert classified[0]['ocr_pages'][0]['regions']
    assert result['pages'][0]['ocr_attempts'][0]['status']=='empty'


def test_all_local_errors_fail_before_paid_classifier_and_final_success(monkeypatch,tmp_path):
    calls=[]
    def broken(path,*a,**k):
        calls.append(path.stem)
        raise RuntimeError('OCR process unavailable')
    source,classified=setup(monkeypatch,tmp_path,broken)
    with pytest.raises(pipeline.LocalOCRUnavailable): run(source,tmp_path)
    assert calls==['f0','f1','f2','f3'] and classified==[]
    assert not (tmp_path/'out/results.json').exists()
    rows=[json.loads(l) for l in (tmp_path/'out/trace.jsonl').read_text().splitlines()]
    assert rows[-1]['reason']=='local_ocr_unavailable'
    assert rows[-1]['model_requests_attempted']==0
    assert not any(r['event']=='run_completed' for r in rows)
