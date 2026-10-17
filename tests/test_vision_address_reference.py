"""Source-role regressions; no provider calls or artifact mutations."""
from pathlib import Path
import json

import pytest

from sever.schema import FIELDS_BY_TYPE
from sever.normalize import normalize
from sever.vision import VisionDecision, VisionField, _validate_with_ocr


def _fixture(address, *, reference=True):
    texts = [
        ('Продавец:', [.05,.06,.12,.01]),
        ('ООО "Продавец"', [.2,.06,.5,.01]),
        ('Anoec', [.05,.09,.07,.01]),  # Actual OCR can damage the address caption.
        ('111111, Первый город, улица 1, дом 8', [.2,.09,.6,.01]),
        ('Грузоотправитель и его адрес:', [.05,.15,.16,.01]),
        ('он же' if reference else 'ООО "Отправитель"', [.2,.15,.5,.01]),
        ('Грузополучатель и его адрес:', [.05,.21,.16,.01]),
        ('ООО "Получатель"', [.2,.19,.5,.01]),
        ('222222, Второй город, улица 2, дом 9', [.2,.21,.6,.01]),
    ]
    page = {'frame_id':'f','regions':[{'text':text,'bbox':box,'processing_pass':'full_image'} for text,box in texts]}
    crop = {'frame_id':'f','crop_bbox_normalized_top_left':[0,0,1,.35],'source_full_page_visible':True,
            'role_panels':{
                'seller':{'panel_id':'seller','bbox':[0,.05,.9,.07],'caption_id':'r0','ids':['r0','r1','r2','r3']},
                'consignor':{'panel_id':'consignor','bbox':[0,.14,.9,.10],'caption_id':'r4',
                             'ids':['r4','r5','r6','r7','r8'],'explicit_seller_reference':reference},
                'consignee':{'panel_id':'consignee','bbox':[0,.18,.9,.06],'caption_id':'r6','ids':['r6','r7','r8']},
                'document_header':{'panel_id':'document_header','bbox':[0,0,.6,.04],'ids':[]}}}
    fields = {name:VisionField(status='unreadable') for name in FIELDS_BY_TYPE['upd']}
    fields['consignor_address'] = VisionField(raw_value=address,status='recognized',panel_id='consignor',
        evidence_text=address,evidence_ids=[],confidence=.5)
    output = VisionDecision(document_type='upd',heading_text='Универсальный передаточный документ',fields=fields)
    return page,crop,output


def test_empty_citations_cannot_assign_consignee_address_to_seller_reference():
    page,crop,output = _fixture('222222, Второй город, улица 2, дом 9')
    field = _validate_with_ocr(output,crop,'upd','s','p','test',page).fields['consignor_address']
    assert field.status == 'unreadable'
    assert field.raw_value is None and field.normalized_value is None
    assert 'consignee band' in field.reason
    assert {e.region_index for e in field.evidence} == {3,5,8}
    # The validator rejects the wrong role instead of copying a known address.
    assert output.fields['consignor_address'].raw_value.startswith('222222')


def test_matching_seller_postal_address_remains_a_visual_reading():
    raw = '111111, Первый город, улица 1, дом 8'
    page,crop,output = _fixture(raw)
    field = _validate_with_ocr(output,crop,'upd','s','p','test',page).fields['consignor_address']
    assert field.status == 'recognized' and field.raw_value == raw
    assert field.confidence == .5


def test_without_same_party_reference_shared_consignor_and_consignee_address_is_allowed():
    raw = '222222, Второй город, улица 2, дом 9'
    page,crop,output = _fixture(raw,reference=False)
    field = _validate_with_ocr(output,crop,'upd','s','p','test',page).fields['consignor_address']
    assert field.status == 'recognized' and field.raw_value == raw


def test_ambiguous_seller_postal_ocr_does_not_establish_a_role_mismatch():
    raw = '222222, Второй город, улица 2, дом 9'
    page,crop,output = _fixture(raw)
    page['regions'].append({'text':'222222, Первый город, улица 1','bbox':[.2,.09,.6,.01],'processing_pass':'header'})
    crop['role_panels']['seller']['ids'].append('r9')
    field = _validate_with_ocr(output,crop,'upd','s','p','test',page).fields['consignor_address']
    assert 'consignee band' not in (field.reason or '')


def test_actual_cached_source4_address_decisions_revalidate_without_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ocr_path = root/'runs/closing/20260818-4_132351/ocr/page-0004.json'
    cache_dir = root/'runs/mvp/vision-cache'
    if not ocr_path.exists() or not cache_dir.exists():
        pytest.skip('Private source-bound OCR and paid decisions are not distributed')
    frames = {'frame-000030','frame-000031','page-0004-extra-629df6cc3c57-pts-1751253'}
    pages = {page['frame_id']:page for page in json.loads(ocr_path.read_text()) if page['frame_id'] in frames}
    source_id = '3f6c876d8974fc549923cf005eff399833f445976fd50f2a714487c8260336e2'
    saved = {}
    for path in cache_dir.glob('vision-*.json'):
        value = json.loads(path.read_text())
        frame_id = value.get('crop',{}).get('frame_id')
        crop = value.get('crop', {})
        bound_image = pages.get(frame_id, {}).get('processing', {}).get('ocr_image_path')
        if (value.get('status')=='success' and frame_id in frames
                and value.get('source_id') == source_id and value.get('page_id') == 'page-0004'
                and bound_image and Path(crop.get('oriented_image_path', '')).resolve() == Path(bound_image).resolve()):
            saved[frame_id] = value
    if set(pages) != frames or set(saved) != frames:
        pytest.skip('All three actual cached views are required for this source regression')
    for frame_id in frames:
        cache = saved[frame_id]
        result = _validate_with_ocr(VisionDecision.model_validate(cache['decision']),cache['crop'],
            'upd',cache['source_id'],cache['page_id'],cache['model'],pages[frame_id])
        field = result.fields['consignor_address']
        if frame_id.startswith('frame-'):
            assert field.status == 'unreadable' and field.raw_value is None
        else:
            assert field.status == 'recognized'
            # The raw OCR spelling of the numero mark may differ from the
            # model's typography. The address words and all digits must agree.
            assert field.normalized_value == normalize('consignor_address',cache['decision']['fields']['consignor_address']['raw_value'])
            assert 'он же' not in field.raw_value.casefold()
