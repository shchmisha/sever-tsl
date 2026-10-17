"""Observable image failures and candidate-preserving neighbor fallback."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from sever.quality import assess_frame_quality, evaluate_page_frames


def _text_page() -> Image.Image:
    page = Image.new('RGB', (800, 1100), 'white')
    draw = ImageDraw.Draw(page)
    for row in range(24):
        draw.text((40, 60 + row * 35), f'INVOICE 2026-10-04 Parties Customer Shipping {row} 123456789',
                  fill='black', font_size=20)
    return page


def _frame(tmp_path: Path, name: str, image: Image.Image, **extra) -> dict:
    path = tmp_path / f'{name}.png'
    image.save(path)
    return {'frame_id': name, 'path': str(path), **extra}


def test_readable_text_and_cropped_line_are_accepted(tmp_path):
    page = _text_page()
    full = assess_frame_quality(_frame(tmp_path, 'full', page))
    # Whole page corners, complete fields and document identity are intentionally
    # not requirements of the pixel screen. This cropped line is still readable.
    crop = assess_frame_quality(_frame(tmp_path, 'crop', page.crop((35, 55, 700, 90))))
    assert full['accepted'] and crop['accepted']
    assert not full['reasons'] and not crop['reasons']


def test_white_paper_does_not_imply_overexposure(tmp_path):
    page = Image.new('RGB', (800, 1100), 'white')
    ImageDraw.Draw(page).text((80, 140), 'Header 2026-10-04 / 7654321', font_size=24, fill='black')
    assert float((np.asarray(page) == 255).all(axis=2).mean()) > .99
    result = assess_frame_quality(_frame(tmp_path, 'mostly-white', page))
    assert result['accepted']
    assert result['metrics']['thumbnail']['canny_edge_fraction'] > 0


@pytest.mark.parametrize('color', ['black', (4, 5, 7)])
def test_blank_black_is_rejected_with_measured_reason(tmp_path, color):
    result = assess_frame_quality(_frame(tmp_path, 'black', Image.new('RGB', (600, 900), color)))
    assert not result['accepted'] and result['status'] == 'rejected'
    assert result['reasons'] == ['near_black_without_visible_detail']
    assert result['metrics']['maximum_rgb_channel'] <= 7


def test_uniform_frame_is_explicitly_rejected_without_white_fraction_rule(tmp_path):
    result = assess_frame_quality(_frame(tmp_path, 'uniform', Image.new('RGB', (600, 900), '#aaaabb')))
    assert result['reasons'] == ['near_uniform_without_visible_detail']
    assert result['metrics']['channel_ranges'] == [0, 0, 0]
    assert 'near_white_fraction' not in result['metrics']


def test_extreme_blurred_text_rejected_but_moderate_blur_passes(tmp_path):
    page = _text_page()
    moderate = assess_frame_quality(_frame(tmp_path, 'moderate', page.filter(ImageFilter.GaussianBlur(2))))
    blurred = assess_frame_quality(_frame(tmp_path, 'extreme', page.filter(ImageFilter.GaussianBlur(12))))
    assert moderate['accepted']
    assert not blurred['accepted']
    assert blurred['reasons'] == ['extreme_blur_or_smooth_content_without_fine_detail']
    assert blurred['metrics']['detail_confirmation'] is not None
    assert blurred['metrics']['detail_confirmation']['maximum_adjacent_gray_gradient'] < 12


def test_tiny_image_has_explicit_pixel_extent_reason(tmp_path):
    rng = np.random.default_rng(31)
    tiny = Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8))
    result = assess_frame_quality(_frame(tmp_path, 'tiny', tiny))
    assert not result['accepted']
    assert result['reasons'] == ['insufficient_pixel_extent']
    assert result['metrics']['original_pixel_size'] == [24, 24]


def test_skin_colored_overlay_alone_does_not_reject_readable_crop(tmp_path):
    page = _text_page()
    ImageDraw.Draw(page).rectangle((450, 500, 799, 1099), fill='#d59a7e')
    assert assess_frame_quality(_frame(tmp_path, 'overlay', page))['accepted']


def test_rejected_primary_uses_good_neighbor_preserving_order_pts_and_candidate(tmp_path):
    first = _frame(tmp_path, 'first', _text_page(), timestamp_s=1.2, source_pts=108000,
                   quality={'relative_selection_score': .8})
    bad = _frame(tmp_path, 'bad-primary', Image.new('RGB', (800, 1100), 'black'),
                 timestamp_s=1.5, source_pts=135000, quality={'relative_selection_score': 1.0})
    later = _frame(tmp_path, 'later', _text_page(), timestamp_s=1.8, source_pts=162000,
                   quality={'relative_selection_score': .9})
    page = {'page_id': 'page-0003', 'start_s': 1., 'end_s': 2.,
            'selected_frame_id': bad['frame_id'], 'frames': [first, bad, later]}
    original = deepcopy(page)
    result = evaluate_page_frames(page)
    assert result['status'] == 'accepted'
    assert result['selected_frame_id'] == 'later'
    assert result['original_selected_frame_id'] == 'bad-primary'
    assert result['accepted_frame_ids'] == ['first', 'later']
    assert [d['frame_id'] for d in result['frame_dispositions']] == ['first', 'bad-primary', 'later']
    assert [d['frame_id'] for d in result['rejected_frames']] == ['bad-primary']
    assert page == original


def test_existing_good_primary_retained_and_frame_count_not_capped(tmp_path):
    source = _frame(tmp_path, 'source', _text_page())
    frames = [{**source, 'frame_id': f'frame-{i}', 'source_pts': i,
               'quality': {'relative_selection_score': i}} for i in range(7)]
    result = evaluate_page_frames({'selected_frame_id': 'frame-2', 'frames': frames})
    assert result['selected_frame_id'] == 'frame-2'
    assert result['accepted_frame_ids'] == [f'frame-{i}' for i in range(7)]


def test_all_rejected_keeps_candidate_and_null_selection(tmp_path):
    frame = _frame(tmp_path, 'black', Image.new('RGB', (400, 600), 'black'))
    page = {'page_id': 'page-0009', 'selected_frame_id': 'black', 'frames': [frame], 'start_s': 4., 'end_s': 6.}
    original = deepcopy(page)
    result = evaluate_page_frames(page)
    assert result['status'] == 'rejected'
    assert result['selected_frame_id'] is None
    assert result['accepted_frame_ids'] == []
    assert len(result['rejected_frames']) == 1
    assert page == original


def test_unavailable_image_is_auditable_rejection(tmp_path):
    result = assess_frame_quality({'frame_id': 'missing', 'path': str(tmp_path / 'missing.png')})
    assert result['reasons'] == ['frame_image_unavailable']
    assert result['metrics']['error_type'] == 'FileNotFoundError'


def test_empty_candidate_is_not_silently_accepted():
    result = evaluate_page_frames({'page_id': 'empty', 'frames': [], 'selected_frame_id': None})
    assert result['status'] == 'rejected' and result['selected_frame_id'] is None
    assert result['frame_dispositions'] == []
