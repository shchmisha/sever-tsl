"""Image-only regressions for short displays and partial page-turn association."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from sever import video
from sever.frame_pool import _small_feature


def _page(seed):
    image = Image.new('RGB', (384, 680), 'white')
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    for y in range(25, 650, 22):
        draw.text((15, y), ' '.join(str(value) for value in rng.integers(10000, 99999, 7)), fill='black')
    return image


def test_readable_single_sample_between_stable_documents_is_retained():
    first, short, last = [_page(seed) for seed in (12, 45, 89)]
    images = [first, first, short, last, last]
    groups, discarded = video._segment_samples([video._features(image) for image in images])
    assert groups == [[0, 1], [2], [3, 4]]
    assert discarded == []


def test_single_brief_document_between_two_views_of_same_document_is_retained_in_order():
    first, short = [_page(seed) for seed in (12, 45)]
    groups, discarded = video._segment_samples([video._features(image) for image in [first, first, short, first, first]])
    assert groups == [[0, 1], [2], [3, 4]]
    assert discarded == []


def test_empty_transient_does_not_split_resumed_same_leaf():
    page = _page(12)
    groups, discarded = video._segment_samples([video._features(image) for image in [page, page, Image.new('RGB', page.size, 'gray'), page, page]])
    assert groups == [[0, 1, 3, 4]]
    assert discarded == [2]


def test_repeated_partial_sheet_does_not_override_spatial_content_difference():
    first = Image.new('RGB', (384, 680), 'white')
    draw = ImageDraw.Draw(first)
    font = ImageFont.load_default(size=18)
    for y in range(20, 650, 36):
        draw.text((15, y), f'ROW {y} DOCUMENT {y * y}', fill='black', font=font)
    mixed = first.copy()
    draw = ImageDraw.Draw(mixed)
    draw.rectangle((0, 160, 384, 360), fill='white')
    for x in range(20, 370, 18):
        draw.line((x, 160, x, 360), fill='black', width=2)
    features = [video._features(image) for image in (first, mixed)]
    proof = video._pair_evidence(*features)
    # A shared majority of the original table still supplies many inliers.
    assert proof['inliers'] >= 16
    assert proof['registered_difference'] < .11
    assert proof['partial_page_overlap'] is True
    assert proof['same_page'] is False


def test_single_changed_digit_does_not_become_a_page_turn():
    first = _page(12)
    changed = first.copy()
    ImageDraw.Draw(changed).rectangle((80, 300, 88, 310), fill='white')
    proof = video._pair_evidence(video._features(first), video._features(changed))
    assert proof['same_page'] is True


def test_real_page_turn_rejected_and_auditable_when_local_images_available():
    root = Path(__file__).resolve().parents[1] / 'output/accuracy-review'
    paths = [root / f'frame-{index:06d}.png' for index in (27, 28)]
    if not all(path.exists() for path in paths):
        pytest.skip('Original video frames are private local regression evidence')
    proof = video._pair_evidence(*[_small_feature(path) for path in paths])
    assert proof['inliers'] >= 100  # shared fragment alone previously admitted this turn
    assert proof['registered_difference'] < .11
    assert proof['partial_page_overlap'] is True
    assert proof['regional_agreement']['inconsistent_edge_fraction'] >= .18
    assert proof['same_page'] is False


def test_real_clean_page_with_exposure_change_is_not_fragmented():
    root = Path(__file__).resolve().parents[1] / 'output/accuracy-review'
    paths = [root / f'video4-thumb-{index:06d}.png' for index in (58, 60)]
    if not all(path.exists() for path in paths):
        pytest.skip('Private local source evidence unavailable')
    proof = video._pair_evidence(*[_small_feature(path) for path in paths])
    assert proof['same_page'] is True
    assert proof['exposure_change_rescued'] is True
    assert proof['registered_difference'] >= .11
    assert not proof['partial_page_overlap']


@pytest.mark.parametrize('brief_stable', [True, False])
def test_dense_confirmation_retains_stable_brief_leaf_but_audits_single_frame_blink(tmp_path, brief_stable):
    import json
    import shutil
    import subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg tools unavailable')
    first, brief = _page(12), _page(45)
    for index in range(40):
        shown = brief if (8 <= index <= 12 if brief_stable else index == 10) else first
        shown.save(tmp_path / f'input-{index:03d}.png')
    source = tmp_path / 'brief.mkv'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-framerate', '20',
                    '-i', str(tmp_path / 'input-%03d.png'), '-c:v', 'ffv1', str(source)], check=True)
    pages = video.analyze_video(source, tmp_path / 'result', sample_interval=.5)
    manifest = json.loads((tmp_path / 'result/video-analysis.json').read_text())
    if brief_stable:
        assert len(pages) == 3
        assert pages[1]['segmentation']['brief_view_confirmation']['confirmed'] is True
        assert pages[1]['frames'][0]['source_pts'] == 500
    else:
        assert len(pages) == 1
        assert manifest['sampling']['discarded_transition_samples'] == [1]
        rejected = manifest['sampling']['discarded_transition_evidence'][0]
        assert rejected['source_pts'] == 500
        assert rejected['dense_confirmation']['confirmed'] is False
        assert rejected['dense_confirmation']['probes']
