from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from sever import frame_pool


def _log(points, *, time_base="1/1000"):
    # Deliberately rounded decimal timestamps: integer PTS remain authoritative.
    return "[Parsed_showinfo_2 @ x] config in time_base: " + time_base + ", frame_rate: 4/1\n" + "\n".join(
        f"[Parsed_showinfo_2 @ x] n: {i} pts: {point} pts_time:0.1 fmt:yuv" for i, point in enumerate(points))


def _registered(a, b):
    if a.identity == b.identity:
        return {"same_page": True, "method": "near_identical_image", "mean_abs_difference": 0.}
    return {"same_page": b.identity != 7, "method": "ORB_RANSAC_registered_image", "inliers": 30,
            "inlier_fraction": .2, "feature_coverage": .4, "registered_difference": .02,
            "mean_abs_difference": min(.3, .03 * abs(a.identity - b.identity))}


def _setup(tmp_path, monkeypatch, *, mismatch=False, no_registration=False, mismatch_on_seek=False):
    source = tmp_path / "source.mp4"; source.write_bytes(b"local-source-fixture")
    primary = tmp_path / "primary.jpg"; Image.new("RGB", (128, 96), "black").save(primary)
    page = {"page_id": "page-0001", "start_s": 0., "end_s": 2., "selected_frame_id": "original-primary",
            "frames": [{"frame_id": "original-primary", "path": str(primary), "source_pts": 2000,
                        "source_time_base": "1/1000", "timestamp_s": 0.}]}
    points = [2000 + index * 250 for index in range(8)]
    identities = [0, 1, 2, 3, 4, 5, 5, 7]  # duplicate view, then another leaf
    state = {"calls": [], "full_extractions": [], "temporary_dirs": []}
    def features(image):
        identity = round(image.convert("RGB").getpixel((0, 0))[0] / 20)
        return SimpleNamespace(identity=identity, quality={"sharpness_laplacian_variance": 10 + identity * 10,
                                                         "limitations": []})
    def run(command):
        state["calls"].append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "index": 0, "start_time": "2.0", "time_base": "1/1000",
                 "side_data_list": [{"rotation": 90}]}]}), stderr="")
        target = Path(command[-1]);state["temporary_dirs"].append(target.parent)
        if "scale=384:-2" in command[command.index("-vf") + 1]:
            for index, identity in enumerate(identities, 1):
                Image.new("RGB", (384, 288), (identity * 20, 0, 0)).save(target.parent / f"sample-{index:06}.jpg")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr=_log(points))
        match = re.search(r"eq\(pts,(-?\d+)\)", command[command.index("-vf") + 1])
        assert match and "-copyts" in command
        if "-ss" in command:
            assert command.index("-ss") < command.index("-i")
        point = int(match.group(1)); state["full_extractions"].append(point)
        Image.new("RGB", (128, 96), (identities[points.index(point)] * 20, 0, 0)).save(target)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=_log([point + 1 if mismatch or mismatch_on_seek and "-ss" in command else point]))
    monkeypatch.setattr(frame_pool, "_tool", lambda name: name)
    monkeypatch.setattr(frame_pool, "_run", run)
    monkeypatch.setattr(frame_pool, "_features", features)
    monkeypatch.setattr(frame_pool, "_pair_evidence", (lambda a, b: {"same_page": False}) if no_registration else _registered)
    return source, page, state


def test_unlimited_registered_distinct_pts_are_lazy_exact_pts_jpeg(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch)
    unchanged = deepcopy(page)
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    assert not state["calls"]  # no decoding before iteration
    first = next(generator)
    assert len(state["full_extractions"]) == 1
    assert not any(directory.exists() for directory in state["temporary_dirs"])
    frames = [first, *generator]
    # Distinct timestamps survive even when their thumbnails look equal.
    # The caller's native-pixel policy owns redundancy, not this thumbnail pool.
    assert len(frames) == 6  # no two/three-frame retention cap
    assert [frame["source_pts"] for frame in frames] == [3250, 3500, 3000, 2750, 2500, 2250]
    assert [frame["timestamp_s"] for frame in frames] == [1.25, 1.5, 1., .75, .5, .25]
    assert all(frame["source_pts_time_s"] == float(frame["source_pts"] * Fraction("1/1000")) for frame in frames)
    assert all(frame["encoding"]["exact_pts_confirmed"] and frame["encoding"]["ffmpeg_qscale"] == 2 for frame in frames)
    assert all(frame["pixel_size"] == [128, 96] and Path(frame["path"]).suffix == ".jpg" for frame in frames)
    assert all(frame["display_rotation_applied"] and frame["display_rotation_metadata_degrees"] == 90 for frame in frames)
    assert all(frame["association"]["primary_frame_id"] == "original-primary" for frame in frames)
    assert all(frame["sampling"]["count_limit"] is None for frame in frames)
    assert all(frame["association"]["thumbnail_similarity_pruned"] is False for frame in frames)
    sampled = next(command for command in state["calls"] if "-vf" in command and "scale=384:-2" in command[command.index("-vf") + 1])
    assert sampled.index("-t") < sampled.index("-i")
    assert float(sampled[sampled.index("-t") + 1]) == 2.25  # elapsed duration, not start_time + duration
    assert not any(directory.exists() for directory in state["temporary_dirs"])
    assert page == unchanged


def test_early_close_keeps_only_requested_original_and_cleans_every_thumbnail(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch)
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    first = next(generator)
    # Even a caller keeping the iterator open has no thumbnail files left.
    assert all(not directory.exists() for directory in state["temporary_dirs"])
    generator.close()
    assert state["full_extractions"] == [first["source_pts"]]
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 1


def test_existing_pts_are_excluded_using_exact_rational_time_not_rounded_timestamp(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch)
    existing = tmp_path / "existing.jpg";Image.new("RGB", (128, 96), (20, 0, 0)).save(existing)
    page["frames"].append({"frame_id": "other-baseline", "path": str(existing),
                            "source_pts": 202500, "source_time_base": "1/90000", "timestamp_s": 99.})
    frames = list(frame_pool.iter_additional_page_frames(source, page, tmp_path / "out"))
    assert 2250 not in [frame["source_pts"] for frame in frames]
    assert len(frames) == 5


def test_no_registration_never_yields_or_decodes_full_size_candidate(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch, no_registration=True)
    assert list(frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")) == []
    assert not state["full_extractions"]
    assert not any(directory.exists() for directory in state["temporary_dirs"])
    audit = json.loads((tmp_path / "out/page-0001-candidate-scan.json").read_text())
    assert audit["association_is_heuristic"] is True
    assert len(audit["samples"]) == 8
    assert audit["samples"][0]["reason"] == "already_retained_exact_source_pts"
    rejected = audit["samples"][1]
    assert rejected["source_pts"] == 2250
    assert rejected["timestamp_s"] == .25
    assert rejected["accepted"] is False
    assert rejected["evidence"]["same_page"] is False


def _printed_page(digit):
    image = Image.new("RGB", (768, 1368), "white")
    draw = ImageDraw.Draw(image)
    for row in range(50, 1280, 45):
        draw.text((40, row), "DOCUMENT HEADER AND ORGANIZATION " + str(row), fill="black")
        draw.line((40, row + 20, 720, row + 20), fill="black")
    draw.rectangle((400, 700, 426, 726), fill="white")
    draw.text((402, 702), digit, fill="black")
    return image


def test_real_small_digit_change_is_not_proof_of_thumbnail_redundancy():
    first, second = (_printed_page(digit) for digit in ("1", "8"))
    features = [frame_pool._features(image.resize((384, 684), Image.Resampling.LANCZOS))
                for image in (first, second)]
    thumbnail = frame_pool._pair_evidence(*features)
    assert thumbnail["method"] == "near_identical_image"
    assert thumbnail["mean_abs_difference"] < .014
    assert first.tobytes() != second.tobytes()
    registered = frame_pool._register_near_view(*features, thumbnail)
    assert registered["same_page"] is True
    assert registered["method"] == "ORB_RANSAC_registered_image"
    assert registered["thumbnail_similarity_method"] == "near_identical_image"
    assert registered["inliers"] >= 16


def test_featureless_near_identical_thumbnails_do_not_establish_same_leaf():
    features = [frame_pool._features(Image.new("RGB", (384, 684), color))
                for color in ("white", (254, 254, 254))]
    thumbnail = frame_pool._pair_evidence(*features)
    assert thumbnail["same_page"] and thumbnail["method"] == "near_identical_image"
    registered = frame_pool._register_near_view(*features, thumbnail)
    assert registered["same_page"] is False


def test_real_ffmpeg_small_digit_change_materializes_distinct_source_pixels(tmp_path):
    """Real decoding, feature registration and extraction, with no association stubs."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    for index, digit in enumerate(("1", "1", "8", "8")):
        _printed_page(digit).save(tmp_path / f"input-{index:02d}.png")
    source = tmp_path / "digit.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-framerate", "4",
                    "-i", str(tmp_path / "input-%02d.png"), "-c:v", "ffv1", str(source)], check=True)
    primary = tmp_path / "primary.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", "select='eq(pts,0)'", "-fps_mode", "vfr", "-frames:v", "1", "-update", "1",
                    str(primary)], check=True)
    page = {"page_id": "changed-digit-leaf", "start_s": 0., "end_s": 1., "selected_frame_id": "primary",
            "frames": [{"frame_id": "primary", "path": str(primary), "source_pts": 0,
                        "source_time_base": "1/1000", "timestamp_s": 0.}]}
    frames = list(frame_pool.iter_additional_page_frames(source, page, tmp_path / "out"))
    assert sorted(frame["source_pts"] for frame in frames) == [250, 500, 750]
    changed = next(frame for frame in frames if frame["source_pts"] == 500)
    assert changed["association"]["evidence"]["thumbnail_similarity_method"] == "near_identical_image"
    assert changed["association"]["evidence"]["same_page"] is True
    expected = tmp_path / "expected.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", "select='eq(pts,500)'", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2",
                    "-update", "1", str(expected)], check=True)
    assert Path(changed["path"]).read_bytes() == expected.read_bytes()
    baseline = next(frame for frame in frames if frame["source_pts"] == 250)
    assert Path(changed["path"]).read_bytes() != Path(baseline["path"]).read_bytes()


def test_exact_pts_mismatch_refuses_substitution_and_cleans_tempfiles(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(RuntimeError, match="exact source PTS"):
        list(frame_pool.iter_additional_page_frames(source, page, tmp_path / "out"))
    assert not list((tmp_path / "out").glob("*.jpg"))
    assert not any(directory.exists() for directory in state["temporary_dirs"])


def test_seek_timestamp_mismatch_falls_back_to_exact_decode_from_beginning(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch, mismatch_on_seek=True)
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    frame = next(generator)
    generator.close()
    assert state["full_extractions"] == [3250, 3250]
    assert frame["encoding"]["seek_fallback_used"] is True
    assert frame["encoding"]["input_seek_elapsed_s"] == 0.
    assert frame["encoding"]["exact_pts_confirmed"]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_interval_fails_before_decode(tmp_path, monkeypatch, interval):
    source, page, state = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="sample_interval"):
        next(frame_pool.iter_additional_page_frames(source, page, tmp_path / "out", sample_interval=interval))
    assert not state["calls"]


def test_changed_source_is_rejected_before_next_original_decode(tmp_path, monkeypatch):
    source, page, state = _setup(tmp_path, monkeypatch)
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    next(generator)
    source.write_bytes(b"changed-source")
    with pytest.raises(RuntimeError, match="source changed"):
        next(generator)
    assert len(state["full_extractions"]) == 1


def test_real_ffmpeg_tiny_source_preserves_source_start_pts_and_lazy_full_resolution(tmp_path, monkeypatch):
    """Actual FFmpeg decode/encoding; registration is separately tested/mocked."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    source = tmp_path / "offset.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=4:duration=2", "-c:v", "ffv1", "-output_ts_offset", "1.25", str(source)], check=True)
    primary = tmp_path / "primary.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", "select='eq(pts,1250)'", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2",
                    "-update", "1", str(primary)], check=True)
    page = {"page_id": "offset-leaf", "start_s": 0., "end_s": .625, "selected_frame_id": "baseline",
            "frames": [{"frame_id": "baseline", "path": str(primary), "source_pts": 1250,
                        "source_time_base": "1/1000", "timestamp_s": 0.}]}
    monkeypatch.setattr(frame_pool, "_pair_evidence", lambda a, b: {"same_page": True, "method": "ORB_RANSAC_registered_image",
                                                               "inliers": 30, "mean_abs_difference": .1})
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    first = next(generator)
    assert first["source_time_base"] == "1/1000"
    assert first["source_pts"] in [1500, 1750]
    assert first["timestamp_s"] == pytest.approx(first["source_pts"] / 1000 - 1.25)
    assert first["pixel_size"] == [96, 64]
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 1
    # Compare pixels with an independent exact-PTS decode of the actual source.
    expected = tmp_path / "expected.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", f"select='eq(pts,{first['source_pts']})'", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2",
                    "-update", "1", str(expected)], check=True)
    assert Path(first["path"]).read_bytes() == expected.read_bytes()
    remaining = list(generator)
    assert sorted(frame["source_pts"] for frame in [first, *remaining]) == [1500, 1750]


def test_real_ffmpeg_input_seek_preserves_offset_source_pts_and_identical_jpeg(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    source = tmp_path / "seek-offset.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=4:duration=2", "-c:v", "ffv1", "-output_ts_offset", "1.25", str(source)], check=True)
    primary = tmp_path / "primary.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", "select='eq(pts,2250)'", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2", "-update", "1", str(primary)], check=True)
    page = {"page_id": "seek-leaf", "start_s": 1., "end_s": 2., "selected_frame_id": "baseline",
            "frames": [{"frame_id": "baseline", "path": str(primary), "source_pts": 2250, "source_time_base": "1/1000"}]}
    monkeypatch.setattr(frame_pool, "_pair_evidence", lambda a, b: {"same_page": True, "method": "ORB_RANSAC_registered_image",
                                                               "inliers": 30, "mean_abs_difference": .1})
    generator = frame_pool.iter_additional_page_frames(source, page, tmp_path / "out")
    frame = next(generator);generator.close()
    assert frame["source_pts"] in {2500, 2750, 3000}
    assert frame["encoding"]["input_seek_elapsed_s"] > 0
    assert frame["encoding"]["seek_fallback_used"] is False
    expected = tmp_path / "expected.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts", "-i", str(source),
                    "-vf", f"select='eq(pts,{frame['source_pts']})'", "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2", "-update", "1", str(expected)], check=True)
    assert Path(frame["path"]).read_bytes() == expected.read_bytes()


def test_late_leaf_thumbnail_seek_preserves_source_pts_and_decodes_only_local_window(tmp_path, monkeypatch):
    """Thumbnail seeking is a speed change; actual source PTS remain authoritative."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg tools unavailable')
    source = tmp_path / 'late-offset.mkv'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    'testsrc2=size=96x64:rate=4:duration=6', '-c:v', 'ffv1', '-output_ts_offset', '1.25', str(source)], check=True)
    primary = tmp_path / 'primary.png'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-copyts', '-i', str(source),
                    '-vf', "select='eq(pts,5250)'", '-fps_mode', 'vfr', '-frames:v', '1', '-update', '1', str(primary)], check=True)
    page = {'page_id': 'late-leaf', 'start_s': 4., 'end_s': 5., 'selected_frame_id': 'primary',
            'frames': [{'frame_id': 'primary', 'path': str(primary), 'source_pts': 5250, 'source_time_base': '1/1000'}]}
    monkeypatch.setattr(frame_pool, '_pair_evidence', lambda a, b: {'same_page': True, 'method': 'ORB_RANSAC_registered_image',
                                                                 'inliers': 30, 'mean_abs_difference': .1})
    original_run = frame_pool._run
    commands = []
    def tracked_run(command):
        commands.append(command)
        return original_run(command)
    monkeypatch.setattr(frame_pool, '_run', tracked_run)
    frames = list(frame_pool.iter_additional_page_frames(source, page, tmp_path / 'out'))
    assert sorted(frame['source_pts'] for frame in frames) == [5500, 5750, 6000]
    assert all(frame['sampling']['input_seek_elapsed_s'] == 3. for frame in frames)
    assert all(frame['sampling']['seek_fallback_used'] is False for frame in frames)
    assert all(frame['sampling']['exact_source_timebase_confirmed'] for frame in frames)
    sample_command = next(command for command in commands if '-vf' in command and 'scale=384:-2' in command[command.index('-vf') + 1])
    assert sample_command.index('-ss') < sample_command.index('-i')
    assert float(sample_command[sample_command.index('-t') + 1]) == 2.25
    # Independent no-seek full-resolution decode must have identical output.
    for frame in frames:
        expected = tmp_path / ('expected-' + str(frame['source_pts']) + '.jpg')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-copyts', '-i', str(source),
                        '-vf', f"select='eq(pts,{frame['source_pts']})'", '-fps_mode', 'vfr', '-frames:v', '1', '-q:v', '2',
                        '-update', '1', str(expected)], check=True)
        assert Path(frame['path']).read_bytes() == expected.read_bytes()
