from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from sever.agents import AnalysisError
from sever.image_detail import prepare_detail_panels, prepare_orientation_panels


def pattern(size):
    image = Image.new("RGB", size)
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), ((x*11)%256, (y*7)%256, ((x+y)*5)%256))
    return image


def rotate(image, degrees):
    return image.copy() if degrees == 0 else image.transpose(
        {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
         270: Image.Transpose.ROTATE_90}[degrees])


def jpeg(image):
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("size", [(13, 19), (1, 1), (1, 3), (3, 1), (2, 2), (3, 5)])
def test_exact_upright_quarter_pixels_source_bounds_and_inverse_geometry(tmp_path, size, rotation):
    source = tmp_path/"source.png"
    original = pattern(size)
    original.save(source)
    frame = {"path": str(source), "frame_id": "actual-frame", "timestamp_s": 1.25}
    uploads, proof = prepare_detail_panels(frame, tmp_path/"evidence", rotation=rotation)
    assert len(uploads) == len(proof["panels"]) == 5
    assert proof["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert proof["source_pixel_size"] == list(size)
    assert proof["source_unchanged_during_preparation"] is True
    assert proof["timestamp_s"] == 1.25 and proof["frame_id"] == "actual-frame"
    upright = rotate(original, rotation)
    boxes = []
    for row, vertical in enumerate(((0, max(1, round(.6*upright.height))),
                                    (min(upright.height-1, round(.4*upright.height)), upright.height))):
        for column, horizontal in enumerate(((0, max(1, round(.6*upright.width))),
                                             (min(upright.width-1, round(.4*upright.width)), upright.width))):
            box = (horizontal[0], vertical[0], horizontal[1], vertical[1])
            boxes.append((f"detail_{row}_{column}_r{rotation}", box))
    expected = {f"context_r{rotation}": jpeg(upright)}
    expected.update({identifier: jpeg(upright.crop(box)) for identifier, box in boxes})
    covered = set()
    for identifier, data in uploads:
        panel = proof["panels"][identifier]
        assert data == expected[identifier]
        assert data == Path(panel["upload_path"]).read_bytes()
        assert sha256(data).hexdigest() == panel["upload_sha256"]
        assert Image.open(BytesIO(data)).size == tuple(panel["upload_pixel_size"])
        assert panel["frame_id"] == frame["frame_id"]
        left, top, right, bottom = panel["source_bbox_pixels_top_left"]
        assert 0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]
        assert panel["bbox"] == [left/size[0], top/size[1], (right-left)/size[0], (bottom-top)/size[1]]
        if identifier.startswith("detail_"):
            assert panel["prep"]["native_source_detail"] is True
            assert panel["prep"]["resize"] == "none"
            covered.update((x, y) for y in range(top, bottom) for x in range(left, right))
        matrix, inverse = panel["source_to_upload_pixels"], panel["upload_to_source_pixels"]
        mapped = []
        for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
            u, v = (matrix[axis][0]*x+matrix[axis][1]*y+matrix[axis][2] for axis in (0, 1))
            mapped.append((u, v))
            restored = [inverse[axis][0]*u+inverse[axis][1]*v+inverse[axis][2] for axis in (0, 1)]
            assert restored == pytest.approx((x, y))
        width, height = panel["upload_pixel_size"]
        assert set(mapped) == {(0, 0), (width, 0), (0, height), (width, height)}
    assert covered == {(x, y) for y in range(size[1]) for x in range(size[0])}
    assert not list((tmp_path/"evidence"/"image-panels").glob("*.tmp"))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_only_context_is_resized_while_native_detail_dimensions_are_preserved(tmp_path, rotation):
    source = tmp_path/"large.png"
    Image.new("RGB", (1703, 1101), "white").save(source)
    _, proof = prepare_detail_panels({"path": str(source)}, tmp_path/"evidence", rotation=rotation)
    context = proof["panels"][f"context_r{rotation}"]
    assert max(context["upload_pixel_size"]) == 1024
    for identifier, panel in proof["panels"].items():
        if identifier.startswith("detail_"):
            left, top, right, bottom = panel["source_bbox_pixels_top_left"]
            expected = [bottom-top, right-left] if rotation % 180 else [right-left, bottom-top]
            assert panel["upload_pixel_size"] == expected
        matrix, inverse = panel["source_to_upload_pixels"], panel["upload_to_source_pixels"]
        for x, y in ((0, 0), (17.25, 36.75), (1703, 1101)):
            u, v = (matrix[axis][0]*x+matrix[axis][1]*y+matrix[axis][2] for axis in (0, 1))
            assert [inverse[axis][0]*u+inverse[axis][1]*v+inverse[axis][2]
                    for axis in (0, 1)] == pytest.approx((x, y))


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_alpha_is_composited_on_white_before_rotation_and_jpeg(tmp_path, mode):
    source = tmp_path/"alpha.png"
    if mode == "RGBA":
        original = Image.new("RGBA", (9, 11), (10, 80, 150, 128))
    elif mode == "LA":
        original = Image.new("LA", (9, 11), (80, 128))
    else:
        original = Image.new("P", (9, 11), 0)
        original.putpalette([10, 80, 150]+[0, 0, 0]*255)
        original.info["transparency"] = 0
    original.save(source)
    rgba = original.convert("RGBA")
    expected = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    uploads, proof = prepare_detail_panels({"path": str(source)}, tmp_path/"evidence", rotation=270)
    assert uploads[0][1] == jpeg(rotate(expected, 270))
    assert all(Image.open(BytesIO(data)).mode == "RGB" for _, data in uploads)
    assert proof["source_sha256"] == sha256(source.read_bytes()).hexdigest()


def test_exif_orientation_is_recorded_and_explicit_rotation_is_the_only_applied_orientation(tmp_path):
    source = tmp_path/"orientation.png"
    original = pattern((13, 19))
    exif = Image.Exif()
    exif[274] = 6
    original.save(source, exif=exif)
    uploads, proof = prepare_detail_panels({"path": str(source)}, tmp_path/"evidence", rotation=0)
    assert proof["source_exif_orientation"] == 6
    assert uploads[0][1] == jpeg(original)
    assert all(not panel["prep"]["exif_orientation_applied"] for panel in proof["panels"].values())


@pytest.mark.parametrize("rotation", [-90, 45, 360, "90", True, None])
def test_invalid_rotation_fails_before_reading_or_writing_pixels(tmp_path, rotation):
    with pytest.raises(ValueError):
        prepare_detail_panels({"path": str(tmp_path/"missing.png")}, tmp_path/"evidence", rotation=rotation)
    assert not (tmp_path/"evidence").exists()


def test_expected_source_hash_mismatch_is_rejected_before_any_panel_write(tmp_path):
    source = tmp_path/"source.png"
    pattern((13, 19)).save(source)
    with pytest.raises(AnalysisError, match="SHA"):
        prepare_detail_panels({"path": str(source), "source_sha256": "0"*64},
                              tmp_path/"evidence", rotation=90)
    assert not (tmp_path/"evidence").exists()


def test_source_mutation_during_panel_preparation_is_rejected(tmp_path, monkeypatch):
    source = tmp_path/"source.png"
    pattern((13, 19)).save(source)
    original_save = Image.Image.save
    mutated = False
    def mutate(self, destination, *args, **kwargs):
        nonlocal mutated
        result = original_save(self, destination, *args, **kwargs)
        if isinstance(destination, BytesIO) and not mutated:
            source.write_bytes(b"changed-after-source-decoding")
            mutated = True
        return result
    monkeypatch.setattr(Image.Image, "save", mutate)
    with pytest.raises(AnalysisError, match="changed during"):
        prepare_detail_panels({"path": str(source)}, tmp_path/"evidence", rotation=180)


@pytest.mark.parametrize("content", [None, b"not-an-image"])
def test_missing_or_invalid_source_fails_without_model_or_panel_write(tmp_path, content):
    source = tmp_path/"source.png"
    if content is not None:
        source.write_bytes(content)
    with pytest.raises(AnalysisError, match="no model call made"):
        prepare_detail_panels({"path": str(source)}, tmp_path/"evidence", rotation=0)
    assert not (tmp_path/"evidence").exists()


@pytest.mark.parametrize("size", [(1, 1), (13, 19)])
@pytest.mark.parametrize("alpha", [False, True])
def test_orientation_contexts_share_actual_pixels_alpha_and_source_provenance(tmp_path, size, alpha):
    source = tmp_path/"source.png"
    original = pattern(size)
    if alpha:
        original = original.convert("RGBA")
        original.putalpha(128)
    original.save(source)
    expected = original.convert("RGB") if not alpha else Image.alpha_composite(
        Image.new("RGBA", size, "white"), original).convert("RGB")
    uploads, proof = prepare_orientation_panels({"path": str(source)}, tmp_path/"evidence")
    assert len(uploads) == len(proof["panels"]) == 4
    assert proof["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    for (identifier, data), rotation in zip(uploads, (0, 90, 180, 270), strict=True):
        panel = proof["panels"][identifier]
        assert identifier == f"context_r{rotation}"
        assert data == jpeg(rotate(expected, rotation))
        assert data == Path(panel["upload_path"]).read_bytes()
        assert panel["bbox"] == [0, 0, 1, 1]
        assert panel["source_bbox_pixels_top_left"] == [0, 0, *size]
        assert panel["prep"]["native_source_detail"] is False
