"""Lazy additional views of an existing leaf, with exact source PTS provenance.

No resegmentation or frame-count limit. Small temporary JPEGs determine paper
association/diversity; full-size JPEGs are decoded individually only on demand.
Association and quality are visual heuristics, not calibrated legibility scores.
Thumbnail similarity never establishes full-resolution pixel redundancy.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterator

from PIL import Image

from .video import _features, _pair_evidence, _registered_pair_evidence, _run, _showinfo, _tool


def _small_feature(path):
    with Image.open(path) as image:
        height = max(2, round(image.height * 384 / image.width / 2) * 2)
        return _features(image.convert("RGB").resize((384, height), Image.Resampling.LANCZOS))


def _register_near_view(primary, candidate, thumbnail_proof):
    """Require paper/text correspondence even when thumbnail averages agree.

    The segmenter's near-image shortcut alone also matches blank images. Run
    registration on the actual feature pixels/descriptors before admitting a
    distinct PTS; only the downstream native-pixel policy can skip redundancy.
    """
    return _registered_pair_evidence(primary, candidate, {
        **thumbnail_proof, "thumbnail_similarity_method": "near_identical_image",
        "method": "ORB_RANSAC_registered_image"})


def _rank_candidates(source, page, ffmpeg, stream, start, rotation, interval, *, audit=None):
    primary = next((frame for frame in page.get("frames", []) if frame.get("frame_id") == page.get("selected_frame_id")), None)
    if not primary or not Path(primary.get("path", "")).is_file():
        raise ValueError("Additional views require the existing selected primary frame")
    primary_feature = _small_feature(primary["path"])
    existing_times = set()
    for frame in page.get("frames", []):
        if frame.get("source_pts") is not None and frame.get("source_time_base"):
            existing_times.add(int(frame["source_pts"]) * Fraction(frame["source_time_base"]))
    # Seek close to this leaf rather than decoding every earlier leaf again.
    # Keep source PTS and validate observed timestamps; fall back to a full
    # prefix decode if seeking cannot establish the same timestamp contract.
    lower, upper = start + float(page["start_s"]), start + float(page["end_s"])
    expression = f"gte(t,{lower:.12f})*lt(t,{upper:.12f})*(isnan(prev_selected_t)+gte(t-prev_selected_t,{interval:.12f}))"
    with tempfile.TemporaryDirectory(prefix="sever-frame-pool-") as temporary:
        root = Path(temporary)
        seek = max(0., float(page["start_s"]) - 1.)
        sampling_fallback = False
        for attempt_seek in ([seek, 0.] if seek else [0.]):
            for leftover in root.glob("sample-*.jpg"):
                leftover.unlink()
            command = [ffmpeg, "-hide_banner", "-nostdin", "-copyts"]
            if attempt_seek:
                command.extend(["-ss", f"{attempt_seek:.12f}"])
            command.extend(["-t", f"{float(page['end_s']) - attempt_seek + interval:.12f}",
                            "-i", str(source), "-map", f"0:{stream['index']}", "-an", "-vf",
                            f"select='{expression}',scale=384:-2,showinfo", "-fps_mode", "vfr", "-q:v", "5",
                            str(root / "sample-%06d.jpg")])
            try:
                sampled = _run(command)
                samples, time_base = _showinfo(sampled.stderr)
                thumbnails = sorted(root.glob("sample-*.jpg"))
                if len(samples) != len(thumbnails):
                    raise RuntimeError("Additional thumbnail images do not match source PTS observations")
                if samples and not time_base:
                    raise RuntimeError("Additional sampling did not expose an exact source time base")
                if samples and stream.get("time_base") and Fraction(time_base) != Fraction(stream["time_base"]):
                    raise RuntimeError("Additional sampling changed the source time base")
                if any(not lower - 1e-8 <= float(sample["source_pts"] * Fraction(time_base)) < upper + 1e-8 for sample in samples):
                    raise RuntimeError("Selected thumbnail PTS lies outside the leaf interval")
                if attempt_seek and not samples:
                    raise RuntimeError("Seeked sampling returned no observed source frames")
            except RuntimeError:
                if not attempt_seek:
                    raise
                sampling_fallback = True
                continue
            sampling_seek = attempt_seek
            break
        accepted = []
        seen_pts = set()
        observations = []
        if audit is not None:
            audit.update(page_id=page["page_id"], primary_frame_id=primary["frame_id"],
                         leaf_interval_s=[page["start_s"], page["end_s"]], source_time_base=time_base,
                         input_seek_elapsed_s=sampling_seek, seek_fallback_used=sampling_fallback,
                         samples=observations, association_is_heuristic=True)
        for sample, thumbnail in zip(samples, thumbnails):
            point = sample["source_pts"] * Fraction(time_base)
            elapsed = float(point) - start
            if point in existing_times or point in seen_pts:
                observations.append({"source_pts": sample["source_pts"], "timestamp_s": elapsed,
                                     "accepted": False, "reason": "already_retained_exact_source_pts"})
                continue
            seen_pts.add(point)
            if not float(page["start_s"]) - 1e-8 <= elapsed < float(page["end_s"]) + 1e-8:
                raise RuntimeError("Selected thumbnail PTS lies outside the leaf interval")
            with Image.open(thumbnail) as image:
                feature = _features(image)
            proof = _pair_evidence(primary_feature, feature)
            if proof.get("same_page") and proof.get("method") == "near_identical_image":
                proof = _register_near_view(primary_feature, feature, proof)
            # Both thumbnail-similar and visibly changed candidates require
            # registered paper/text correspondence. A small changed digit must
            # reach full-resolution reading even when its thumbnail looks equal.
            associated = bool(proof.get("same_page") and proof.get("method") == "ORB_RANSAC_registered_image")
            observations.append({"source_pts": sample["source_pts"], "timestamp_s": elapsed,
                                 "accepted": associated,
                                 "reason": "registered_same_page" if associated else "unconfirmed_or_partial_page_association",
                                 "evidence": proof})
            if not associated:
                continue
            accepted.append((sample, feature, proof))
        if not accepted:
            return []
        sharpest = max([primary_feature.quality.get("sharpness_laplacian_variance", 0),
                        *(feature.quality.get("sharpness_laplacian_variance", 0) for _, feature, _ in accepted)]) or 1.
        scored = []
        for sample, feature, proof in accepted:
            relative_quality = math.sqrt(max(0., feature.quality.get("sharpness_laplacian_variance", 0)) / sharpest)
            novelty = max(0., float(proof.get("mean_abs_difference", 0)))
            association = proof.get("regional_agreement") or {}
            strength = float(association.get("minimum_edge_agreement", 1.))
            score = relative_quality * strength
            scored.append((score, sample, feature, proof, relative_quality, novelty))
        scored.sort(key=lambda row: (-row[0], row[1]["source_pts"]))
        candidates = []
        for score, sample, feature, proof, relative_quality, novelty in scored:
            pts = sample["source_pts"]
            seconds = float(pts * Fraction(time_base))
            quality = dict(feature.quality)
            quality.update(relative_selection_score=round(score, 5), relative_quality_score=round(relative_quality, 5),
                           primary_image_difference=round(novelty, 5),
                           additional_view_ranking="within-leaf relative sharpness times registered edge agreement; no accuracy calibration")
            candidates.append({"source_pts": pts, "source_time_base": time_base,
                               "source_pts_time_s": seconds, "timestamp_s": max(0., seconds - start),
                               "display_rotation_metadata_degrees": rotation, "display_rotation_applied": True,
                               "quality": quality,
                               "association": {"primary_frame_id": primary["frame_id"], "page_id": page["page_id"],
                                               "method": "registered_primary_paper_text_and_distinct_source_pts", "evidence": proof,
                                               "thumbnail_similarity_pruned": False,
                                               "leaf_interval_s": [page["start_s"], page["end_s"]],
                                               "boundaries_are_approximate": True},
                               "sampling": {"interval_s": interval, "thumbnail_width_px": 384,
                                            "thumbnail_encoding": "jpeg", "thumbnail_ffmpeg_qscale": 5,
                                            "count_limit": None, "input_seek_elapsed_s": sampling_seek,
                                            "seek_fallback_used": sampling_fallback,
                                            "exact_source_timebase_confirmed": True}})
        # TemporaryDirectory removes every thumbnail before the first yield,
        # including when consumers stop without explicitly closing the iterator.
        return candidates


def iter_additional_page_frames(source: Path, page: dict, out: Path, *, sample_interval: float = .25) -> Iterator[dict]:
    """Yield informative same-leaf full-size frames in quality/novelty order.

    Existing page/frame metadata remains unchanged. Candidate enumeration is
    unlimited; only requested yields are materialized. Thumbnails are removed
    before yielding, and each extraction's temporary directory closes before
    its yield. Originals seek one second before the known elapsed timestamp,
    preserve source PTS, and fall back to decoding from the beginning if exact
    PTS/timebase validation fails. ``generator.close()`` is safe at any point.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("sample_interval must be finite and positive")
    lower, upper = float(page["start_s"]), float(page["end_s"])
    if not all(math.isfinite(value) for value in (lower, upper)) or lower < 0 or upper <= lower:
        raise ValueError("A finite nonempty leaf interval is required")
    ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
    metadata = json.loads(_run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)]).stdout)
    stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video" and
                   not s.get("disposition", {}).get("attached_pic")), None)
    if stream is None:
        raise ValueError("Source has no usable video stream")
    start = float(stream.get("start_time") or metadata.get("format", {}).get("start_time") or 0)
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s),
                    float(stream.get("tags", {}).get("rotate", 0)))
    stat = source.stat()
    identity = {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    source_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    audit = {"source_identity": identity}
    candidates = _rank_candidates(source, page, ffmpeg, stream, start, rotation, sample_interval, audit=audit)
    safe_page_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(page["page_id"]))
    out.mkdir(parents=True, exist_ok=True)
    # Keep exact PTS and reasons even for rejected mixed-sheet candidates;
    # these are reviewable image heuristics, not proof of complete coverage.
    (out / (safe_page_id + "-candidate-scan.json")).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    for candidate in candidates:
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Video source changed after additional-view sampling")
        pts = candidate["source_pts"]
        frame_id = f"{safe_page_id}-extra-{source_id}-pts-{pts}"
        target = out / (frame_id + ".jpg")
        with tempfile.TemporaryDirectory(prefix="sever-frame-exact-") as temporary:
            original = Path(temporary) / "exact.jpg"
            seek = max(0., candidate["timestamp_s"] - 1.)
            fallback, exact = False, False
            for attempt_seek in ([seek, 0.] if seek > 0 else [0.]):
                command = [ffmpeg, "-hide_banner", "-nostdin", "-copyts"]
                if attempt_seek:
                    command.extend(["-ss", f"{attempt_seek:.12f}"])
                command.extend(["-i", str(source), "-map", f"0:{stream['index']}", "-an", "-vf", f"select='eq(pts,{pts})',showinfo",
                                "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2", "-update", "1", str(original)])
                try:
                    decoded = _run(command)
                    actual, actual_time_base = _showinfo(decoded.stderr)
                    exact = bool(original.is_file() and len(actual) == 1 and actual[0]["source_pts"] == pts and
                                 actual_time_base and Fraction(actual_time_base) == Fraction(candidate["source_time_base"]))
                except RuntimeError:
                    exact = False
                if exact:
                    actual_seek = attempt_seek
                    break
                original.unlink(missing_ok=True)
                fallback = seek > 0
            if not exact:
                raise RuntimeError("Additional original did not match exact source PTS; refusing approximate substitution")
            with Image.open(original) as image:
                pixel_size = list(image.size)
                if image.format != "JPEG":
                    raise RuntimeError("Additional original was not the requested JPEG encoding")
            shutil.move(str(original), target)
        yield {**candidate, "frame_id": frame_id, "path": str(target), "pixel_size": pixel_size,
               "encoding": {"format": "jpeg", "ffmpeg_qscale": 2, "full_source_resolution": True,
                            "exact_pts_confirmed": True, "input_seek_elapsed_s": actual_seek,
                            "seek_fallback_used": fallback, "file_size_bytes": target.stat().st_size},
               "source_identity": identity}
