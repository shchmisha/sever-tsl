"""Local, bounded frame selection from document-flipping videos.

Segmentation and quality numbers are visual heuristics, not calibrated confidence.
Initial selection retains a primary plus up to two seeds. The pipeline lazily
requests additional informative views through frame_pool without a leaf cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image


@dataclass
class _Feature:
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    quality: dict


def analyze_image_quality(image: Image.Image) -> dict:
    """Geometry/exposure/possible-skin indicators; unknown remains null with a reason."""
    small = image.convert("RGB").copy()
    small.thumbnail((640, 640))
    rgb = np.asarray(small)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    near_white = (gray >= 250) & (hsv[:, :, 1] <= 20)
    grid = [float(part.mean()) for row in np.array_split(near_white, 6, axis=0)
            for part in np.array_split(row, 6, axis=1) if part.size]
    brightest = int(np.argmax(grid))
    skin = (((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) &
            (hsv[:, :, 1] >= 45) & (hsv[:, :, 1] <= 210) & (hsv[:, :, 2] >= 90))
    mask = ((cv2.GaussianBlur(gray, (5, 5), 0) > 140).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geometry = {"quadrilateral_normalized": None, "opposite_edge_length_ratios": None,
                "perspective_indicator": None, "clipped_edges": None,
                "reason": "no sufficiently large paper outline detected"}
    full_page, reason = None, "paper extent could not be established"
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour) / (h * w)
        if area > .18:
            points = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True).reshape(-1, 2)
            clipped = sorted({edge for x, y in contour.reshape(-1, 2) for edge, condition in
                              [("left", x <= 1), ("right", x >= w - 2), ("top", y <= 1), ("bottom", y >= h - 2)] if condition})
            geometry["clipped_edges"] = clipped
            if clipped:
                full_page, reason = False, "bright paper-like outline reaches image boundary; complete extent unconfirmed"
            if len(points) == 4 and cv2.isContourConvex(points.reshape(-1, 1, 2)):
                sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
                ratios = [float(min(sides[i], sides[i + 2]) / max(1., max(sides[i], sides[i + 2]))) for i in (0, 1)]
                geometry.update(quadrilateral_normalized=(points / [w, h]).tolist(),
                                opposite_edge_length_ratios=ratios, perspective_indicator=1 - min(ratios), reason=None)
                if not clipped and all(3 < x < w - 4 and 3 < y < h - 4 for x, y in points):
                    full_page, reason = True, "four paper-like outline corners lie inside the image; internal occlusions are not excluded"
            else:
                geometry["reason"] = "paper-like outline is not a reliable four-corner polygon"
    return {"full_page_visible": full_page, "full_page_visibility_reason": reason,
            "paper_geometry": geometry,
            "exposure": {"near_white_fraction": float(near_white.mean()), "maximum_local_near_white_fraction": max(grid),
                         "brightest_cell_normalized_bbox": [(brightest % 6) / 6, (brightest // 6) / 6, 1 / 6, 1 / 6],
                         "glare_confirmed": None, "reason": "white paper and specular glare cannot be distinguished by these indicators"},
            "possible_hand_occlusion": {"skin_colored_area_fraction": float(skin.mean()), "hand_occlusion_confirmed": None,
                                        "reason": "HSV skin-color heuristic also matches printed colors, desk and lighting"},
            "quality_limitations": ["geometry_is_a_bright_outline_heuristic", "four_corners_do_not_prove_no_internal_occlusion",
                                    "indicators_do_not_measure_field_accuracy", "white_paper_is_not_proof_of_overexposure"]}


def _features(image: Image.Image) -> _Feature:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Exclude most desk above the paper, without clipping the original saved frame.
    h, w = gray.shape
    gray = gray[int(h * .18):int(h * .97), int(w * .035):int(w * .965)]
    mask = (cv2.GaussianBlur(gray, (5, 5), 0) > 115).astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=1400, fastThreshold=9, edgeThreshold=12)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    paper = mask > 0
    quality = {
        "sharpness_laplacian_variance": round(float(lap[paper].var()) if paper.any() else 0., 3),
        "bright_area_fraction": round(float(paper.mean()), 5),
        "gray_standard_deviation": round(float(gray.std()), 3),
        "detected_feature_count": len(keypoints),
        "method": "downsampled document-area indicators; compare within a page only",
        "limitations": [],
    }
    if len(keypoints) < 20:
        quality["limitations"].append("little_visible_text_or_edge_detail")
    if quality["bright_area_fraction"] < .25:
        quality["limitations"].append("little_bright_paper_visible")
    quality.update(analyze_image_quality(image))
    return _Feature(gray, list(keypoints), descriptors, quality)


def _regional_edge_agreement(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> dict:
    """Expose newly visible text that a homography of a shared fragment misses.

    Each image has its own paper mask: intersecting the masks would hide a new
    sheet exposed over what used to be desk. Small glyph changes do not reject a
    view. The indicators are image-association heuristics, never OCR confidence.
    """
    kernel = np.ones((3, 3), np.uint8)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edges = [(cv2.Canny(gray, 45, 100) > 0) &
             (cv2.dilate((gray > 115).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
             for gray in (first, second)]
    expanded = [cv2.dilate(edge.astype(np.uint8), kernel) > 0 for edge in edges]
    directions = []
    for edge, other in zip(edges, reversed(expanded)):
        total = int((edge & valid).sum())
        support = float((edge & other & valid).sum() / max(1, total))
        inconsistent = 0
        cells = []
        for erow, orow, vrow in zip(np.array_split(edge, 4), np.array_split(other, 4), np.array_split(valid, 4)):
            for part, peer, mask in zip(np.array_split(erow, 3, axis=1), np.array_split(orow, 3, axis=1),
                                        np.array_split(vrow, 3, axis=1)):
                count = int((part & mask).sum())
                agreement = float((part & peer & mask).sum() / max(1, count))
                # Ignore isolated outlines and tiny changed digits; reject only
                # a substantial textured region inconsistent with registration.
                poor = count >= 100 and agreement < .55
                if poor:
                    inconsistent += count
                cells.append({"edge_count": count, "agreement": round(agreement, 5), "inconsistent": poor})
        directions.append({"edge_count": total, "agreement": round(support, 5),
                           "inconsistent_edge_fraction": round(inconsistent / max(1, total), 5), "cells": cells})
    fraction = max(direction["inconsistent_edge_fraction"] for direction in directions)
    return {"directions": directions, "inconsistent_edge_fraction": fraction,
            "minimum_edge_agreement": min(direction["agreement"] for direction in directions),
            "partial_page_overlap": fraction >= .18,
            "method": "bidirectional_4x3_registered_paper_edges"}


def _registered_pair_evidence(a: _Feature, b: _Feature, evidence: dict) -> dict:
    """Register paper features and require spatially consistent image evidence."""
    evidence = {**evidence, "same_page": False}
    if a.descriptors is None or b.descriptors is None:
        return evidence
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a.descriptors, b.descriptors, k=2)
    matches = [x for pair in pairs if len(pair) == 2 for x, y in [pair] if x.distance < .72 * y.distance]
    unique = {}
    for match in sorted(matches, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if len(matches) < 12:
        return evidence
    pa = np.float32([a.keypoints[m.queryIdx].pt for m in matches])
    pb = np.float32([b.keypoints[m.trainIdx].pt for m in matches])
    matrix, inlier_mask = cv2.findHomography(pb, pa, cv2.RANSAC, 3.5)
    if matrix is None or inlier_mask is None:
        return evidence
    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    fraction = count / max(1, min(len(a.keypoints), len(b.keypoints)))
    points = pa[inliers]
    coverage = hull_coverage = 0.
    if len(points):
        extent = points.max(axis=0) - points.min(axis=0)
        area = a.gray.shape[0] * a.gray.shape[1]
        coverage = float(extent.prod() / area)
        if len(points) >= 3:
            hull_coverage = float(cv2.contourArea(cv2.convexHull(points)) / area)
    warped = cv2.warpPerspective(b.gray, matrix, (a.gray.shape[1], a.gray.shape[0]))
    valid = cv2.warpPerspective(np.ones_like(b.gray), matrix, (a.gray.shape[1], a.gray.shape[0])) > 0
    overlap = valid & (a.gray > 115)
    difference = np.abs(a.gray.astype(np.float32) - warped)
    residual = float((difference[overlap] if overlap.any() else difference).mean() / 255)
    geometry_matches = bool(count >= 16 and fraction >= .075 and coverage >= .08)
    regional = _regional_edge_agreement(a.gray, warped, valid) if geometry_matches else None
    # Exposure can change while all printed edges remain aligned. Mean gray
    # residual alone fragmented a stationary sheet as auto-exposure changed.
    exposure_rescue = bool(regional and residual < .22 and regional["minimum_edge_agreement"] >= .90 and
                           regional["inconsistent_edge_fraction"] < .05)
    registered = bool(geometry_matches and (residual < .11 or exposure_rescue))
    evidence.update(inliers=count, inlier_fraction=round(fraction, 5), feature_coverage=round(coverage, 5),
                    feature_hull_coverage=round(hull_coverage, 5),
                    registered_difference=round(residual, 5), method="ORB_RANSAC_registered_image",
                    regional_agreement=regional, exposure_change_rescued=bool(exposure_rescue and residual >= .11), partial_page_overlap=bool(regional and regional["partial_page_overlap"]),
                    same_page=bool(registered and not regional["partial_page_overlap"]))
    return evidence


def _pair_evidence(a: _Feature, b: _Feature) -> dict:
    """Registered text/image evidence, with desk features masked at detection."""
    ga = cv2.resize(a.gray, (128, 192)).astype(np.float32)
    gb = cv2.resize(b.gray, (128, 192)).astype(np.float32)
    raw_difference = float(np.abs(ga - gb).mean() / 255)
    evidence = {"mean_abs_difference": round(raw_difference, 5), "inliers": 0,
                "inlier_fraction": 0., "feature_coverage": 0., "same_page": False}
    # Exact/near-exact repeat; moving edges otherwise require registration below.
    ea = cv2.Canny(ga.astype(np.uint8), 45, 100) > 0
    eb = cv2.Canny(gb.astype(np.uint8), 45, 100) > 0
    union = np.logical_or(ea, eb).sum()
    edge_overlap = float(np.logical_and(ea, eb).sum() / union) if union else 1.
    if raw_difference < .014 and edge_overlap > .52:
        evidence["same_page"] = True
        evidence["method"] = "near_identical_image"
        return evidence
    return _registered_pair_evidence(a, b, evidence)


def _sample_pair(features: list[_Feature], first: int, second: int, cache: dict) -> dict:
    key = (min(first, second), max(first, second))
    if key not in cache:
        cache[key] = _pair_evidence(features[key[0]], features[key[1]])
    return cache[key]


def _segment_samples(features: list[_Feature], *, evidence_cache: dict | None = None) -> tuple[list[list[int]], list[int]]:
    """Confirm stable leaves, retain brief printed views and audit turning samples.

    A single printed sample can be a short document display. Preserve it as an
    uncertain candidate unless registered evidence shows a partial page turn;
    do not silently discard every sample lacking a second matching neighbour.
    """
    if not features:
        return [], []
    cache = evidence_cache if evidence_cache is not None else {}
    groups: list[list[int]] = []
    current = [0]
    anchor = 0
    pending: list[int] = []
    discarded: list[int] = []

    def preserve_short(indices: list[int], previous: int, following: int | None):
        for index in indices:
            peers = [previous] + ([following] if following is not None else [])
            partial = any(_sample_pair(features, index, peer, cache).get("partial_page_overlap", False) for peer in peers)
            quality = features[index].quality
            printed = quality.get("detected_feature_count", len(features[index].keypoints)) >= 20 and quality.get("bright_area_fraction", 0) >= .25
            if printed and not partial:
                groups.append([index])
            else:
                discarded.append(index)

    for index in range(1, len(features)):
        if _sample_pair(features, anchor, index, cache)["same_page"]:
            if pending:
                # A -> brief B -> A remains three ordered candidates when B has
                # independent printed content; it is not swallowed into A.
                groups.append(current)
                count_before = len(groups)
                preserve_short(pending, anchor, index)
                # A blurred/empty transition followed by the same leaf does
                # not create two candidates; a retained short leaf does.
                current = groups.pop() if len(groups) == count_before else []
                pending = []
            current.append(index)
            if features[index].quality["sharpness_laplacian_variance"] > features[anchor].quality["sharpness_laplacian_variance"]:
                anchor = index
            continue
        pending.append(index)
        if len(pending) >= 2 and _sample_pair(features, pending[-2], index, cache)["same_page"]:
            groups.append(current)
            preserve_short(pending[:-2], anchor, pending[-2])
            current = pending[-2:]
            anchor = max(current, key=lambda i: features[i].quality["sharpness_laplacian_variance"])
            pending = []
    groups.append(current)
    if pending:
        preserve_short(pending, anchor, None)
    return groups, discarded


def _confirm_short_candidates(source, stream, samples, features, groups, ffmpeg, start, interval, temporary):
    """Verify coarse singleton candidates with a denser local temporal window.

    A printed fragment during a turn is not automatically a separate document.
    All rejected/unconfirmed candidates keep exact probe PTS and visual evidence.
    The source frame is retained only when independent nearby times provide
    stable, spatially distributed correspondence spanning at least 0.12 seconds.
    """
    kept, rejected, audits = [], [], {}
    time_base = Fraction(stream.get("time_base", "1/1"))
    radius = min(.3, max(.2, interval * .5))
    for group in groups:
        if len(group) != 1:
            kept.append(group)
            continue
        index = group[0]
        center = samples[index]["timestamp_s"]
        lower, upper = max(0., center - radius), center + radius
        seek = max(0., lower - .5)
        root = temporary / f"short-{index:06d}"
        root.mkdir()
        expression = (f"gte(t,{start + lower:.12f})*lt(t,{start + upper:.12f})*"
                      "(isnan(prev_selected_t)+gte(t-prev_selected_t,0.049999))")
        command = [ffmpeg, "-hide_banner", "-nostdin", "-copyts"]
        if seek:
            command.extend(["-ss", f"{seek:.12f}"])
        command.extend(["-t", f"{upper - seek + .1:.12f}", "-i", str(source), "-map", f"0:{stream['index']}",
                        "-an", "-vf", f"select='{expression}',scale=384:-2,showinfo", "-fps_mode", "vfr", "-q:v", "5",
                        str(root / "probe-%06d.jpg")])
        decoded = _run(command)
        observed, observed_tb = _showinfo(decoded.stderr)
        thumbnails = sorted(root.glob("probe-*.jpg"))
        if len(observed) != len(thumbnails) or (observed and (not observed_tb or Fraction(observed_tb) != time_base)):
            raise RuntimeError("Short-candidate probes do not match exact source timestamps")
        proof_rows, support_times = [], []
        for observation, thumbnail in zip(observed, thumbnails):
            elapsed = float(observation["source_pts"] * time_base) - start
            if not lower - 1e-8 <= elapsed < upper + 1e-8:
                raise RuntimeError("Short-candidate probe lies outside its source-time window")
            with Image.open(thumbnail) as image:
                probe = _features(image)
            proof = _pair_evidence(features[index], probe)
            if proof.get("same_page") and proof.get("method") == "near_identical_image":
                proof = _registered_pair_evidence(features[index], probe, proof)
            independent = observation["source_pts"] != samples[index]["source_pts"]
            # A small surviving corner of the old sheet cannot confirm a new
            # whole candidate. This remains a disclosed coverage heuristic.
            support = bool(independent and proof.get("same_page") and proof.get("feature_hull_coverage", 0) >= .2)
            if support:
                support_times.append(elapsed)
            proof_rows.append({"source_pts": observation["source_pts"], "timestamp_s": elapsed,
                               "supports_candidate": support, "evidence": proof})
        span = max(support_times) - min(support_times) if len(support_times) >= 2 else 0.
        confirmed = bool(len(support_times) >= 2 and span >= .12)
        audits[index] = {"confirmed": confirmed, "method": "dense_local_temporal_and_spatial_support",
                         "support_span_s": span, "source_time_base": str(time_base), "probes": proof_rows,
                         "reason": "stable_brief_printed_view" if confirmed else "no_stable_brief_view_confirmed",
                         "limitations": ["visual_stability_does_not_prove_legibility", "unconfirmed_glimpses_require_review"]}
        if confirmed:
            kept.append(group)
        else:
            rejected.append(index)
    merged = []
    rejected_set = set(rejected)
    for group in kept:
        intervening = set(range(merged[-1][-1] + 1, group[0])) if merged else set()
        if (merged and intervening and intervening <= rejected_set and
                _pair_evidence(features[merged[-1][-1]], features[group[0]])["same_page"]):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged, rejected, audits


def _tool(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"{name} is required for local video analysis")
    return binary


def _run(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed: {result.stderr[-3000:]}")
    return result


def _showinfo(log: str) -> tuple[list[dict], str | None]:
    entries = []
    time_base = None
    for line in log.splitlines():
        if "showinfo" not in line:
            continue
        tb = re.search(r"config in time_base:\s*(\d+/\d+)", line)
        if tb:
            time_base = tb.group(1)
        match = re.search(r"n:\s*(\d+).*?pts:\s*(-?\d+).*?pts_time:\s*([-+\d.eE]+)", line)
        if match:
            entries.append({"sample_index": int(match.group(1)), "source_pts": int(match.group(2)),
                            "source_pts_time_s": float(match.group(3))})
    return entries, time_base


def _pts_selection_expression(points: list[int]) -> str:
    """Match exact source PTS with logarithmic expression-parser depth.

    FFmpeg's expression parser recursively descends the right side of a flat
    sum. A few hundred comparisons can exhaust its parser depth even though
    the expression is small. Pairwise grouping retains the same nonzero/zero
    selection semantics while keeping depth proportional to log(frame count).
    """
    if not points:
        raise ValueError("At least one source PTS is required")
    terms = [f"eq(pts,{point})" for point in points]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i + 1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0]


def analyze_video(source: Path, out: Path, *, sample_interval: float = .5, max_pages: int = 100) -> list[dict]:
    """Detect leaf intervals and retain up to three initial seed views.

    ``timestamp_s`` is elapsed video time; ``source_pts`` and ``source_time_base``
    preserve the actual source timestamp. FFmpeg display autorotation is applied;
    paper/text orientation is handled separately by :func:`sever.ocr.read_image`.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sample_interval <= 0 or not np.isfinite(sample_interval) or max_pages < 1:
        raise ValueError("sample_interval must be finite and positive; max_pages must be positive")
    ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
    signature = {"path": str(source), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
                 "sample_interval": sample_interval, "max_pages": max_pages, "algorithm_version": 4}
    manifest = out / "video-analysis.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("signature") == signature and all(Path(f["path"]).is_file() for p in previous["pages"] for f in p["frames"]):
            return previous["pages"]
        raise ValueError("Output contains a different/incomplete analysis; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    frames_out = out / "frames"
    if frames_out.exists() and any(frames_out.iterdir()):
        raise ValueError("Output contains frames without a matching manifest; use a fresh output directory")
    frames_out.mkdir(exist_ok=True)
    metadata = json.loads(_run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)]).stdout)
    stream = next((s for s in metadata["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if stream is None:
        raise ValueError("Source has no usable video stream")
    start = float(stream.get("start_time") or metadata.get("format", {}).get("start_time") or 0)
    duration = float(stream.get("duration") or metadata.get("format", {}).get("duration") or 0)
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), float(stream.get("tags", {}).get("rotate", 0)))
    # Bound only downsampled diagnostics on unusually long videos; disclose coarser sampling.
    effective_interval = max(sample_interval, duration / 12000) if duration else sample_interval
    limits = ["visual_segmentation_is_heuristic", "identical_looking_leaves_may_merge",
              "sub_sampling_short_leaf_glimpses_may_be_missed", "paper_occlusion_and_field_legibility_not_measured"]
    if effective_interval > sample_interval:
        limits.append("sampling_coarsened_to_bound_thumbnail_count")
    pages = []
    with tempfile.TemporaryDirectory(prefix="sever-video-") as temporary:
        temp = Path(temporary)
        select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{effective_interval:.9f})',scale=384:-2,showinfo"
        sampled = _run([ffmpeg, "-hide_banner", "-nostdin", "-copyts", "-i", str(source), "-map", f"0:{stream['index']}",
                        "-an", "-vf", select, "-fps_mode", "vfr", "-q:v", "5", str(temp / "sample-%06d.jpg")])
        samples, filter_time_base = _showinfo(sampled.stderr)
        thumbnails = sorted(temp.glob("sample-*.jpg"))
        if not samples or len(samples) != len(thumbnails):
            raise RuntimeError(f"Frame/PTS mismatch: {len(thumbnails)} images and {len(samples)} timestamps")
        time_base = filter_time_base or stream.get("time_base", "1/1")
        # Integer PTS + time base is authoritative; showinfo's decimal may be rounded.
        for sample in samples:
            sample["source_pts_time_s"] = float(sample["source_pts"] * Fraction(time_base))
            sample["timestamp_s"] = max(0., sample["source_pts_time_s"] - start)
        features = []
        for thumbnail in thumbnails:
            with Image.open(thumbnail) as image:
                features.append(_features(image))
        pair_cache = {}
        groups, discarded = _segment_samples(features, evidence_cache=pair_cache)
        groups, unconfirmed, short_audits = _confirm_short_candidates(
            source, stream, samples, features, groups, ffmpeg, start, effective_interval, temp)
        discarded = sorted([*discarded, *unconfirmed])
        if not groups:
            # Keep a reviewable fallback instead of reporting zero coverage as
            # success when every visible view was too brief or uncertain.
            best = max(range(len(features)), key=lambda i: features[i].quality["sharpness_laplacian_variance"])
            groups = [[best]]
            discarded = [i for i in discarded if i != best]
            limits.append("no_temporally_confirmed_candidate_retained_best_unconfirmed_view")
        if len(groups) > max_pages:
            limits.append(f"page_limit_reached:{len(groups)}_candidates_found:{max_pages}_retained")
        boundaries = [0.] + [(samples[a[-1]]["timestamp_s"] + samples[b[0]]["timestamp_s"]) / 2 for a, b in zip(groups, groups[1:])]
        boundaries.append(max(duration, samples[-1]["timestamp_s"]))
        groups = groups[:max_pages]
        retained: set[int] = set()
        for page_index, group in enumerate(groups):
            # Quality is ranked inside this candidate only; QR density must not rank pages.
            sharpest = max(features[i].quality["sharpness_laplacian_variance"] for i in group) or 1.
            scores = {}
            group_members = set(group)
            for i in group:
                neighbors = [j for j in (i - 1, i + 1) if j in group_members]
                motion = np.mean([_sample_pair(features, i, j, pair_cache)["mean_abs_difference"] for j in neighbors]) if neighbors else 0.
                scores[i] = float(np.sqrt(features[i].quality["sharpness_laplacian_variance"] / sharpest) / (1 + 4 * motion))
                features[i].quality["adjacent_image_difference"] = round(float(motion), 5)
                features[i].quality["relative_selection_score"] = round(scores[i], 5)
            selected = max(group, key=lambda i: scores[i])
            seed_evidence = {}
            eligible = []
            for i in group:
                if i == selected:
                    continue
                proof = _sample_pair(features, selected, i, pair_cache)
                seed_evidence[i] = proof
                if proof["same_page"]:
                    association = proof.get("regional_agreement") or {}
                    strength = float(association.get("minimum_edge_agreement", 1.))
                    eligible.append((scores[i] * strength, i))
            # Time proximity cannot establish document identity: a close view
            # during a turn can contain the next document. Prefer clean,
            # registered views with the strongest within-page image evidence.
            nearby = [i for _, i in sorted(eligible, key=lambda row: (-row[0], abs(samples[row[1]]["timestamp_s"] - samples[selected]["timestamp_s"])))[:2]]
            chosen = sorted([selected, *nearby])
            retained.update(chosen)
            frame_entries = []
            for i in chosen:
                frame_id = f"frame-{i + 1:06d}"
                quality = dict(features[i].quality)
                quality["limitations"] = list(quality["limitations"])
                if i != selected:
                    quality["primary_association"] = seed_evidence[i]
                if len(group) == 1:
                    if (short_audits.get(group[0]) or {}).get("confirmed"):
                        quality["brief_view_confirmed_by_dense_sampling"] = True
                    else:
                        quality["limitations"].append("single_sample_candidate_stability_unconfirmed")
                frame_entries.append({"frame_id": frame_id, "timestamp_s": samples[i]["timestamp_s"],
                                      "source_pts": samples[i]["source_pts"], "source_pts_time_s": samples[i]["source_pts_time_s"],
                                      "source_time_base": time_base, "display_rotation_metadata_degrees": rotation,
                                      "display_rotation_applied": True, "path": str(frames_out / (frame_id + ".png")), "quality": quality})
            pages.append({"page_id": f"page-{page_index + 1:04d}", "start_s": boundaries[page_index], "end_s": boundaries[page_index + 1],
                          "frames": frame_entries, "selected_frame_id": f"frame-{selected + 1:06d}",
                          "segmentation": {"method": "spatially_consistent_ORB_RANSAC_with_preserved_short_candidates", "sample_count": len(group),
                                           "sample_indices": group,
                                           "brief_view_confirmation": short_audits.get(group[0]) if len(group) == 1 else None,
                                           "seed_ranking": "within_page_quality_times_registered_edge_agreement",
                                           "seed_excluded_samples": [{"sample_index": i, "evidence": proof} for i, proof in seed_evidence.items() if not proof["same_page"]],
                                           "boundaries_are_approximate": True, "limitations": list(limits)}})
        selected_indices = sorted(retained)
        expression = _pts_selection_expression([samples[i]["source_pts"] for i in selected_indices])
        extracted = _run([ffmpeg, "-hide_banner", "-nostdin", "-copyts", "-i", str(source), "-map", f"0:{stream['index']}",
                          "-an", "-vf", f"select='{expression}',showinfo", "-fps_mode", "vfr", "-frames:v", str(len(selected_indices)),
                          str(temp / "original-%06d.png")])
        originals = sorted(temp.glob("original-*.png"))
        actual, _ = _showinfo(extracted.stderr)
        expected_pts = [samples[i]["source_pts"] for i in selected_indices]
        if len(originals) != len(selected_indices) or [e["source_pts"] for e in actual] != expected_pts:
            raise RuntimeError("Selected full-resolution frames do not match source PTS; refusing approximate substitution")
        for i, original in zip(selected_indices, originals):
            target = frames_out / f"frame-{i + 1:06d}.png"
            shutil.move(str(original), target)
            with Image.open(target) as image:
                for page in pages:
                    for frame in page["frames"]:
                        if frame["path"] == str(target):
                            frame["pixel_size"] = list(image.size)
    result = {"signature": signature, "source_metadata": metadata, "sampling": {"requested_interval_s": sample_interval,
              "effective_interval_s": effective_interval, "sample_count": len(samples), "discarded_transition_samples": discarded,
              "discarded_samples_are_heuristic_and_require_review": bool(discarded),
              "discarded_transition_evidence": [{"sample_index": i, "source_pts": samples[i]["source_pts"],
                   "source_time_base": time_base, "timestamp_s": samples[i]["timestamp_s"],
                   "reason": "no_stable_brief_view_confirmed" if i in unconfirmed else "partial_page_overlap_or_insufficient_visible_printed_detail",
                   "dense_confirmation": short_audits.get(i),
                   "quality": features[i].quality,
                   "comparisons": [{"other_sample_index": b if a == i else a, "evidence": proof}
                                   for (a, b), proof in pair_cache.items() if i in (a, b)]} for i in discarded],
              "stored_original_count": sum(len(p["frames"]) for p in pages), "full_resolution_frames_per_page_limit": 3}, "pages": pages}
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return pages
