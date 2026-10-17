"""Conservative pixel-only rejection, separate from document segmentation.

An accepted frame passes this coarse screen; it is not a promise of readable
fields or a complete sheet. Cropping, paper whiteness, skin color and document
identity are deliberately not rejection criteria. No frames/files are changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = 1


@dataclass(frozen=True)
class QualityPolicy:
    """Uncalibrated conservative thresholds, reported with every disposition."""

    thumbnail_long_edge: int = 640
    detail_long_edge: int = 1600
    minimum_long_edge: int = 64
    minimum_pixel_area: int = 2048
    black_maximum_channel: int = 24
    black_percentile_999: float = 8.
    uniform_maximum_channel_range: float = 8.
    uniform_maximum_channel_std: float = 1.25
    suspect_laplacian_variance: float = 2.
    suspect_edge_fraction: float = .0005
    detail_laplacian_variance: float = 1.
    detail_maximum_gradient: float = 12.
    detail_edge_fraction: float = .0001


class FrameDisposition(TypedDict):
    frame_id: str | None
    accepted: bool
    status: Literal["accepted", "rejected"]
    reasons: list[str]
    metrics: dict
    version: int
    limitations: list[str]


class PageFrameDisposition(TypedDict):
    status: Literal["accepted", "rejected"]
    selected_frame_id: str | None
    original_selected_frame_id: str | None
    accepted_frame_ids: list[str]
    rejected_frames: list[FrameDisposition]
    frame_dispositions: list[FrameDisposition]
    version: int


_LIMITATIONS = [
    "pixel_thresholds_are_heuristics_not_calibrated_readability_confidence",
    "acceptance_does_not_confirm_document_identity_required_fields_or_full_sheet",
    "sharp_background_or_page_edges_can_hide_blurred_text",
    "cropping_white_paper_and_skin_color_are_not_rejection_criteria",
]


def _gray_detail(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # First differences retain the scale of 8-bit pixel intensity (no Sobel gain).
    maximum_gradient = max(float(np.abs(np.diff(gray.astype(np.float32), axis=axis)).max())
                           if gray.shape[axis] > 1 else 0. for axis in (0, 1))
    return {
        "laplacian_variance": round(float(lap.var()), 6),
        "maximum_adjacent_gray_gradient": round(maximum_gradient, 6),
        "canny_edge_fraction": round(float((cv2.Canny(gray, 40, 100) > 0).mean()), 8),
        "gray_standard_deviation": round(float(gray.std()), 6),
        "pixel_size": [int(gray.shape[1]), int(gray.shape[0])],
    }


def _thumbnail(image: Image.Image, edge: int) -> np.ndarray:
    small = image.copy()
    small.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return np.asarray(small)


def _assess_image(image: Image.Image, policy: QualityPolicy) -> tuple[list[str], dict]:
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands():
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image.convert("RGBA"))
    image = image.convert("RGB")
    width, height = image.size
    # Check native extrema before subsampling: a sparse readable label must not
    # disappear into a nearly uniform thumbnail and trigger a hard rejection.
    extrema = image.getextrema()
    original_ranges = [high - low for low, high in extrema]
    rgb = _thumbnail(image, policy.thumbnail_long_edge)
    flat = rgb.reshape(-1, 3).astype(np.float32)
    ranges = np.ptp(flat, axis=0)
    channel_std = flat.std(axis=0)
    metrics = {
        "original_pixel_size": [width, height],
        "thumbnail": _gray_detail(rgb),
        "maximum_rgb_channel": max(high for _, high in extrema),
        "original_channel_ranges": original_ranges,
        "rgb_percentile_999": round(float(np.percentile(flat, 99.9)), 6),
        "channel_ranges": [round(float(x), 6) for x in ranges],
        "channel_standard_deviations": [round(float(x), 6) for x in channel_std],
        "detail_confirmation": None,
        "policy": asdict(policy),
    }
    reasons = []
    if max(width, height) < policy.minimum_long_edge and width * height < policy.minimum_pixel_area:
        reasons.append("insufficient_pixel_extent")
    if metrics["maximum_rgb_channel"] <= policy.black_maximum_channel and metrics["rgb_percentile_999"] <= policy.black_percentile_999:
        reasons.append("near_black_without_visible_detail")
    elif (max(original_ranges) <= policy.uniform_maximum_channel_range and
          float(ranges.max()) <= policy.uniform_maximum_channel_range and
          float(channel_std.max()) <= policy.uniform_maximum_channel_std):
        # Uniformity is checked over all RGB channels, not the white-paper ratio.
        reasons.append("near_uniform_without_visible_detail")
    else:
        coarse = metrics["thumbnail"]
        if (coarse["laplacian_variance"] < policy.suspect_laplacian_variance and
                coarse["canny_edge_fraction"] < policy.suspect_edge_fraction):
            detail = _gray_detail(_thumbnail(image, policy.detail_long_edge))
            metrics["detail_confirmation"] = detail
            if (detail["laplacian_variance"] < policy.detail_laplacian_variance and
                    detail["maximum_adjacent_gray_gradient"] <= policy.detail_maximum_gradient and
                    detail["canny_edge_fraction"] < policy.detail_edge_fraction):
                reasons.append("extreme_blur_or_smooth_content_without_fine_detail")
    return reasons, metrics


def assess_frame_quality(frame: dict, *, policy: QualityPolicy | None = None) -> FrameDisposition:
    """Read ``frame['path']`` and return an explicit, JSON-safe disposition.

    Uses a 640px thumbnail, confirming suspected extreme blur at up to 1600px.
    An unavailable/corrupt source gets an explicit rejected disposition. Only
    clear pixel failures are screened; no OCR/model or paper-outline rule runs.
    """
    policy = policy or QualityPolicy()
    frame_id = frame.get("frame_id")
    try:
        with Image.open(Path(frame["path"])) as image:
            reasons, metrics = _assess_image(image, policy)
    except (OSError, KeyError, TypeError, ValueError) as error:
        reasons = ["frame_image_unavailable"]
        metrics = {"error_type": type(error).__name__, "policy": asdict(policy)}
    accepted = not reasons
    return {"frame_id": frame_id, "accepted": accepted,
            "status": "accepted" if accepted else "rejected", "reasons": reasons,
            "metrics": metrics, "version": VERSION, "limitations": list(_LIMITATIONS)}


def _rank(frame: dict) -> tuple[float, float]:
    quality = frame.get("quality") or {}
    def finite(key: str) -> float:
        try:
            value = float(quality.get(key) or 0.)
            return value if isfinite(value) else 0.
        except (ValueError, TypeError):
            return 0.
    return finite("relative_selection_score"), finite("sharpness_laplacian_variance")


def evaluate_page_frames(page: dict, *, policy: QualityPolicy | None = None) -> PageFrameDisposition:
    """Preserve candidate inventory/PTS; choose a passing primary or neighbor.

    Returns frame IDs in their input order. The prior primary is retained if it
    passes. If it fails, the best passing neighbor by existing within-page rank
    is chosen (stable first-in-input tie break). If all fail, selection is null;
    the caller should report rejection while retaining this candidate.
    """
    frames = page.get("frames") or []
    ids = [frame["frame_id"] for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("Frame IDs must be unique inside a page candidate")
    dispositions = [assess_frame_quality(frame, policy=policy) for frame in frames]
    accepted = [frame for frame, disposition in zip(frames, dispositions) if disposition["accepted"]]
    original = page.get("selected_frame_id")
    selected = next((frame for frame in accepted if frame["frame_id"] == original), None)
    if selected is None and accepted:
        selected = max(accepted, key=_rank)
    return {
        "status": "accepted" if selected is not None else "rejected",
        "selected_frame_id": selected["frame_id"] if selected is not None else None,
        "original_selected_frame_id": original,
        "accepted_frame_ids": [frame["frame_id"] for frame in accepted],
        "rejected_frames": [d for d in dispositions if not d["accepted"]],
        "frame_dispositions": dispositions,
        "version": VERSION,
    }
