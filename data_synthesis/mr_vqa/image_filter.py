"""Image content-richness filter.

Decides whether an image is rich enough to support multi-region VQA
synthesis. Five components, all computed locally on the decoded image:

  - resolution: long side must be >= MIN_RESOLUTION
  - entropy: per-channel Shannon entropy averaged over RGB
    (texture / saturation proxy)
  - edge_density: fraction of Canny edge pixels (visual structure
    richness proxy)
  - std_dev: overall luminance standard deviation (rejects flat images)
  - long/short side ratio guard

The strict default thresholds below are tuned for information-rich
images, so low-content scenes (solid walls, empty sky, single-object
portraits) are filtered out before stage 02 spends region-extraction
budget on them.
"""
import io
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageStat

# Strict defaults — tuned for information-rich images.
MIN_RESOLUTION = 1280
MIN_ENTROPY = 6.5
MIN_EDGE_DENSITY = 0.04
MIN_STD_DEV = 60.0
MIN_LONG_SHORT_RATIO = 1.0  # disabled — near-square images are acceptable


def _channel_entropy(img_array: np.ndarray) -> float:
    """Mean Shannon entropy across RGB channels."""
    if img_array.ndim == 2:
        img_array = img_array[:, :, None]
    entropies = []
    for c in range(img_array.shape[2]):
        ch = img_array[:, :, c]
        hist, _ = np.histogram(ch.flatten(), bins=256, range=[0, 256])
        hist = hist[hist > 0]
        if hist.size == 0:
            continue
        p = hist / hist.sum()
        entropies.append(float(-np.sum(p * np.log2(p))))
    return float(np.mean(entropies)) if entropies else 0.0


def _edge_density(img_array: np.ndarray) -> float:
    """Fraction of pixels flagged as edges by Canny(50, 150)."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size)


def analyze(img_or_bytes) -> Dict[str, float]:
    """Analyze a PIL image (or raw image bytes); returns a metric dict."""
    if isinstance(img_or_bytes, bytes):
        img = Image.open(io.BytesIO(img_or_bytes))
    else:
        img = img_or_bytes
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    arr = np.array(img)
    std = float(ImageStat.Stat(img.convert("L")).stddev[0]) if w and h else 0.0
    return {
        "width": w,
        "height": h,
        "long_side": max(w, h),
        "short_side": min(w, h),
        "entropy": _channel_entropy(arr),
        "edge_density": _edge_density(arr),
        "std_dev": std,
    }


def evaluate(metrics: Dict[str, float],
             min_resolution: int = MIN_RESOLUTION,
             min_entropy: float = MIN_ENTROPY,
             min_edge_density: float = MIN_EDGE_DENSITY,
             min_std_dev: float = MIN_STD_DEV,
             min_long_short_ratio: float = MIN_LONG_SHORT_RATIO) -> Tuple[bool, str, Dict[str, float]]:
    """Return (pass, reason, metrics). reason is '' when pass=True."""
    if metrics["long_side"] < min_resolution:
        return False, f"low_resolution:{metrics['long_side']}", metrics
    if metrics["entropy"] < min_entropy:
        return False, f"low_entropy:{metrics['entropy']:.2f}", metrics
    if metrics["edge_density"] < min_edge_density:
        return False, f"low_edge_density:{metrics['edge_density']:.4f}", metrics
    if metrics["std_dev"] < min_std_dev:
        return False, f"low_std_dev:{metrics['std_dev']:.1f}", metrics
    if metrics["short_side"] <= 0:
        return False, "invalid_size", metrics
    ratio = metrics["long_side"] / metrics["short_side"]
    if ratio < min_long_short_ratio:
        return False, f"too_square:{ratio:.2f}", metrics
    return True, "", metrics


def filter_image(img_or_bytes, **kwargs) -> Tuple[bool, str, Dict[str, float]]:
    """One-shot analyze+evaluate. kwargs forwarded to evaluate()."""
    metrics = analyze(img_or_bytes)
    return evaluate(metrics, **kwargs)