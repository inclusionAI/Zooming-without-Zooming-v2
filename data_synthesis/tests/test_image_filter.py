"""Tests for mr_vqa.image_filter (stage-01 content-richness filter)."""
import numpy as np
import pytest
from PIL import Image

from mr_vqa.image_filter import analyze, evaluate, filter_image


def _make_image(w, h, fill=None, noise=False):
    if noise:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    color = fill if fill is not None else (0, 0, 0)
    return Image.new("RGB", (w, h), color)


def test_flat_small_image_rejected():
    img = _make_image(64, 64, fill=(10, 10, 10))
    ok, reason, metrics = filter_image(img)
    assert not ok
    assert reason.startswith("low_resolution:"), metrics


def test_flat_large_image_rejected():
    img = _make_image(1600, 900, fill=(5, 5, 5))
    metrics = analyze(img)
    ok, reason, _ = evaluate(metrics)
    assert not ok
    # All-flat image trips at least one of the entropy/std_dev/edge checks.
    assert reason.startswith(("low_entropy:", "low_edge_density:", "low_std_dev:"))


def test_noisy_large_image_passes_with_relaxed_thresholds():
    img = _make_image(1400, 900, noise=True)
    metrics = analyze(img)
    ok, reason, _ = evaluate(
        metrics,
        min_resolution=1280,
        min_entropy=5.0,
        min_edge_density=0.02,
        min_std_dev=40.0,
        min_long_short_ratio=1.0,
    )
    assert ok, reason
    assert metrics["long_side"] == 1400
    assert metrics["short_side"] == 900


def test_too_square_rejected_when_ratio_enabled():
    img = _make_image(1600, 1600, noise=True)
    metrics = analyze(img)
    ok, reason, _ = evaluate(
        metrics,
        min_resolution=1280,
        min_entropy=0.0,
        min_edge_density=0.0,
        min_std_dev=0.0,
        min_long_short_ratio=2.0,
    )
    assert not ok
    assert reason.startswith("too_square:")