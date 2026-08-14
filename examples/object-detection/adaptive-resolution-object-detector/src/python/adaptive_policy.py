# Copyright 2026 SiMa Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure resolution-tier policy and shared compute budget.

This module has NO cv2/numpy/pyneat dependencies on purpose: the decision logic
is the heart of the example and must be unit-testable with plain Python. It is a
line-for-line mirror of src/cpp/adaptive_policy.h so both languages behave
identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class FrameStats:
    """Per-frame scene summary the policy reasons over."""

    object_count: int = 0
    # smallest min(w, h) across kept boxes; inf => "no small objects"
    min_object_px: float = math.inf
    # lowest score across kept boxes; 1.0 => "nothing uncertain"
    min_confidence: float = 1.0


@dataclass
class PolicyConfig:
    """Policy knobs. The first four map to the issue's ``adaptive:`` block."""

    resolutions: list[int] = field(default_factory=lambda: [320, 640, 960])
    confidence_low: float = 0.40
    min_object_px: float = 24.0
    hysteresis_frames: int = 15
    density_high: int = 20
    density_low: int = 5
    down_size_factor: float = 2.0
    confidence_margin: float = 0.10


@dataclass
class PolicyState:
    """Per-stream mutable state carried across frames."""

    tier_index: int = 0
    pending_index: int = 0
    pending_count: int = 0


@dataclass
class OutputPolicyConfig:
    """Delivered-video resolution policy driven by a shared bandwidth budget.

    This axis is independent of the model input tier: it controls the resolution
    of the H.264 video (and detection working frame) published to Insight, not
    the compiled MLA input size. ``heights`` are candidate delivered heights and
    ``budget_megapixels_per_s`` is the total output pixel-rate fair-shared across
    all active streams (there is no hardware decoder-bandwidth number to query, so
    the operator sets this to the platform's sustainable encode/deliver capacity).
    """

    heights: list[int] = field(default_factory=lambda: [2160, 1080, 720, 480])
    budget_megapixels_per_s: float = 280.0


def _even(value: float) -> int:
    """Round to the nearest positive even integer (H.264 requires even dims)."""
    rounded = int(round(float(value) / 2.0)) * 2
    return max(2, rounded)


def output_candidates(
    native_w: int, native_h: int, out_fps: int, heights: list[int]
) -> list[tuple[int, int, float]]:
    """Per-stream delivered-resolution candidates as (width, height, pixels_per_s).

    The source's native resolution is always the top candidate (a source is never
    upscaled, and the list is never empty). Each configured height strictly below
    native adds an aspect-preserving, even-width candidate. Duplicates are dropped
    and the result is sorted highest-area first, matching select_output_index.
    """
    fps = float(out_fps) if out_fps and out_fps > 0 else 30.0
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, float]] = []

    def add(width: int, height: int) -> None:
        w = _even(width)
        h = _even(height)
        if (w, h) in seen:
            return
        seen.add((w, h))
        candidates.append((w, h, float(w) * float(h) * fps))

    add(native_w, native_h)  # deliver native when the budget allows; never upscale
    for target_h in sorted({int(h) for h in heights}, reverse=True):
        if target_h <= 0 or target_h >= native_h:
            continue
        add(round(native_w * target_h / native_h), target_h)

    candidates.sort(key=lambda c: c[0] * c[1], reverse=True)
    return candidates


def select_output_index(
    active_count: int, candidates: list[tuple[int, int, float]], budget_megapixels_per_s: float
) -> int:
    """Index of the highest-res candidate whose pixel-rate fits the fair share.

    Fair share = budget / active_streams (in pixels/s). ``candidates`` are
    highest-area first, so the first that fits is the richest we can afford. When
    even the smallest candidate exceeds the share we clamp to it (the lowest tier)
    rather than dropping the stream.
    """
    if not candidates:
        return 0
    n = max(1, int(active_count))
    share_pps = (float(budget_megapixels_per_s) * 1_000_000.0) / float(n)
    for index, (_w, _h, rate) in enumerate(candidates):
        if rate <= share_pps + 1e-6:
            return index
    return len(candidates) - 1


def frame_stats(boxes: list[dict], min_score: float) -> FrameStats:
    """Reduce detections (dicts with x1/y1/x2/y2/score) to a FrameStats."""
    stats = FrameStats()
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        stats.object_count += 1
        dim = min(float(box["x2"]) - float(box["x1"]), float(box["y2"]) - float(box["y1"]))
        if dim < stats.min_object_px:
            stats.min_object_px = dim
        if score < stats.min_confidence:
            stats.min_confidence = score
    return stats


def tier_cost(size: int, base_size: int) -> float:
    """Cost of a tier, proportional to pixel area, normalised so smallest == 1."""
    if base_size <= 0:
        return 1.0
    ratio = float(size) / float(base_size)
    return ratio * ratio


def budget_allowed_index(active_count: int, resolutions: list[int], budget_units: float) -> int:
    """Highest tier index a stream may use given the shared budget split across streams."""
    if not resolutions:
        return 0
    n = max(1, active_count)
    share = budget_units / float(n)
    base = resolutions[0]
    allowed = 0
    for i, size in enumerate(resolutions):
        if tier_cost(size, base) <= share + 1e-9:
            allowed = i
        else:
            break
    return allowed


def desired_tier(current: int, stats: FrameStats, cfg: PolicyConfig) -> int:
    """The tier the scene wants this frame, moving at most one step from ``current``."""
    ntiers = len(cfg.resolutions)
    if ntiers == 0:
        return 0
    want_up = (
        stats.min_object_px < cfg.min_object_px
        or (stats.object_count > 0 and stats.min_confidence < cfg.confidence_low)
        or stats.object_count >= cfg.density_high
    )
    if want_up:
        return min(current + 1, ntiers - 1)

    want_down = (
        stats.object_count <= cfg.density_low
        and stats.min_object_px >= cfg.min_object_px * cfg.down_size_factor
        and (
            stats.object_count == 0
            or stats.min_confidence >= cfg.confidence_low + cfg.confidence_margin
        )
    )
    if want_down:
        return max(current - 1, 0)
    return current


def select_tier(state: PolicyState, stats: FrameStats, cfg: PolicyConfig) -> int:
    """Advance the hysteresis state machine one frame; return the committed tier.

    A change only commits after the desired tier has persisted hysteresis_frames.
    """
    desired = desired_tier(state.tier_index, stats, cfg)
    if desired == state.tier_index:
        state.pending_index = state.tier_index
        state.pending_count = 0
        return state.tier_index
    if desired == state.pending_index:
        state.pending_count += 1
    else:
        state.pending_index = desired
        state.pending_count = 1
    if state.pending_count >= cfg.hysteresis_frames:
        state.tier_index = desired
        state.pending_index = desired
        state.pending_count = 0
    return state.tier_index


def effective_tier(
    state: PolicyState,
    stats: FrameStats,
    cfg: PolicyConfig,
    active_count: int,
    budget_units: float,
) -> int:
    """Content desire (hysteresis-gated) clamped by the shared budget."""
    content = select_tier(state, stats, cfg)
    allowed = budget_allowed_index(active_count, cfg.resolutions, budget_units)
    return min(content, allowed)
