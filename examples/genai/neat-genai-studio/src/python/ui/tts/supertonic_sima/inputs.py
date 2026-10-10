# Vendored from https://github.com/florianvoss-commit/supertonic-sima
# (app/supertonic_sima/inputs.py) at commit 3b837b3e1b6a378ab8c24c3c04b079429b67e237,
# included in Neat GenAI Studio at the upstream author's request. See README.md
# in this directory for how to refresh it and THIRD_PARTY_TTS_MODELS.md for
# attribution. Local changes: none.
"""Host-side inputs for the externalized Supertonic sinusoidal operators."""

from __future__ import annotations

import numpy as np


MAX_SEQUENCE_LENGTH = 192
DEFAULT_STEPS = 8
TIME_INPUT = "time_sinusoidal"
TIME_TABLE_KEY = TIME_INPUT
TIME_INPUT_SHAPE = (1, 64, 1, 1)

ROPE_INPUT = "rope_tables"
ROPE_TABLE_KEY = "rope_sin_cos_by_length"
ROPE_PAIR_SHAPE = (1, 64, 1, MAX_SEQUENCE_LENGTH)
ROPE_INPUT_SHAPE = (1, 128, 1, MAX_SEQUENCE_LENGTH)
ROPE_BANK_SHAPE = (MAX_SEQUENCE_LENGTH, *ROPE_PAIR_SHAPE)

CONDITIONAL_STYLE_KEY = "conditional_style_key"
UNCONDITIONAL_TEXT = "unconditional_text"
UNCONDITIONAL_STYLE_KEY = "unconditional_style_key"
UNCONDITIONAL_STYLE_VALUE = "unconditional_style_value"
RUNTIME_CONSTANT_KEYS = (
    CONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_TEXT,
    UNCONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_STYLE_VALUE,
)

# Exact frequency vector embedded by the pinned upstream Supertonic 3 time
# encoder. Keeping it host-side lets the same compiled MLA graph run a
# different Euler schedule without recompilation.
_TIME_FREQUENCIES = np.asarray(
    [
        1.0,
        0.7429639101028442,
        0.5519954562187195,
        0.4101127088069916,
        0.3046989440917969,
        0.2263803482055664,
        0.16819243133068085,
        0.12496090680360794,
        0.09284145385026932,
        0.06897785514593124,
        0.051248062402009964,
        0.03807545825839043,
        0.028288694098591805,
        0.02101748064160347,
        0.01561522763222456,
        0.011601552367210388,
        0.008619535714387894,
        0.006404004525393248,
        0.004757944494485855,
        0.0035349815152585506,
        0.00262636411935091,
        0.0019512929720804095,
        0.0014497404918074608,
        0.0010771049419417977,
        0.0008002502145245671,
        0.0005945570883341134,
        0.00044173450442031026,
        0.0003281926619820297,
        0.0002438353403704241,
        0.00018116086721420288,
        0.00013459600450005382,
        9.99999901978299e-05,
    ],
    dtype=np.float32,
)


def build_time_table(total_steps: int) -> np.ndarray:
    """Build the pinned model's sinusoidal timestep inputs."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    ratios = np.arange(total_steps, dtype=np.float32) / np.float32(total_steps)
    angles = (
        ratios[:, None] * np.float32(1000.0) * _TIME_FREQUENCIES[None, :]
    )
    table = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
    return np.ascontiguousarray(
        table.reshape(total_steps, *TIME_INPUT_SHAPE), dtype=np.float32
    )


def effective_mask_length(mask: np.ndarray, *, name: str) -> int:
    """Return the number of valid positions represented by a binary mask."""

    value = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or np.any((rounded < 0) | (rounded > 1)):
        raise ValueError(f"{name} must be a binary 0/1 mask")
    length = int(rounded.sum())
    if length < 1 or length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{name} effective length {length} is outside 1..{MAX_SEQUENCE_LENGTH}"
        )
    return length


def validate_rope_bank(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float32)
    if table.shape != ROPE_BANK_SHAPE:
        raise ValueError(f"RoPE bank shape {table.shape} != {ROPE_BANK_SHAPE}")
    if not np.isfinite(table).all():
        raise ValueError("RoPE bank contains a non-finite value")
    return table


def pack_rope_input(
    table: np.ndarray,
    latent_mask: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """Select and pack latent/text sin-cos pairs for the static MLA graph."""

    table = validate_rope_bank(table)
    latent_length = effective_mask_length(latent_mask, name="latent_mask")
    text_length = effective_mask_length(text_mask, name="text_mask")
    packed = np.concatenate(
        [table[latent_length - 1], table[text_length - 1]], axis=1
    )
    if packed.shape != ROPE_INPUT_SHAPE:
        raise RuntimeError(f"packed RoPE shape {packed.shape} != {ROPE_INPUT_SHAPE}")
    return np.ascontiguousarray(packed, dtype=np.float32)
