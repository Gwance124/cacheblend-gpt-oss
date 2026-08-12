# SPDX-License-Identifier: Apache-2.0
"""Dependency-free reference math for GPT-OSS YaRN key correction.

The inverse-frequency construction mirrors vLLM 0.19.1 at commit
b1388b1fbf5aaef47937fabe98931211684666a6. This module is intentionally a
scalar reference, not the eventual GPU implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class YarnRopeConfig:
    """The GPT-OSS YaRN fields that determine rotated key values."""

    head_dim: int
    rope_theta: float
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    truncate: bool = True
    extrapolation_factor: float = 1.0
    attention_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be positive")
        if self.beta_fast <= 0 or self.beta_slow <= 0:
            raise ValueError("YaRN beta values must be positive")
        if self.extrapolation_factor < 0:
            raise ValueError("extrapolation_factor must be non-negative")
        if self.attention_factor <= 0:
            raise ValueError("attention_factor must be positive")


def yarn_magnitude_scale(factor: float, attention_factor: float = 1.0) -> float:
    """Return the magnitude scaling embedded in vLLM's YaRN cos/sin cache."""

    if factor <= 0:
        raise ValueError("factor must be positive")
    if attention_factor <= 0:
        raise ValueError("attention_factor must be positive")
    interpolation_scale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return interpolation_scale * attention_factor


def _correction_dimension(
    rotations: float,
    head_dim: int,
    rope_theta: float,
    original_max_position_embeddings: int,
) -> float:
    return (
        head_dim
        * math.log(
            original_max_position_embeddings / (rotations * 2.0 * math.pi)
        )
        / (2.0 * math.log(rope_theta))
    )


def _correction_range(config: YarnRopeConfig) -> tuple[float, float]:
    low = _correction_dimension(
        config.beta_fast,
        config.head_dim,
        config.rope_theta,
        config.original_max_position_embeddings,
    )
    high = _correction_dimension(
        config.beta_slow,
        config.head_dim,
        config.rope_theta,
        config.original_max_position_embeddings,
    )
    if config.truncate:
        low = float(math.floor(low))
        high = float(math.ceil(high))
    return max(low, 0.0), min(high, float(config.head_dim - 1))


def _linear_ramp(index: int, low: float, high: float) -> float:
    if low == high:
        high += 0.001
    value = (index - low) / (high - low)
    return min(max(value, 0.0), 1.0)


def yarn_inverse_frequencies(config: YarnRopeConfig) -> tuple[float, ...]:
    """Build vLLM-compatible blended YaRN inverse frequencies."""

    low, high = _correction_range(config)
    frequencies: list[float] = []
    for pair_index in range(config.head_dim // 2):
        positional_frequency = config.rope_theta ** (
            (2.0 * pair_index) / config.head_dim
        )
        extrapolated = 1.0 / positional_frequency
        interpolated = 1.0 / (config.factor * positional_frequency)
        ramp = _linear_ramp(pair_index, low, high)
        extrapolation_mask = (1.0 - ramp) * config.extrapolation_factor
        frequencies.append(
            interpolated * (1.0 - extrapolation_mask)
            + extrapolated * extrapolation_mask
        )
    return tuple(frequencies)


def _rotate_neox(
    vector: Sequence[float],
    position: int,
    inverse_frequencies: Sequence[float],
    magnitude_scale: float,
) -> tuple[float, ...]:
    if position < 0:
        raise ValueError("position must be non-negative")
    if magnitude_scale <= 0:
        raise ValueError("magnitude_scale must be positive")
    half_dim = len(inverse_frequencies)
    if len(vector) != 2 * half_dim:
        raise ValueError("vector length must be twice the inverse-frequency count")

    output = [0.0] * len(vector)
    for pair_index, inverse_frequency in enumerate(inverse_frequencies):
        angle = position * inverse_frequency
        cosine = math.cos(angle) * magnitude_scale
        sine = math.sin(angle) * magnitude_scale
        first = float(vector[pair_index])
        second = float(vector[pair_index + half_dim])
        output[pair_index] = first * cosine - second * sine
        output[pair_index + half_dim] = second * cosine + first * sine
    return tuple(output)


def apply_yarn_to_key(
    raw_key: Sequence[float], position: int, config: YarnRopeConfig
) -> tuple[float, ...]:
    """Reference the post-RoPE key produced directly at ``position``."""

    if len(raw_key) != config.head_dim:
        raise ValueError("raw key length does not match head_dim")
    return _rotate_neox(
        raw_key,
        position,
        yarn_inverse_frequencies(config),
        yarn_magnitude_scale(config.factor, config.attention_factor),
    )


def correct_shifted_key(
    post_rope_key: Sequence[float],
    source_position: int,
    target_position: int,
    config: YarnRopeConfig,
) -> tuple[float, ...]:
    """Move a post-YaRN key from source to target absolute position.

    GPT-OSS's cached key already contains YaRN magnitude scaling. The correction
    is therefore the unit delta rotation and must not apply that scale again.
    """

    if len(post_rope_key) != config.head_dim:
        raise ValueError("post-RoPE key length does not match head_dim")
    if source_position < 0 or target_position < 0:
        raise ValueError("source and target positions must be non-negative")
    delta = target_position - source_position
    if delta >= 0:
        return _rotate_neox(
            post_rope_key,
            delta,
            yarn_inverse_frequencies(config),
            magnitude_scale=1.0,
        )

    # _rotate_neox validates absolute positions. A negative delta is equivalent
    # to negating every inverse frequency and rotating by its magnitude.
    inverse_frequencies = tuple(-x for x in yarn_inverse_frequencies(config))
    return _rotate_neox(
        post_rope_key,
        -delta,
        inverse_frequencies,
        magnitude_scale=1.0,
    )

