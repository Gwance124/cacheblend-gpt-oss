import math

import pytest

from cacheblend_gpt_oss.gpt_oss import (
    YarnRopeConfig,
    apply_yarn_to_key,
    correct_shifted_key,
    yarn_inverse_frequencies,
    yarn_magnitude_scale,
)


@pytest.fixture
def gpt_oss_yarn() -> YarnRopeConfig:
    return YarnRopeConfig(
        head_dim=64,
        rope_theta=150_000.0,
        factor=32.0,
        original_max_position_embeddings=4_096,
        beta_fast=32.0,
        beta_slow=1.0,
        truncate=True,
    )


def _raw_key(head_dim: int) -> tuple[float, ...]:
    return tuple(math.sin(index * 0.37) + index / 100.0 for index in range(head_dim))


@pytest.mark.parametrize(
    ("source_position", "target_position"),
    [
        (0, 0),
        (7, 31),
        (31, 7),
        (4_095, 4_096),
        (1_337, 65_537),
        (65_537, 131_071),
    ],
)
def test_shift_correction_matches_direct_target_rotation(
    gpt_oss_yarn: YarnRopeConfig,
    source_position: int,
    target_position: int,
) -> None:
    raw_key = _raw_key(gpt_oss_yarn.head_dim)
    stored_key = apply_yarn_to_key(raw_key, source_position, gpt_oss_yarn)

    corrected = correct_shifted_key(
        stored_key,
        source_position,
        target_position,
        gpt_oss_yarn,
    )
    direct = apply_yarn_to_key(raw_key, target_position, gpt_oss_yarn)

    assert corrected == pytest.approx(direct, rel=2e-11, abs=2e-11)


def test_correction_does_not_apply_magnitude_scale_twice(
    gpt_oss_yarn: YarnRopeConfig,
) -> None:
    raw_key = _raw_key(gpt_oss_yarn.head_dim)
    stored_key = apply_yarn_to_key(raw_key, 11, gpt_oss_yarn)
    corrected = correct_shifted_key(stored_key, 11, 29, gpt_oss_yarn)
    direct = apply_yarn_to_key(raw_key, 29, gpt_oss_yarn)

    corrected_norm = math.sqrt(sum(value * value for value in corrected))
    direct_norm = math.sqrt(sum(value * value for value in direct))
    assert corrected_norm == pytest.approx(direct_norm, rel=1e-12)
    assert yarn_magnitude_scale(gpt_oss_yarn.factor) > 1.0


def test_inverse_frequency_shape_and_extremes(
    gpt_oss_yarn: YarnRopeConfig,
) -> None:
    inverse_frequencies = yarn_inverse_frequencies(gpt_oss_yarn)

    assert len(inverse_frequencies) == gpt_oss_yarn.head_dim // 2
    assert all(value > 0 for value in inverse_frequencies)
    assert inverse_frequencies[0] >= inverse_frequencies[-1]


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive even"):
        YarnRopeConfig(
            head_dim=63,
            rope_theta=150_000.0,
            factor=32.0,
            original_max_position_embeddings=4_096,
            beta_fast=32.0,
            beta_slow=1.0,
        )


def test_key_shape_and_positions_are_validated(
    gpt_oss_yarn: YarnRopeConfig,
) -> None:
    with pytest.raises(ValueError, match="head_dim"):
        apply_yarn_to_key((1.0, 2.0), 0, gpt_oss_yarn)
    with pytest.raises(ValueError, match="non-negative"):
        correct_shifted_key(_raw_key(gpt_oss_yarn.head_dim), -1, 0, gpt_oss_yarn)
