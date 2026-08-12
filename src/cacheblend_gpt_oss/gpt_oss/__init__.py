"""GPT-OSS-20B-specific KV correction and model-adapter boundary."""

from cacheblend_gpt_oss.gpt_oss.yarn import (
    YarnRopeConfig,
    apply_yarn_to_key,
    correct_shifted_key,
    yarn_inverse_frequencies,
    yarn_magnitude_scale,
)

__all__ = [
    "YarnRopeConfig",
    "apply_yarn_to_key",
    "correct_shifted_key",
    "yarn_inverse_frequencies",
    "yarn_magnitude_scale",
]

