# SPDX-License-Identifier: Apache-2.0
"""Production GPT-OSS YaRN key-position correction boundary.

This module implements the callable consumed by the pinned vLLM data plane
without importing Torch at module import time.  The supported input is exactly
one GPT-OSS-20B K slice shaped ``[tokens, 8, 64]`` in BF16 on one CUDA device.

The implementation is grounded in these exact vLLM 0.19.1 sources at commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* GPT-OSS constructs YaRN with its model-provided theta, factor, original
  context, beta values, and ``truncate`` flag, and explicitly selects NeoX
  layout:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L99
* vLLM's YaRN inverse-frequency blend and magnitude scale are defined here:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py#L10-L83
* NeoX rotation splits the final dimension into equal halves and computes
  ``(x1*cos-x2*sin, x2*cos+x1*sin)``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/rotary_embedding/common.py#L146-L182

The exact target configuration is validated before this callable is wired in:
``theta=150000``, ``factor=32``, original context ``4096``, ``beta_fast=32``,
``beta_slow=1``, and ``truncate=False``.  A cached K already contains YaRN's
magnitude scale, so correction applies only the unit-magnitude delta rotation
``R(target-source)``.  Applying the magnitude scale again would be incorrect.

Production arithmetic promotes cached BF16 K to FP32 for angle and multiply-add
work, then casts once back to BF16.  This follows vLLM's float32 construction of
YaRN frequencies while reducing additional correction-rounding error.  It does
not claim bitwise identity with vLLM's original BF16 RoPE operation; GPU
correctness gates must compare logits/hidden states under documented tolerances.
All operations are out of place, and the result is rejected if it aliases the
input storage.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any, NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GPT_OSS_NUM_LAYERS,
)
from cacheblend_gpt_oss.gpt_oss.yarn import YarnRopeConfig, yarn_inverse_frequencies
from cacheblend_gpt_oss.targets import PINNED_TARGET

GPT_OSS_NUM_KV_HEADS = 8
GPT_OSS_HEAD_DIM = 64
GPT_OSS_KV_DTYPE = "torch.bfloat16"
GPT_OSS_YARN_CONFIG = YarnRopeConfig(
    head_dim=GPT_OSS_HEAD_DIM,
    rope_theta=150_000.0,
    factor=32.0,
    original_max_position_embeddings=4096,
    beta_fast=32.0,
    beta_slow=1.0,
    truncate=False,
)
GPT_OSS_YARN_INVERSE_FREQUENCIES = yarn_inverse_frequencies(GPT_OSS_YARN_CONFIG)


class TorchYarnErrorCode(str, Enum):
    """Bounded reasons why shifted K correction was rejected."""

    INVALID_TENSOR = "invalid_tensor"
    INVALID_SHAPE = "invalid_shape"
    INVALID_DTYPE = "invalid_dtype"
    INVALID_DEVICE = "invalid_device"
    INVALID_POSITIONS = "invalid_positions"
    INVALID_LAYER = "invalid_layer"
    ARITHMETIC_FAILED = "arithmetic_failed"
    INVALID_OUTPUT = "invalid_output"
    OUTPUT_ALIASES_INPUT = "output_aliases_input"
    TORCH_DEPENDENCY_MISSING = "torch_dependency_missing"
    TORCH_VERSION_MISMATCH = "torch_version_mismatch"
    TORCH_CUDA_MISMATCH = "torch_cuda_mismatch"


class TorchYarnError(RuntimeError):
    """Fail-closed YaRN correction error with a stable machine code."""

    def __init__(self, code: TorchYarnErrorCode, message: str = "") -> None:
        self.code = code
        super().__init__(message or code.value)


def _fail(code: TorchYarnErrorCode, message: str = "") -> NoReturn:
    raise TorchYarnError(code, message)


class YarnTensorArithmetic(Protocol):
    """Injected, out-of-place tensor arithmetic used by the corrector.

    ``rotate_neox_delta_fp32`` must not mutate ``key_rows``.  It must compute
    angles and rotation arithmetic in FP32, allocate independent output
    storage, and cast the result back to the input dtype before returning.
    """

    def shape(self, tensor: object) -> tuple[int, ...]:
        """Return the tensor's shape as plain integers."""

    def dtype_name(self, tensor: object) -> str:
        """Return a stable dtype name, for example ``torch.bfloat16``."""

    def device_name(self, tensor: object) -> str:
        """Return a device name including its CUDA index."""

    def rotate_neox_delta_fp32(
        self,
        key_rows: object,
        *,
        deltas: tuple[int, ...],
        inverse_frequencies: tuple[float, ...],
    ) -> object:
        """Return an independently allocated NeoX delta rotation."""

    def shares_storage(self, left: object, right: object) -> bool:
        """Return whether two tensors share any backing storage."""


class GptOssTorchYarnCorrector:
    """Validate and position-correct one GPT-OSS-20B K tensor."""

    def __init__(self, arithmetic: YarnTensorArithmetic) -> None:
        self._arithmetic = arithmetic

    def __call__(
        self,
        key_rows: object,
        *,
        source_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        layer_index: int,
    ) -> object:
        """Apply unit-magnitude ``R(target-source)`` to K without mutation."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < GPT_OSS_NUM_LAYERS
        ):
            _fail(
                TorchYarnErrorCode.INVALID_LAYER,
                "layer_index must be a plain integer in [0, 23]",
            )

        shape, dtype_name, device_name = self._inspect(key_rows)
        if (
            len(shape) != 3
            or shape[0] <= 0
            or shape[1] != GPT_OSS_NUM_KV_HEADS
            or shape[2] != GPT_OSS_HEAD_DIM
        ):
            _fail(
                TorchYarnErrorCode.INVALID_SHAPE,
                "GPT-OSS K must have shape [tokens, 8, 64] with tokens > 0",
            )
        if dtype_name != GPT_OSS_KV_DTYPE:
            _fail(
                TorchYarnErrorCode.INVALID_DTYPE,
                f"GPT-OSS K must use {GPT_OSS_KV_DTYPE}; got {dtype_name!r}",
            )
        if not _is_indexed_cuda_device(device_name):
            _fail(
                TorchYarnErrorCode.INVALID_DEVICE,
                "GPT-OSS K must be on an explicitly indexed CUDA device",
            )

        token_count = shape[0]
        source = _validate_positions("source_positions", source_positions, token_count)
        target = _validate_positions("target_positions", target_positions, token_count)
        deltas = tuple(
            target_position - source_position
            for source_position, target_position in zip(source, target, strict=True)
        )

        try:
            corrected = self._arithmetic.rotate_neox_delta_fp32(
                key_rows,
                deltas=deltas,
                inverse_frequencies=GPT_OSS_YARN_INVERSE_FREQUENCIES,
            )
        except Exception as exc:
            raise TorchYarnError(
                TorchYarnErrorCode.ARITHMETIC_FAILED,
                "out-of-place FP32 NeoX delta rotation failed",
            ) from exc

        output_shape, output_dtype, output_device = self._inspect_output(corrected)
        if (
            output_shape != shape
            or output_dtype != dtype_name
            or output_device != device_name
        ):
            _fail(
                TorchYarnErrorCode.INVALID_OUTPUT,
                "corrected K must preserve input shape, dtype, and device",
            )
        try:
            aliases = corrected is key_rows or self._arithmetic.shares_storage(
                corrected, key_rows
            )
        except Exception as exc:
            raise TorchYarnError(
                TorchYarnErrorCode.INVALID_OUTPUT,
                "could not verify corrected K storage independence",
            ) from exc
        if aliases:
            _fail(
                TorchYarnErrorCode.OUTPUT_ALIASES_INPUT,
                "corrected K must not alias cached K input storage",
            )
        return corrected

    def _inspect(self, tensor: object) -> tuple[tuple[int, ...], str, str]:
        try:
            shape = self._arithmetic.shape(tensor)
            dtype_name = self._arithmetic.dtype_name(tensor)
            device_name = self._arithmetic.device_name(tensor)
        except Exception as exc:
            raise TorchYarnError(
                TorchYarnErrorCode.INVALID_TENSOR,
                "could not inspect GPT-OSS K tensor",
            ) from exc
        if not _is_shape(shape) or not dtype_name or not device_name:
            _fail(TorchYarnErrorCode.INVALID_TENSOR, "invalid K tensor metadata")
        return tuple(shape), dtype_name, device_name

    def _inspect_output(self, tensor: object) -> tuple[tuple[int, ...], str, str]:
        try:
            return self._inspect(tensor)
        except TorchYarnError as exc:
            raise TorchYarnError(
                TorchYarnErrorCode.INVALID_OUTPUT,
                "could not inspect corrected K tensor",
            ) from exc


class TorchYarnTensorArithmetic:
    """Out-of-place FP32 NeoX arithmetic over a lazily supplied Torch module."""

    def __init__(self, torch_module: object) -> None:
        self._torch = torch_module

    def _require_tensor(self, tensor: object) -> Any:
        tensor_type = getattr(self._torch, "Tensor", None)
        if tensor_type is None or not isinstance(tensor, tensor_type):
            raise TypeError("expected a torch.Tensor")
        return tensor

    def shape(self, tensor: object) -> tuple[int, ...]:
        value = self._require_tensor(tensor)
        return tuple(int(dimension) for dimension in value.shape)

    def dtype_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).dtype)

    def device_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).device)

    def rotate_neox_delta_fp32(
        self,
        key_rows: object,
        *,
        deltas: tuple[int, ...],
        inverse_frequencies: tuple[float, ...],
    ) -> object:
        torch: Any = self._torch
        value = self._require_tensor(key_rows)
        float32 = torch.float32
        working = value.to(dtype=float32)
        delta_tensor = torch.tensor(
            deltas,
            dtype=float32,
            device=value.device,
        ).reshape(len(deltas), 1, 1)
        frequency_tensor = torch.tensor(
            inverse_frequencies,
            dtype=float32,
            device=value.device,
        ).reshape(1, 1, GPT_OSS_HEAD_DIM // 2)
        angles = delta_tensor * frequency_tensor
        cosine = angles.cos()
        sine = angles.sin()
        first = working[..., : GPT_OSS_HEAD_DIM // 2]
        second = working[..., GPT_OSS_HEAD_DIM // 2 :]
        first_rotated = first * cosine - second * sine
        second_rotated = second * cosine + first * sine
        rotated = torch.cat((first_rotated, second_rotated), dim=-1)
        return rotated.to(dtype=value.dtype)

    def shares_storage(self, left: object, right: object) -> bool:
        left_tensor = self._require_tensor(left)
        right_tensor = self._require_tensor(right)
        return bool(
            left_tensor.untyped_storage().data_ptr()
            == right_tensor.untyped_storage().data_ptr()
        )


def load_torch_yarn_corrector() -> GptOssTorchYarnCorrector:
    """Lazily construct the corrector for exact Torch 2.10/CUDA 12.8."""

    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise TorchYarnError(
            TorchYarnErrorCode.TORCH_DEPENDENCY_MISSING,
            "Torch is not installed; install the pinned GPU runtime extras",
        ) from exc
    observed_version = str(getattr(torch, "__version__", ""))
    if observed_version != PINNED_TARGET.torch_version:
        _fail(
            TorchYarnErrorCode.TORCH_VERSION_MISMATCH,
            f"expected Torch {PINNED_TARGET.torch_version}; got {observed_version!r}",
        )
    observed_cuda = str(getattr(getattr(torch, "version", None), "cuda", ""))
    if observed_cuda != PINNED_TARGET.cuda_runtime:
        _fail(
            TorchYarnErrorCode.TORCH_CUDA_MISMATCH,
            f"expected CUDA runtime {PINNED_TARGET.cuda_runtime}; "
            f"got {observed_cuda!r}",
        )
    return GptOssTorchYarnCorrector(TorchYarnTensorArithmetic(torch))


def _validate_positions(
    name: str, positions: object, token_count: int
) -> tuple[int, ...]:
    if not isinstance(positions, tuple) or len(positions) != token_count:
        _fail(
            TorchYarnErrorCode.INVALID_POSITIONS,
            f"{name} must be a tuple containing exactly one position per token",
        )
    for position in positions:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < GPT_OSS_MAX_CONTEXT_TOKENS
        ):
            _fail(
                TorchYarnErrorCode.INVALID_POSITIONS,
                f"{name} entries must be plain integers in [0, 131071]",
            )
    return positions


def _is_shape(value: object) -> bool:
    return isinstance(value, tuple) and all(
        not isinstance(dimension, bool)
        and isinstance(dimension, int)
        and dimension >= 0
        for dimension in value
    )


def _is_indexed_cuda_device(value: str) -> bool:
    prefix = "cuda:"
    return value.startswith(prefix) and value[len(prefix) :].isdigit()


__all__ = [
    "GPT_OSS_HEAD_DIM",
    "GPT_OSS_KV_DTYPE",
    "GPT_OSS_NUM_KV_HEADS",
    "GPT_OSS_YARN_CONFIG",
    "GPT_OSS_YARN_INVERSE_FREQUENCIES",
    "GptOssTorchYarnCorrector",
    "TorchYarnError",
    "TorchYarnErrorCode",
    "TorchYarnTensorArithmetic",
    "YarnTensorArithmetic",
    "load_torch_yarn_corrector",
]
