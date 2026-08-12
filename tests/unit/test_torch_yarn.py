"""CPU-only tests for GPT-OSS Torch YaRN correction."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn

import pytest

from cacheblend_gpt_oss.gpt_oss import torch_yarn as module
from cacheblend_gpt_oss.gpt_oss.torch_yarn import (
    GPT_OSS_YARN_CONFIG,
    GPT_OSS_YARN_INVERSE_FREQUENCIES,
    GptOssTorchYarnCorrector,
    TorchYarnError,
    TorchYarnErrorCode,
)
from cacheblend_gpt_oss.gpt_oss.yarn import correct_shifted_key

NestedRows = tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(slots=True)
class FakeTensor:
    values: NestedRows
    reported_shape: tuple[int, ...]
    dtype: str = "torch.bfloat16"
    device: str = "cuda:0"
    storage: object = field(default_factory=object)


class FakeArithmetic:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
        self.raise_on_rotate = False
        self.output_shape: tuple[int, ...] | None = None
        self.output_dtype: str | None = None
        self.output_device: str | None = None
        self.output_aliases = False

    def shape(self, tensor: object) -> tuple[int, ...]:
        if not isinstance(tensor, FakeTensor):
            raise TypeError("not a fake tensor")
        return tensor.reported_shape

    def dtype_name(self, tensor: object) -> str:
        if not isinstance(tensor, FakeTensor):
            raise TypeError("not a fake tensor")
        return tensor.dtype

    def device_name(self, tensor: object) -> str:
        if not isinstance(tensor, FakeTensor):
            raise TypeError("not a fake tensor")
        return tensor.device

    def rotate_neox_delta_fp32(
        self,
        key_rows: object,
        *,
        deltas: tuple[int, ...],
        inverse_frequencies: tuple[float, ...],
    ) -> object:
        if self.raise_on_rotate:
            raise ValueError("injected arithmetic failure")
        if not isinstance(key_rows, FakeTensor):
            raise TypeError("not a fake tensor")
        self.calls.append((deltas, inverse_frequencies))
        rotated_tokens: list[tuple[tuple[float, ...], ...]] = []
        for token, delta in zip(key_rows.values, deltas, strict=True):
            rotated_heads: list[tuple[float, ...]] = []
            for head in token:
                first = head[:32]
                second = head[32:]
                first_output: list[float] = []
                second_output: list[float] = []
                for index, inverse_frequency in enumerate(inverse_frequencies):
                    angle = delta * inverse_frequency
                    cosine = math.cos(angle)
                    sine = math.sin(angle)
                    first_output.append(
                        first[index] * cosine - second[index] * sine
                    )
                    second_output.append(
                        second[index] * cosine + first[index] * sine
                    )
                rotated_heads.append(tuple(first_output + second_output))
            rotated_tokens.append(tuple(rotated_heads))
        storage = key_rows.storage if self.output_aliases else object()
        return FakeTensor(
            tuple(rotated_tokens),
            self.output_shape or key_rows.reported_shape,
            self.output_dtype or key_rows.dtype,
            self.output_device or key_rows.device,
            storage,
        )

    def shares_storage(self, left: object, right: object) -> bool:
        if not isinstance(left, FakeTensor) or not isinstance(right, FakeTensor):
            raise TypeError("not a fake tensor")
        return left.storage is right.storage


def _key_rows(token_count: int = 4) -> FakeTensor:
    values = tuple(
        tuple(
            tuple(
                math.sin((token_index + 1) * (head_index + 2) * (dimension + 3))
                for dimension in range(64)
            )
            for head_index in range(8)
        )
        for token_index in range(token_count)
    )
    return FakeTensor(values, (token_count, 8, 64))


def _assert_error(
    code: TorchYarnErrorCode, operation: Callable[[], object]
) -> TorchYarnError:
    with pytest.raises(TorchYarnError) as caught:
        operation()
    assert caught.value.code is code
    return caught.value


def test_exact_gpt_oss_yarn_configuration_is_fixed() -> None:
    assert GPT_OSS_YARN_CONFIG.head_dim == 64
    assert GPT_OSS_YARN_CONFIG.rope_theta == 150_000.0
    assert GPT_OSS_YARN_CONFIG.factor == 32.0
    assert GPT_OSS_YARN_CONFIG.original_max_position_embeddings == 4096
    assert GPT_OSS_YARN_CONFIG.beta_fast == 32.0
    assert GPT_OSS_YARN_CONFIG.beta_slow == 1.0
    assert GPT_OSS_YARN_CONFIG.truncate is False
    assert len(GPT_OSS_YARN_INVERSE_FREQUENCIES) == 32


def test_per_token_positive_negative_and_zero_delta_matches_scalar_reference() -> None:
    arithmetic = FakeArithmetic()
    corrector = GptOssTorchYarnCorrector(arithmetic)
    key_rows = _key_rows()
    original_values = key_rows.values
    source = (10, 20, 30, 40)
    target = (14, 15, 30, 57)

    result = corrector(
        key_rows,
        source_positions=source,
        target_positions=target,
        layer_index=23,
    )

    assert isinstance(result, FakeTensor)
    assert result is not key_rows
    assert result.storage is not key_rows.storage
    assert key_rows.values == original_values
    assert arithmetic.calls == [
        ((4, -5, 0, 17), GPT_OSS_YARN_INVERSE_FREQUENCIES)
    ]
    for token_index, (source_position, target_position) in enumerate(
        zip(source, target, strict=True)
    ):
        for head_index in range(8):
            expected = correct_shifted_key(
                original_values[token_index][head_index],
                source_position,
                target_position,
                GPT_OSS_YARN_CONFIG,
            )
            for observed, reference in zip(
                result.values[token_index][head_index], expected, strict=True
            ):
                assert observed == pytest.approx(reference, rel=1e-12, abs=1e-12)
    # A zero delta is an identity rotation: no YaRN magnitude rescale occurred.
    assert result.values[2] == original_values[2]


def test_maximum_context_position_and_layer_zero_are_accepted() -> None:
    result = GptOssTorchYarnCorrector(FakeArithmetic())(
        _key_rows(1),
        source_positions=(131_071,),
        target_positions=(0,),
        layer_index=0,
    )
    assert isinstance(result, FakeTensor)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ([1], (2,)),
        ((1, 2), (2,)),
        ((-1,), (2,)),
        ((1,), (131_072,)),
        ((True,), (2,)),
        ((1.0,), (2,)),
    ],
)
def test_invalid_position_sequences_fail_before_arithmetic(
    source: object, target: object
) -> None:
    arithmetic = FakeArithmetic()
    _assert_error(
        TorchYarnErrorCode.INVALID_POSITIONS,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            _key_rows(1),
            source_positions=source,  # type: ignore[arg-type]
            target_positions=target,  # type: ignore[arg-type]
            layer_index=0,
        ),
    )
    assert arithmetic.calls == []


@pytest.mark.parametrize("layer_index", [-1, 24, True, 1.0])
def test_invalid_layer_fails_before_arithmetic(layer_index: object) -> None:
    arithmetic = FakeArithmetic()
    _assert_error(
        TorchYarnErrorCode.INVALID_LAYER,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            _key_rows(1),
            source_positions=(1,),
            target_positions=(2,),
            layer_index=layer_index,  # type: ignore[arg-type]
        ),
    )
    assert arithmetic.calls == []


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda tensor: setattr(tensor, "reported_shape", (4, 8, 63)),
            TorchYarnErrorCode.INVALID_SHAPE,
        ),
        (
            lambda tensor: setattr(tensor, "reported_shape", (0, 8, 64)),
            TorchYarnErrorCode.INVALID_SHAPE,
        ),
        (
            lambda tensor: setattr(tensor, "dtype", "torch.float16"),
            TorchYarnErrorCode.INVALID_DTYPE,
        ),
        (
            lambda tensor: setattr(tensor, "device", "cpu"),
            TorchYarnErrorCode.INVALID_DEVICE,
        ),
        (
            lambda tensor: setattr(tensor, "device", "cuda"),
            TorchYarnErrorCode.INVALID_DEVICE,
        ),
    ],
)
def test_invalid_tensor_contract_fails_before_arithmetic(
    mutation: Callable[[FakeTensor], None], code: TorchYarnErrorCode
) -> None:
    arithmetic = FakeArithmetic()
    key_rows = _key_rows()
    mutation(key_rows)
    _assert_error(
        code,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            key_rows,
            source_positions=(1, 2, 3, 4),
            target_positions=(2, 3, 4, 5),
            layer_index=0,
        ),
    )
    assert arithmetic.calls == []


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("output_shape", (4, 8, 63)),
        ("output_dtype", "torch.float32"),
        ("output_device", "cuda:1"),
    ],
)
def test_changed_output_contract_is_rejected(attribute: str, value: object) -> None:
    arithmetic = FakeArithmetic()
    setattr(arithmetic, attribute, value)
    _assert_error(
        TorchYarnErrorCode.INVALID_OUTPUT,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            _key_rows(),
            source_positions=(1, 2, 3, 4),
            target_positions=(2, 3, 4, 5),
            layer_index=0,
        ),
    )


def test_output_sharing_input_storage_is_rejected() -> None:
    arithmetic = FakeArithmetic()
    arithmetic.output_aliases = True
    _assert_error(
        TorchYarnErrorCode.OUTPUT_ALIASES_INPUT,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            _key_rows(),
            source_positions=(1, 2, 3, 4),
            target_positions=(2, 3, 4, 5),
            layer_index=0,
        ),
    )


def test_arithmetic_failure_is_wrapped_fail_closed() -> None:
    arithmetic = FakeArithmetic()
    arithmetic.raise_on_rotate = True
    _assert_error(
        TorchYarnErrorCode.ARITHMETIC_FAILED,
        lambda: GptOssTorchYarnCorrector(arithmetic)(
            _key_rows(),
            source_positions=(1, 2, 3, 4),
            target_positions=(2, 3, 4, 5),
            layer_index=0,
        ),
    )


def test_lazy_loader_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str) -> NoReturn:
        assert name == "torch"
        raise ImportError(name)

    monkeypatch.setattr(module, "import_module", missing_import)
    error = _assert_error(
        TorchYarnErrorCode.TORCH_DEPENDENCY_MISSING,
        module.load_torch_yarn_corrector,
    )
    assert "pinned GPU runtime extras" in str(error)


@pytest.mark.parametrize(
    ("torch_version", "cuda_version", "code"),
    [
        (
            "2.10.1+cu128",
            "12.8",
            TorchYarnErrorCode.TORCH_VERSION_MISMATCH,
        ),
        (
            "2.10.0+cu128",
            "12.7",
            TorchYarnErrorCode.TORCH_CUDA_MISMATCH,
        ),
    ],
)
def test_lazy_loader_rejects_unpinned_runtime(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
    cuda_version: str,
    code: TorchYarnErrorCode,
) -> None:
    class FakeVersion:
        cuda = cuda_version

    class FakeTorch:
        __version__ = torch_version
        version = FakeVersion()

    monkeypatch.setattr(module, "import_module", lambda name: FakeTorch())
    _assert_error(code, module.load_torch_yarn_corrector)


def test_lazy_loader_accepts_only_the_exact_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVersion:
        cuda = "12.8"

    class FakeTorch:
        __version__ = "2.10.0+cu128"
        version = FakeVersion()

    monkeypatch.setattr(module, "import_module", lambda name: FakeTorch())
    assert isinstance(module.load_torch_yarn_corrector(), GptOssTorchYarnCorrector)
