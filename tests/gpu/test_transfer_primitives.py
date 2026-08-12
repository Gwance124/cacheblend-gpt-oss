# SPDX-License-Identifier: Apache-2.0
"""Manual solab-g3 checks for the real Torch/CUDA transfer primitives."""

from __future__ import annotations

import pytest

from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.gpt_oss.torch_yarn import (
    GPT_OSS_YARN_INVERSE_FREQUENCIES,
    load_torch_yarn_corrector,
)
from cacheblend_gpt_oss.planner import TokenRange
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    GptOssDataPlane,
    load_torch_tensor_ops,
)


def _torch_cuda() -> object:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("manual solab-g3 test: CUDA is not available")
    return torch


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def _layout() -> GptOssHybridCacheLayout:
    return GptOssHybridCacheLayout(
        (
            CacheGroupLayout(
                0,
                AttentionKind.FULL,
                tuple(_layer_name(index) for index in range(1, 24, 2)),
                16,
                None,
            ),
            CacheGroupLayout(
                1,
                AttentionKind.SLIDING,
                tuple(_layer_name(index) for index in range(0, 24, 2)),
                16,
                128,
            ),
        )
    )


def _rotate_neox(torch: object, rows: object, positions: tuple[int, ...]) -> object:
    working = rows.to(dtype=torch.float32)
    position_tensor = torch.tensor(
        positions, dtype=torch.float32, device=rows.device
    ).reshape(len(positions), 1, 1)
    frequency_tensor = torch.tensor(
        GPT_OSS_YARN_INVERSE_FREQUENCIES,
        dtype=torch.float32,
        device=rows.device,
    ).reshape(1, 1, 32)
    angles = position_tensor * frequency_tensor
    cosine = angles.cos()
    sine = angles.sin()
    first = working[..., :32]
    second = working[..., 32:]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


@pytest.mark.gpu
@pytest.mark.integration
def test_cuda_yarn_shift_matches_direct_target_rotation() -> None:
    torch = _torch_cuda()
    torch.manual_seed(1234)
    device = "cuda:0"
    raw = torch.randn((5, 8, 64), dtype=torch.float32, device=device)
    source_positions = (0, 4095, 4096, 50_000, 131_071)
    target_positions = (17, 4000, 8192, 49_000, 100_000)
    source_key = _rotate_neox(torch, raw, source_positions).to(torch.bfloat16)
    direct_target = _rotate_neox(torch, raw, target_positions).to(torch.bfloat16)

    corrected = load_torch_yarn_corrector()(
        source_key,
        source_positions=source_positions,
        target_positions=target_positions,
        layer_index=23,
    )

    error = (corrected.float() - direct_target.float()).abs()
    # Frozen before the first solab-g3 run. This threshold includes the first
    # BF16 source-position quantization followed by FP32 delta rotation.
    assert error.max().item() <= 0.0625
    assert error.mean().item() <= 0.004
    assert (
        corrected.untyped_storage().data_ptr()
        != source_key.untyped_storage().data_ptr()
    )


@pytest.mark.gpu
@pytest.mark.integration
def test_cuda_hybrid_gather_scatter_round_trip_preserves_kv() -> None:
    torch = _torch_cuda()
    torch.manual_seed(5678)
    device = "cuda:0"
    layout = _layout()
    transfer = TokenTransfer(TokenRange(5, 27), TokenRange(5, 27))
    plan = plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, 16, (0, 1)),
            GroupBlockTable(1, 16, (2, 3)),
        ),
        transfer,
    )
    source = {
        _layer_name(index): torch.randn(
            (4, 2, 16, 8, 64), dtype=torch.bfloat16, device=device
        )
        for index in range(24)
    }
    destination = {
        name: torch.zeros_like(tensor) for name, tensor in source.items()
    }
    compact_staging = torch.zeros(
        (2, 24, 64, 512), dtype=torch.bfloat16, device=device
    )
    retrieval_staging = torch.zeros_like(compact_staging)
    data_plane = GptOssDataPlane(load_torch_tensor_ops())

    gathered = data_plane.gather_precomputed_kv(
        paged_caches=source,
        staging=compact_staging,
        layer_spans=plan.layer_spans,
        document_target_range=transfer.target_range,
        store_buffer_offset=0,
    )
    retrieval_staging[:, :, 5:27, :].copy_(compact_staging[:, :, :22, :])

    def identity_corrector(
        key_rows: object,
        *,
        source_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        layer_index: int,
    ) -> object:
        assert source_positions == target_positions
        assert 0 <= layer_index < 24
        return key_rows.clone()

    scattered = data_plane.scatter_retrieved_kv(
        staging=retrieval_staging,
        paged_caches=destination,
        layer_spans=plan.layer_spans,
        retrieval_buffer_offset=0,
        query_token_count=27,
        correct_key_positions=identity_corrector,
    )

    assert {span.attention_kind for span in plan.group_spans} == {
        AttentionKind.FULL,
        AttentionKind.SLIDING,
    }
    assert not gathered.sinks_touched
    assert not scattered.sinks_touched
    assert gathered.layer_token_rows == 24 * 22
    assert scattered.layer_token_rows == 24 * 22
    for span in plan.layer_spans:
        source_rows = source[span.layer_name][
            span.group_span.block_id,
            :,
            span.group_span.block_offset : (
                span.group_span.block_offset + span.token_count
            ),
            :,
            :,
        ]
        destination_rows = destination[span.layer_name][
            span.group_span.block_id,
            :,
            span.group_span.block_offset : (
                span.group_span.block_offset + span.token_count
            ),
            :,
            :,
        ]
        assert torch.equal(destination_rows, source_rows)
