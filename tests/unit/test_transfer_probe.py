# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the worker-side all-layer transfer probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cacheblend_gpt_oss.connector.control_plane import (
    METADATA_SCHEMA_VERSION,
    GroupedBlockAllocation,
    RequestAllocation,
    RequestHandoffMetadata,
    RequestPlan,
)
from cacheblend_gpt_oss.connector.control_plane import (
    CacheGroupLayout as ControlCacheGroupLayout,
)
from cacheblend_gpt_oss.correctness import read_transfer_evidence
from cacheblend_gpt_oss.correctness.fixture import build_moved_document_fixture
from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import MatchPlan, VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    TokenRange,
    TokenSegment,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CACHE_KEY_PREFIX,
    LmcacheCandidate,
    VerifiedLmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import AdaptedKvCacheBlocks
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_probe import (
    GptOssTransferEvidenceProbe,
    TransferProbeError,
    TransferProbeErrorCode,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    CandidateScatterWork,
    SchedulerTransferMetadata,
    WorkerLoadPlan,
)


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


def _namespace() -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="1" * 64,
        kv_cache_config_digest="2" * 64,
        adapter_revision="adapter-revision",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def _plan() -> WorkerLoadPlan:
    fixture = build_moved_document_fixture()
    prompt = fixture.target_prompt_token_ids
    source = fixture.source_prompt_token_ids
    target_range = TokenRange(17, 273)
    namespace = _namespace()
    segment = TokenSegment(target_range, source)
    record = CacheRecord(
        namespace=namespace,
        fingerprint=SHA256_FINGERPRINTER.fingerprint(namespace, source),
        token_ids=source,
        source_range=TokenRange(0, 256),
        cache_key=LMCACHE_CACHE_KEY_PREFIX + (b"\x03" * 32).hex(),
    )
    match = VerifiedMatch(CandidateMatch(segment, record.fingerprint, record))
    raw = LmcacheCandidate(
        source_relative_range=TokenRange(0, 256),
        target_range=target_range,
        storage_hash=b"\x03" * 32,
        storage_model_name="pinned-storage-namespace",
        query_digest=query_digest(prompt),
    )
    candidate = VerifiedLmcacheCandidate.bind(
        raw, match, expected_namespace=namespace
    )
    layout = _layout()
    block_count = (len(prompt) + 15) // 16
    block_ids = (
        tuple(range(block_count)),
        tuple(range(block_count, 2 * block_count)),
    )
    control_layout = ControlCacheGroupLayout(
        tuple(group.layer_names for group in layout.groups)
    )
    grouped = GroupedBlockAllocation.capture(control_layout, block_ids)
    adapted = AdaptedKvCacheBlocks(
        grouped,
        tuple(
            GroupBlockTable(group.group_id, group.block_size, block_ids[group.group_id])
            for group in layout.groups
        ),
    )
    request_plan = RequestPlan(
        "opaque-request",
        len(prompt),
        (segment,),
        MatchPlan((match,), (), 256),
    )
    handoff = RequestHandoffMetadata(
        METADATA_SCHEMA_VERSION,
        request_plan,
        RequestAllocation("opaque-request", 0, grouped),
    )
    metadata = SchedulerTransferMetadata(
        namespace,
        prompt,
        (candidate,),
        handoff,
        0,
        len(prompt),
        True,
        True,
    )
    scatter = plan_token_scatter(
        layout,
        adapted.group_block_tables,
        TokenTransfer(record.source_range, target_range),
    )
    return WorkerLoadPlan(
        metadata,
        adapted,
        (CandidateScatterWork(0, candidate, scatter),),
    )


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    payload: bytes


class _FakeTensorOps:
    def __init__(self) -> None:
        self.stage = "before"

    def shape(self, tensor: object) -> tuple[int, ...]:
        assert isinstance(tensor, _FakeTensor)
        return tensor.shape

    def dtype_name(self, tensor: object) -> str:
        assert isinstance(tensor, _FakeTensor)
        return "torch.bfloat16"

    def device_name(self, tensor: object) -> str:
        assert isinstance(tensor, _FakeTensor)
        return "cuda:0"

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> object:
        assert isinstance(tensor, int)
        del block_id, block_offset
        source_stage = "source" if self.stage == "loaded" else self.stage
        return _FakeTensor(
            (token_count, 8, 64),
            f"{source_stage}:{tensor}:{component}:{token_count}".encode(),
        )

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> object:
        del tensor, token_start
        return _FakeTensor(
            (token_count, 512),
            f"source:{layer_index}:{component}:{token_count}".encode(),
        )

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> object:
        assert isinstance(tensor, _FakeTensor)
        return _FakeTensor(shape, tensor.payload)

    def copy(self, destination: object, source: object) -> None:
        raise AssertionError("the evidence probe must not copy tensors")

    def synchronize(self, tensor: object) -> None:
        raise AssertionError("the byte reader owns synchronization")


def _read_bytes(tensor: object) -> bytes:
    assert isinstance(tensor, _FakeTensor)
    return tensor.payload


def _correct_key(
    key_rows: object,
    *,
    source_positions: tuple[int, ...],
    target_positions: tuple[int, ...],
    layer_index: int,
) -> object:
    del source_positions, target_positions, layer_index
    return key_rows


def test_probe_writes_bound_all_layer_evidence(tmp_path: Path) -> None:
    plan = _plan()
    ops = _FakeTensorOps()
    path = tmp_path / "transfer-evidence.json"
    probe = GptOssTransferEvidenceProbe(
        output_path=path,
        tensor_ops=ops,
        tensor_bytes_reader=_read_bytes,
        paged_caches={_layer_name(index): index for index in range(24)},
        correct_key_positions=_correct_key,
    )

    probe.begin_load(plan, staging=object(), retrieval_buffer_offset=0)
    ops.stage = "loaded"
    probe.mark_load_complete()
    ops.stage = "prefill"
    for index in range(24):
        probe.record_prefill_layer(_layer_name(index))
    probe.finish(recomputed_tokens=280, prefill_tokens_avoided=0)

    evidence = read_transfer_evidence(path)
    fixture = build_moved_document_fixture()
    assert evidence.schema_version == 2
    assert evidence.source_prompt_digest == fixture.prompt_identity.source_prompt_digest
    assert evidence.target_prompt_digest == fixture.prompt_identity.target_prompt_digest
    assert evidence.loaded_tokens == 256
    assert evidence.target_prompt_tokens == 280
    assert len(evidence.sliding_layers) == 12
    assert len(evidence.full_layers) == 12
    assert evidence.all_layers_loaded_and_overwritten
    assert all(layer.load_write_observed for layer in evidence.layers)
    assert all(layer.prefill_write_observed for layer in evidence.layers)


def test_probe_rejects_an_existing_create_only_output(tmp_path: Path) -> None:
    path = tmp_path / "transfer-evidence.json"
    path.write_text("preserve", encoding="utf-8")

    with pytest.raises(TransferProbeError) as caught:
        GptOssTransferEvidenceProbe(
            output_path=path,
            tensor_ops=_FakeTensorOps(),
            tensor_bytes_reader=_read_bytes,
            paged_caches={_layer_name(index): index for index in range(24)},
            correct_key_positions=_correct_key,
        )

    assert caught.value.code is TransferProbeErrorCode.INVALID_CONFIG
    assert path.read_text(encoding="utf-8") == "preserve"


def test_probe_refuses_to_finalize_with_a_missing_layer(tmp_path: Path) -> None:
    plan = _plan()
    ops = _FakeTensorOps()
    probe = GptOssTransferEvidenceProbe(
        output_path=tmp_path / "transfer-evidence.json",
        tensor_ops=ops,
        tensor_bytes_reader=_read_bytes,
        paged_caches={_layer_name(index): index for index in range(24)},
        correct_key_positions=_correct_key,
    )
    probe.begin_load(plan, staging=object(), retrieval_buffer_offset=0)
    ops.stage = "loaded"
    probe.mark_load_complete()
    ops.stage = "prefill"
    for index in range(23):
        probe.record_prefill_layer(_layer_name(index))

    with pytest.raises(TransferProbeError) as caught:
        probe.finish(recomputed_tokens=280, prefill_tokens_avoided=0)

    assert caught.value.code is TransferProbeErrorCode.INCOMPLETE_LAYERS
