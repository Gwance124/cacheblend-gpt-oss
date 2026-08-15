"""CPU-only contract tests for the pinned out-of-tree vLLM connector."""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from cacheblend_gpt_oss.connector.control_plane import RequestPlan
from cacheblend_gpt_oss.planner import MatchPlan, TokenRange
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    TokenSegment,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CACHE_KEY_PREFIX,
    LmcacheCandidate,
    VerifiedLmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.storage.lookup import (
    LmcacheLookupCounters,
    LmcacheLookupPlan,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import (
    adapt_kv_cache_config,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.compatibility_digest import (
    derive_runtime_compatibility_digests,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.scheduler_runtime import (
    SchedulerLookupMetadata,
    SchedulerLookupStatus,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    FullPrefillCompletion,
    PostForwardOutcome,
    PreForwardOutcome,
    TransferAttemptState,
)

MODULE_NAME = "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector"
METRICS_MODULE_NAME = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector_metrics"
)


def test_import_without_vllm_fails_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    real_import = builtins.__import__

    def import_without_vllm(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("vllm deliberately hidden by contract test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_vllm)
    with pytest.raises(RuntimeError, match=r"requires the pinned vLLM==0\.19\.1"):
        importlib.import_module(MODULE_NAME)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    class FakeKVConnectorRole(Enum):
        SCHEDULER = 0
        WORKER = 1

    class FakeKVConnectorMetadata:
        pass

    class FakeKVConnectorWorkerMetadata:
        pass

    class FakeSupportsHMA:
        pass

    @dataclass
    class FakeKVConnectorStats:
        data: dict[str, Any] = field(default_factory=dict)

    class FakeMetric:
        def __init__(
            self,
            *,
            name: str,
            documentation: str,
            labelnames: list[str],
            buckets: tuple[float, ...] = (),
        ) -> None:
            self.name = name
            self.documentation = documentation
            self.labelnames = tuple(labelnames)
            self.buckets = buckets
            self.label_values: list[tuple[object, ...]] = []
            self.increments: list[int | float] = []
            self.observations: list[int | float] = []
            self.values: list[int | float] = []

        def labels(self, *values: object) -> FakeMetric:
            self.label_values.append(values)
            return self

        def inc(self, value: int | float) -> None:
            self.increments.append(value)

        def observe(self, value: int | float) -> None:
            self.observations.append(value)

        def set(self, value: int | float) -> None:
            self.values.append(value)

    class FakeGauge(FakeMetric):
        pass

    class FakeCounter(FakeMetric):
        pass

    class FakeHistogram(FakeMetric):
        pass

    class FakeKVConnectorPromMetrics:
        def __init__(
            self,
            vllm_config: Any,
            metric_types: dict[type[Any], type[Any]],
            labelnames: list[str],
            per_engine_labelvalues: dict[int, list[object]],
        ) -> None:
            del vllm_config, labelnames
            self._gauge_cls = metric_types[FakeGauge]
            self._counter_cls = metric_types[FakeCounter]
            self._histogram_cls = metric_types[FakeHistogram]
            self.per_engine_labelvalues = per_engine_labelvalues

    class FakeKVConnectorBase:
        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
            self._vllm_config = vllm_config
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._kv_cache_config = kv_cache_config
            self._role = role
            self._connector_metadata: Any = None

        @property
        def role(self) -> Any:
            return self._role

        def bind_connector_metadata(self, metadata: Any) -> None:
            self._connector_metadata = metadata

        def _get_connector_metadata(self) -> Any:
            assert self._connector_metadata is not None
            return self._connector_metadata

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.distributed": ModuleType("vllm.distributed"),
        "vllm.distributed.kv_transfer": ModuleType("vllm.distributed.kv_transfer"),
        "vllm.distributed.kv_transfer.kv_connector": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.base": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.base"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
        ),
    }
    modules["vllm"].__version__ = "0.19.1"  # type: ignore[attr-defined]
    for name, module in modules.items():
        if name != "vllm.distributed.kv_transfer.kv_connector.v1.base":
            module.__path__ = []  # type: ignore[attr-defined]

    base = modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    base.KVConnectorBase_V1 = FakeKVConnectorBase  # type: ignore[attr-defined]
    base.KVConnectorMetadata = FakeKVConnectorMetadata  # type: ignore[attr-defined]
    base.KVConnectorRole = FakeKVConnectorRole  # type: ignore[attr-defined]
    base.KVConnectorWorkerMetadata = (  # type: ignore[attr-defined]
        FakeKVConnectorWorkerMetadata
    )
    base.SupportsHMA = FakeSupportsHMA  # type: ignore[attr-defined]
    metrics = modules["vllm.distributed.kv_transfer.kv_connector.v1.metrics"]
    metrics.KVConnectorStats = FakeKVConnectorStats  # type: ignore[attr-defined]
    metrics.KVConnectorPromMetrics = (  # type: ignore[attr-defined]
        FakeKVConnectorPromMetrics
    )
    metrics.PromMetric = FakeGauge | FakeCounter | FakeHistogram  # type: ignore[attr-defined]
    metrics.PromMetricT = object  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        base=FakeKVConnectorBase,
        metadata=FakeKVConnectorMetadata,
        role=FakeKVConnectorRole,
        supports_hma=FakeSupportsHMA,
        stats=FakeKVConnectorStats,
        prom=FakeKVConnectorPromMetrics,
        gauge=FakeGauge,
        counter=FakeCounter,
        histogram=FakeHistogram,
    )


@pytest.fixture
def loaded_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, SimpleNamespace]:
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, METRICS_MODULE_NAME, raising=False)
    fake = _install_fake_vllm(monkeypatch)
    module = importlib.import_module(MODULE_NAME)
    try:
        yield module, fake
    finally:
        sys.modules.pop(MODULE_NAME, None)
        sys.modules.pop(METRICS_MODULE_NAME, None)


class FakeHfConfig(SimpleNamespace):
    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in vars(self).items()
            if not callable(value)
        }


def _config() -> SimpleNamespace:
    hf_config = FakeHfConfig(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=24,
        layer_types=["sliding_attention", "full_attention"] * 12,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=201_088,
        head_dim=64,
        sliding_window=128,
        max_position_embeddings=131_072,
        num_local_experts=32,
        num_experts_per_tok=4,
        quantization_config={"quant_method": "mxfp4"},
        attention_bias=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 150_000,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
        },
    )
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_seqs=1,
            long_prefill_token_threshold=0,
            async_scheduling=False,
            scheduler_cls=None,
            max_num_batched_tokens=4096,
            max_num_scheduled_tokens=4096,
        ),
        model_config=SimpleNamespace(
            model="/models/gpt-oss-20b",
            served_model_name=["openai/gpt-oss-20b"],
            hf_config=hf_config,
            disable_sliding_window=False,
            enable_prompt_embeds=False,
            dtype="torch.bfloat16",
            max_model_len=131_072,
            runner_type="generate",
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            kv_offloading_size=None,
            kv_sharing_fast_prefill=False,
            enable_prefix_caching=False,
            cache_dtype="auto",
        ),
        attention_config=SimpleNamespace(
            backend=SimpleNamespace(name="TRITON_ATTN")
        ),
        speculative_config=None,
        lora_config=None,
    )


class SlidingWindowSpec(SimpleNamespace):
    pass


class FullAttentionSpec(SimpleNamespace):
    pass


def _kv_cache_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(0, 24, 2)
                ],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=64,
                    sliding_window=128,
                ),
            ),
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(1, 24, 2)
                ],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=64,
                    head_size_v=64,
                    sliding_window=None,
                    attention_chunk_size=None,
                ),
            ),
        ]
    )


def test_exact_runtime_shape_and_hma_contract(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    connector_type = module.GptOssCacheBlendConnector

    assert issubclass(connector_type, fake.base)
    assert issubclass(connector_type, fake.supports_hma)
    assert list(inspect.signature(connector_type.__init__).parameters) == [
        "self",
        "vllm_config",
        "role",
        "kv_cache_config",
    ]


def test_compatibility_probe_reports_finalized_digests_and_stops_startup(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    config = _config()
    kv_cache_config = _kv_cache_config()
    expected = derive_runtime_compatibility_digests(
        config, adapt_kv_cache_config(kv_cache_config)
    )
    config.kv_transfer_config.kv_connector_extra_config = {
        "mode": "compatibility_probe"
    }

    with pytest.raises(RuntimeError, match="compatibility probe") as caught:
        module.GptOssCacheBlendConnector(
            config, fake.role.SCHEDULER, kv_cache_config
        )

    assert expected.model_config_digest in str(caught.value)
    assert expected.kv_cache_config_digest in str(caught.value)
    assert "transfer remains disabled" in str(caught.value)


def test_scheduler_records_all_groups_while_recomputing_every_token(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3, 4],
        prompt_embeds=None,
        num_prompt_tokens=4,
        num_preemptions=0,
    )
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([3, 4], [11, 12]),
        blocks=(
            [
                SimpleNamespace(block_id=3, is_null=False),
                SimpleNamespace(block_id=4, is_null=False),
            ],
            [
                SimpleNamespace(block_id=11, is_null=False),
                SimpleNamespace(block_id=12, is_null=False),
            ],
        ),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)
    metadata = connector.build_connector_meta(SimpleNamespace())

    assert isinstance(metadata, fake.metadata)
    assert metadata.transfer_enabled is False
    assert len(metadata.group_layer_names) == 2
    assert metadata.group_layer_names[0][0] == "model.layers.0.attn.attn"
    assert metadata.group_layer_names[1][0] == "model.layers.1.attn.attn"
    assert len(metadata.handoffs) == 1
    assert metadata.handoffs[0].allocation.grouped_blocks.block_ids_by_group == (
        (3, 4),
        (11, 12),
    )
    assert metadata.handoffs[0].allocation.external_scheduler_tokens == 0
    assert connector.build_connector_meta(SimpleNamespace()).handoffs == ()

    with pytest.raises(RuntimeError, match="must report zero external tokens"):
        connector.update_state_after_alloc(request, blocks, num_external_tokens=1)
    null_blocks = SimpleNamespace(
        get_block_ids=lambda: ([3], [11]),
        blocks=(
            [SimpleNamespace(block_id=3, is_null=True)],
            [SimpleNamespace(block_id=11, is_null=False)],
        ),
    )
    with pytest.raises(ValueError, match="null_block_unsupported"):
        connector.update_state_after_alloc(
            request,
            null_blocks,
            num_external_tokens=0,
        )
    with pytest.raises(
        RuntimeError, match="not compatible with the request allocation"
    ):
        connector.request_finished_all_groups(request, ([99], [11]))
    # The pinned sliding-window manager replaces released entries with its
    # permanent null block (ID 0) before this hook.  Decode growth is appended
    # after the allocation-time table; full attention retains its allocation
    # prefix while sliding attention may replace a leading prefix.
    assert connector.request_finished_all_groups(
        request, ([0, 4, 5], [11, 12, 13])
    ) == (False, None)
    # vLLM calls this hook on the scheduler connector after every engine step;
    # worker observations are returned by the worker connector/output path.
    assert connector.get_kv_connector_stats() is None


@pytest.mark.parametrize(
    "completion_block_ids",
    [
        ([3, 4, 5], [11, 12, 13]),
        # One decode-grown sliding block was also skipped before completion.
        ([0, 0, 0, 5, 6, 7], [11, 12, 13, 14, 15]),
        # A longer decode can skip several blocks beyond the prompt table.
        ([0, 0, 0, 0, 0, 8, 9], [11, 12, 13, 14, 15, 16, 17]),
    ],
)
def test_completion_accepts_long_decode_after_sliding_blocks_are_skipped(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    completion_block_ids: tuple[list[int], list[int]],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(
        request_id="request-decode-growth",
        prompt_token_ids=list(range(32)),
        prompt_embeds=None,
        num_prompt_tokens=32,
        num_preemptions=0,
    )
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([3, 4], [11, 12]),
        blocks=(
            [
                SimpleNamespace(block_id=3, is_null=False),
                SimpleNamespace(block_id=4, is_null=False),
            ],
            [
                SimpleNamespace(block_id=11, is_null=False),
                SimpleNamespace(block_id=12, is_null=False),
            ],
        ),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    assert connector.request_finished_all_groups(
        request, completion_block_ids
    ) == (False, None)


@pytest.mark.parametrize(
    "completion_block_ids",
    [
        # Null blocks are only valid for the sliding-window group.
        ([0, 0, 0, 5, 6, 7], [0, 11, 12, 13, 14]),
        # Full attention must retain its allocation prefix in order.
        ([0, 0, 0, 5, 6, 7], [12, 11, 13, 14, 15]),
        ([0, 0, 0, 5, 6, 7], [11, 13, 12, 14, 15]),
        # A full-attention completion cannot drop an allocated block.
        ([0, 0, 0, 5, 6, 7], [11]),
        # Booleans are malformed block IDs, even though bool is an int subtype.
        ([0, 0, 0, 5, 6, 7], [11, True, 13, 14]),
    ],
)
def test_completion_rejects_malformed_or_wrong_full_attention_after_long_decode(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    completion_block_ids: tuple[list[int], list[int]],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(
        request_id="request-invalid-full-attention-completion",
        prompt_token_ids=list(range(32)),
        prompt_embeds=None,
        num_prompt_tokens=32,
        num_preemptions=0,
    )
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([3, 4], [11, 12]),
        blocks=(
            [
                SimpleNamespace(block_id=3, is_null=False),
                SimpleNamespace(block_id=4, is_null=False),
            ],
            [
                SimpleNamespace(block_id=11, is_null=False),
                SimpleNamespace(block_id=12, is_null=False),
            ],
        ),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    with pytest.raises(
        RuntimeError, match="not compatible with the request allocation"
    ):
        connector.request_finished_all_groups(request, completion_block_ids)


@pytest.mark.parametrize(
    "completion_block_ids",
    [
        ([4, 3, 5], [11, 12]),  # reordered sliding allocation block
        ([3, 4], [12, 11, 13]),  # reordered full-attention allocation block
        ([3, 3], [11, 12]),  # duplicate real block ID
        ([3, 4], [11, 128]),  # out-of-range block ID
    ],
)
def test_completion_rejects_wrong_or_unsafe_block_tables(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    completion_block_ids: tuple[list[int], list[int]],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(
        request_id="request-invalid-completion",
        prompt_token_ids=[1, 2, 3, 4],
        prompt_embeds=None,
        num_prompt_tokens=4,
        num_preemptions=0,
    )
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([3], [11]),
        blocks=(
            [SimpleNamespace(block_id=3, is_null=False)],
            [SimpleNamespace(block_id=11, is_null=False)],
        ),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    with pytest.raises(
        RuntimeError, match="not compatible with the request allocation"
    ):
        connector.request_finished_all_groups(request, completion_block_ids)


def test_worker_registers_every_layer_and_rejects_transfer_claims(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    scheduler = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3, 4],
        prompt_embeds=None,
        num_prompt_tokens=4,
        num_preemptions=0,
    )
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([3], [11]),
        blocks=(
            [SimpleNamespace(block_id=3, is_null=False)],
            [SimpleNamespace(block_id=11, is_null=False)],
        ),
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, blocks, 0)
    metadata = scheduler.build_connector_meta(SimpleNamespace())

    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.WORKER, _kv_cache_config()
    )
    caches = {
        f"model.layers.{index}.attn.attn": object() for index in range(24)
    }
    connector.register_kv_caches(caches)
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(SimpleNamespace())
    worker_metadata = connector.build_connector_worker_meta()
    assert worker_metadata is not None
    assert len(worker_metadata.receipts) == 1
    assert worker_metadata.receipts[0].loaded_match_indexes == ()
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker_metadata)
    )
    connector.wait_for_layer_load("model.layers.0.attn.attn")
    connector.save_kv_layer(
        "model.layers.1.attn.attn",
        caches["model.layers.1.attn.attn"],
        object(),
    )
    connector.wait_for_save()
    assert connector.get_finished(set()) == (None, None)
    assert connector.get_block_ids_with_load_errors() == set()

    bad_metadata = module.GptOssCacheBlendMetadata(
        schema_version=1,
        group_layer_names=metadata.group_layer_names,
        handoffs=metadata.handoffs,
        transfer_enabled=True,
    )
    connector.bind_connector_metadata(bad_metadata)
    with pytest.raises(RuntimeError, match="transfer modes do not match"):
        connector.start_load_kv(SimpleNamespace())

    bad_schema_metadata = module.GptOssCacheBlendMetadata(
        schema_version=True,
        group_layer_names=metadata.group_layer_names,
        handoffs=metadata.handoffs,
    )
    connector.bind_connector_metadata(bad_schema_metadata)
    with pytest.raises(RuntimeError, match="Unsupported connector metadata schema"):
        connector.start_load_kv(SimpleNamespace())


def _enable_transfer(
    config: SimpleNamespace,
    kv_cache_config: SimpleNamespace,
    *,
    transfer_evidence_path: str | None = None,
) -> None:
    digests = derive_runtime_compatibility_digests(
        config, adapt_kv_cache_config(kv_cache_config)
    )
    config.kv_transfer_config.kv_connector_extra_config = {
        "mode": "transfer_100pct",
        "lmcache_server_url": "tcp://127.0.0.1:5555",
        "sidecar_path": "/var/lib/cacheblend/sidecar.sqlite3",
        "lmcache_server_attestation": {
            "lmcache_version": "0.4.3",
            "source_commit": "7f326118a2f1afc7801988dd02e3055bdf21ef6b",
            "protocol": "multiprocess-blend-v2",
            "hash_algorithm": "blake3",
        },
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "model_config_digest": digests.model_config_digest,
        "kv_cache_config_digest": digests.kv_cache_config_digest,
        "adapter_revision": "adapter-revision",
        "staging_token_capacity": 256,
        "request_timeout_seconds": 10.0,
        "transfer_failure_policy": "full_prefill",
    }
    if transfer_evidence_path is not None:
        config.kv_transfer_config.kv_connector_extra_config[
            "transfer_evidence_path"
        ] = transfer_evidence_path


def _verified_transfer_candidate(
    prompt: tuple[int, ...], namespace: CacheNamespace
) -> VerifiedLmcacheCandidate:
    """Build one exact candidate whose cached document moved from position 1024."""

    target_range = TokenRange(0, len(prompt))
    target_segment = TokenSegment(target_range, prompt)
    fingerprint = SHA256_FINGERPRINTER.fingerprint(namespace, prompt)
    storage_hash = bytes([17]) * 32
    record = CacheRecord(
        namespace=namespace,
        fingerprint=fingerprint,
        token_ids=prompt,
        source_range=TokenRange(1024, 1024 + len(prompt)),
        cache_key=LMCACHE_CACHE_KEY_PREFIX + storage_hash.hex(),
    )
    match = VerifiedMatch(
        CandidateMatch(target_segment, fingerprint, record)
    )
    raw = LmcacheCandidate(
        source_relative_range=TokenRange(0, len(prompt)),
        target_range=target_range,
        storage_hash=storage_hash,
        storage_model_name="fake-storage-namespace",
        query_digest=query_digest(prompt),
    )
    return VerifiedLmcacheCandidate.bind(
        raw, match, expected_namespace=namespace
    )


class FakeSchedulerLookupRuntime:
    def __init__(
        self, verified_candidate: VerifiedLmcacheCandidate | None = None
    ) -> None:
        self.requests: list[object] = []
        self.discards: list[str] = []
        self.verified_candidate = verified_candidate

    def lookup(self, request: Any) -> SchedulerLookupMetadata:
        self.requests.append(request)
        request_id = request.request_id
        prompt = request.prompt_token_ids
        preemptions = request.preemption_count
        candidate = (
            self.verified_candidate
            if self.verified_candidate is not None and len(prompt) == 256
            else None
        )
        if candidate is None:
            matches: tuple[VerifiedMatch, ...] = ()
            lookup_plan = LmcacheLookupPlan(
                (), (), LmcacheLookupCounters(0, 0, 0, 0, 0, 0, 0, 0)
            )
            status = SchedulerLookupStatus.FULL_PREFILL_MISS
        else:
            matches = (candidate.match,)
            lookup_plan = LmcacheLookupPlan(
                (candidate,), (), LmcacheLookupCounters(1, 256, 1, 256, 1, 256, 0, 0)
            )
            status = SchedulerLookupStatus.TRANSFER_READY
        segments = tuple(match.target_segment for match in matches)
        plan = RequestPlan(
            request_id,
            len(prompt),
            segments,
            MatchPlan(matches, (), sum(len(segment) for segment in segments)),
        )
        windows = (TokenRange(0, 256),) if len(prompt) == 256 else ()
        return SchedulerLookupMetadata(
            schema_version=1,
            request_plan=plan,
            prompt_token_ids=prompt,
            query_windows=windows,
            lookup_plan=lookup_plan,
            status=status,
            preemption_count=preemptions,
            allocation_generation=preemptions,
        )

    def discard(self, request_id: str) -> None:
        self.discards.append(request_id)


class FakeSchedulerResources:
    def __init__(
        self,
        verified_candidate: VerifiedLmcacheCandidate | None = None,
        *,
        close_error: bool = False,
    ) -> None:
        self.runtime = FakeSchedulerLookupRuntime(verified_candidate)
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("bounded fake scheduler cleanup failure")


class FakeTransferRuntime:
    def __init__(self) -> None:
        self.before_calls: list[tuple[object, object]] = []
        self.after_calls: list[tuple[object, object]] = []

    def before_forward(
        self, metadata: object, adapted_blocks: object
    ) -> PreForwardOutcome:
        self.before_calls.append((metadata, adapted_blocks))
        transfer_eligible = bool(getattr(metadata, "transfer_eligible", False))
        candidate_count = len(getattr(metadata, "verified_candidates", ()))
        return PreForwardOutcome(
            metadata=metadata,  # type: ignore[arg-type]
            state=(
                TransferAttemptState.SUCCEEDED
                if transfer_eligible
                else TransferAttemptState.NOT_ELIGIBLE
            ),
            failure_code=None,
            loaded_candidate_indexes=tuple(range(candidate_count))
            if transfer_eligible
            else (),
            rejected_candidate_indexes=(),
            loaded_kv_tokens=256 if transfer_eligible else 0,
            tokens_to_recompute=256,
            position_correction_latency_seconds=(0.25 if transfer_eligible else 0.0),
        )

    def mark_full_prefill_complete(
        self, pre_forward: PreForwardOutcome, *, recomputed_token_count: int
    ) -> FullPrefillCompletion:
        return FullPrefillCompletion(pre_forward, recomputed_token_count)

    def after_forward(
        self, completion: FullPrefillCompletion, adapted_blocks: object
    ) -> PostForwardOutcome:
        self.after_calls.append((completion, adapted_blocks))
        return PostForwardOutcome(
            completion=completion,
            state=TransferAttemptState.SUCCEEDED,
            failure_code=None,
            eligible_store_tokens=256,
            stored_tokens=256,
            stored_chunks=1,
            sidecar_records_available=1,
            sidecar_records_inserted=1,
        )


class FakeWorkerResources:
    def __init__(self, *, close_error: bool = False) -> None:
        self.transfer_runtime = FakeTransferRuntime()
        self.bridge = SimpleNamespace(
            capture_calls=[],
            finish_calls=[],
            capture_prefill_layer=lambda layer_name: (
                self.bridge.capture_calls.append(layer_name)
            ),
            finish_transfer_evidence=lambda **kwargs: (
                self.bridge.finish_calls.append(kwargs)
            ),
            abort_transfer_evidence=lambda: None,
        )
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("bounded fake worker cleanup failure")


def test_transfer_mode_wires_full_recompute_scheduler_and_worker_hooks(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake = loaded_connector
    kv_cache_config = _kv_cache_config()
    scheduler_config = _config()
    worker_config = _config()
    _enable_transfer(scheduler_config, kv_cache_config)
    _enable_transfer(worker_config, kv_cache_config)
    scheduler_resources = FakeSchedulerResources()
    worker_resources = FakeWorkerResources()
    worker_factory_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "create_scheduler_runtime_resources",
        lambda _config: scheduler_resources,
    )

    def create_worker_resources(*args: object, **kwargs: object) -> object:
        worker_factory_calls.append({"args": args, **kwargs})
        return worker_resources

    monkeypatch.setattr(
        module, "create_worker_runtime_resources", create_worker_resources
    )

    scheduler = module.GptOssCacheBlendConnector(
        scheduler_config, fake.role.SCHEDULER, kv_cache_config
    )
    prompt = list(range(256))
    request = SimpleNamespace(
        request_id="transfer-request",
        prompt_token_ids=prompt,
        prompt_embeds=None,
        num_prompt_tokens=len(prompt),
        num_preemptions=0,
        num_external_computed_tokens=0,
    )
    group_ids = (list(range(16)), list(range(16, 32)))
    blocks = SimpleNamespace(
        get_block_ids=lambda: group_ids,
        blocks=tuple(
            [SimpleNamespace(block_id=value, is_null=False) for value in group]
            for group in group_ids
        ),
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, blocks, 0)
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={request.request_id: 256})
    )
    assert metadata.transfer_enabled
    assert len(metadata.transfers) == 1
    assert not metadata.transfers[0].transfer_eligible
    assert metadata.transfers[0].store_eligible

    worker = module.GptOssCacheBlendConnector(
        worker_config, fake.role.WORKER, kv_cache_config
    )
    caches = {
        f"model.layers.{index}.attn.attn": SimpleNamespace(device="cuda:0")
        for index in range(24)
    }
    worker.register_kv_caches(caches)
    assert worker_factory_calls[0]["device"] == "cuda:0"
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(SimpleNamespace())
    assert len(worker_resources.transfer_runtime.before_calls) == 1
    for layer_name, cache in caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, cache, object())
    worker.wait_for_save()
    assert len(worker_resources.transfer_runtime.after_calls) == 1

    stats = worker.get_kv_connector_stats()
    assert stats is not None
    assert isinstance(stats, fake.stats)
    reduced = stats.reduce()
    assert reduced == {
        "requests": 1,
        "reusable_segments_requested": 1,
        "reusable_segments_hit": 0,
        "reusable_document_tokens_requested": 256,
        "kv_tokens_found": 0,
        "kv_tokens_verified": 0,
        "kv_tokens_rejected": 0,
        "kv_tokens_loaded": 0,
        "kv_tokens_scatter_suppressed": 0,
        "tokens_recomputed": 256,
        "prefill_tokens_avoided": 0,
        "store_tokens_eligible": 256,
        "store_tokens_completed": 256,
        "load_fallbacks": 0,
        "store_fallbacks": 0,
        "lookup_latency_seconds": pytest.approx(
            reduced["lookup_latency_seconds"]
        ),
        "transfer_latency_seconds": pytest.approx(
            reduced["transfer_latency_seconds"]
        ),
        "position_correction_latency_seconds": pytest.approx(
            reduced["position_correction_latency_seconds"]
        ),
        "selective_recomputation_latency_seconds": pytest.approx(
            reduced["selective_recomputation_latency_seconds"]
        ),
        "store_latency_seconds": pytest.approx(
            reduced["store_latency_seconds"]
        ),
        "document_hit_fraction": 0.0,
        "token_hit_fraction": 0.0,
        "effective_saved_prefill_fraction": 0.0,
    }
    assert reduced["lookup_latency_seconds"] >= 0
    assert reduced["transfer_latency_seconds"] >= 0
    assert reduced["position_correction_latency_seconds"] >= 0
    assert reduced["selective_recomputation_latency_seconds"] >= 0
    assert reduced["store_latency_seconds"] >= 0
    assert worker.get_kv_connector_stats() is None


def test_transfer_mode_rejects_missing_pinned_request_counters(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter API drift must not become an implicit zero-credit transfer."""

    module, fake = loaded_connector
    kv_cache_config = _kv_cache_config()
    scheduler_config = _config()
    _enable_transfer(scheduler_config, kv_cache_config)
    scheduler_resources = FakeSchedulerResources()
    monkeypatch.setattr(
        module,
        "create_scheduler_runtime_resources",
        lambda _config: scheduler_resources,
    )
    connector = module.GptOssCacheBlendConnector(
        scheduler_config, fake.role.SCHEDULER, kv_cache_config
    )
    request = SimpleNamespace(
        request_id="missing-counter-request",
        prompt_token_ids=list(range(256)),
        prompt_embeds=None,
        num_prompt_tokens=256,
        # Both fields intentionally omitted to model an unsupported Request
        # shape.  The connector must fail before performing lookup or claiming
        # that the prompt is eligible for the zero-credit transfer path.
    )

    with pytest.raises(RuntimeError, match="num_preemptions"):
        connector.get_num_new_matched_tokens(request, 0)

    request.num_preemptions = 0
    with pytest.raises(RuntimeError, match="num_external_computed_tokens"):
        connector.get_num_new_matched_tokens(request, 0)
    assert scheduler_resources.runtime.requests == []


def test_transfer_mode_loads_verified_moved_candidate_before_full_recompute(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake = loaded_connector
    kv_cache_config = _kv_cache_config()
    scheduler_config = _config()
    worker_config = _config()
    evidence_path = "/var/lib/cacheblend/transfer-evidence.json"
    _enable_transfer(
        scheduler_config,
        kv_cache_config,
        transfer_evidence_path=evidence_path,
    )
    _enable_transfer(
        worker_config,
        kv_cache_config,
        transfer_evidence_path=evidence_path,
    )
    transfer_config = module.parse_connector_extra_config(
        scheduler_config.kv_transfer_config.kv_connector_extra_config
    )
    candidate = _verified_transfer_candidate(
        tuple(range(256)), transfer_config.namespace
    )
    scheduler_resources = FakeSchedulerResources(candidate)
    worker_resources = FakeWorkerResources()

    monkeypatch.setattr(
        module,
        "create_scheduler_runtime_resources",
        lambda _config: scheduler_resources,
    )
    monkeypatch.setattr(
        module,
        "create_worker_runtime_resources",
        lambda *_args, **_kwargs: worker_resources,
    )

    scheduler = module.GptOssCacheBlendConnector(
        scheduler_config, fake.role.SCHEDULER, kv_cache_config
    )
    request = SimpleNamespace(
        request_id="moved-transfer-request",
        prompt_token_ids=list(range(256)),
        prompt_embeds=None,
        num_prompt_tokens=256,
        num_preemptions=0,
        num_external_computed_tokens=0,
    )
    group_ids = (list(range(16)), list(range(16, 32)))
    blocks = SimpleNamespace(
        get_block_ids=lambda: group_ids,
        blocks=tuple(
            [SimpleNamespace(block_id=value, is_null=False) for value in group]
            for group in group_ids
        ),
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, blocks, 0)
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={request.request_id: 256})
    )
    assert metadata.transfers[0].transfer_eligible
    assert metadata.transfers[0].verified_candidates == (candidate,)

    worker = module.GptOssCacheBlendConnector(
        worker_config, fake.role.WORKER, kv_cache_config
    )
    caches = {
        f"model.layers.{index}.attn.attn": SimpleNamespace(device="cuda:0")
        for index in range(24)
    }
    worker.register_kv_caches(caches)
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(SimpleNamespace())
    for layer_name, cache in caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, cache, object())
    worker.wait_for_save()
    assert worker_resources.bridge.capture_calls == list(caches)
    assert worker_resources.bridge.finish_calls == [
        {"recomputed_tokens": 256, "prefill_tokens_avoided": 0}
    ]

    stats = worker.get_kv_connector_stats()
    assert stats is not None
    reduced = stats.reduce()
    assert reduced["kv_tokens_found"] == 256
    assert reduced["kv_tokens_verified"] == 256
    assert reduced["kv_tokens_loaded"] == 256
    assert reduced["kv_tokens_rejected"] == 0
    assert reduced["tokens_recomputed"] == 256
    assert reduced["document_hit_fraction"] == 1.0
    assert reduced["token_hit_fraction"] == 1.0
    assert reduced["effective_saved_prefill_fraction"] == 0.0
    assert reduced["position_correction_latency_seconds"] == pytest.approx(0.25)

    rebuilt = module.GptOssCacheBlendConnector.build_kv_connector_stats(
        stats.data
    )
    assert rebuilt.reduce() == reduced
    prom = module.GptOssCacheBlendConnector.build_prom_metrics(
        worker_config,
        {
            fake.gauge: fake.gauge,
            fake.counter: fake.counter,
            fake.histogram: fake.histogram,
        },
        ["engine"],
        {0: ["0"]},
    )
    assert isinstance(prom, fake.prom)
    prom.observe(stats.data)
    assert prom._counters["tokens_recomputed"].increments == [256]
    assert prom._gauges["effective_saved_prefill_fraction"].values == [0.0]
    assert prom._histograms["lookup_latency_seconds"].observations
    assert all(
        metric.labelnames == ("engine",)
        for metric in (
            *prom._counters.values(),
            *prom._gauges.values(),
            *prom._histograms.values(),
        )
    )

    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata is not None
    # vLLM may finish/free the scheduler request before applying the worker
    # receipt in the same engine step.  The connector must retain state for
    # that late receipt rather than raising UNKNOWN_REQUEST.
    scheduler.request_finished_all_groups(request, group_ids)
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker_metadata)
    )
    assert scheduler_resources.runtime.discards == [request.request_id]
    scheduler.shutdown()
    worker.shutdown()
    assert scheduler_resources.close_calls == 1
    assert worker_resources.close_calls == 1


def test_shutdown_retains_failed_resources_for_a_later_cleanup_retry(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake = loaded_connector
    kv_cache_config = _kv_cache_config()
    scheduler_config = _config()
    worker_config = _config()
    _enable_transfer(scheduler_config, kv_cache_config)
    _enable_transfer(worker_config, kv_cache_config)

    scheduler_resources = FakeSchedulerResources(close_error=True)
    worker_resources = FakeWorkerResources(close_error=True)
    monkeypatch.setattr(
        module,
        "create_scheduler_runtime_resources",
        lambda _config: scheduler_resources,
    )
    monkeypatch.setattr(
        module,
        "create_worker_runtime_resources",
        lambda *_args, **_kwargs: worker_resources,
    )

    scheduler = module.GptOssCacheBlendConnector(
        scheduler_config, fake.role.SCHEDULER, kv_cache_config
    )
    with pytest.raises(RuntimeError, match="resource shutdown failed"):
        scheduler.shutdown()
    assert scheduler._scheduler_resources is scheduler_resources
    scheduler_resources.close_error = False
    scheduler.shutdown()
    assert scheduler._scheduler_resources is None
    assert scheduler_resources.close_calls == 2

    worker = module.GptOssCacheBlendConnector(
        worker_config, fake.role.WORKER, kv_cache_config
    )
    caches = {
        f"model.layers.{index}.attn.attn": SimpleNamespace(device="cuda:0")
        for index in range(24)
    }
    worker.register_kv_caches(caches)
    with pytest.raises(RuntimeError, match="resource shutdown failed"):
        worker.shutdown()
    assert worker._worker_resources is worker_resources
    worker_resources.close_error = False
    worker.shutdown()
    assert worker._worker_resources is None
    assert worker_resources.close_calls == 2


def test_transfer_mode_partial_scheduler_step_records_full_prefill_fallback(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ineligible one-step transfer is visible as a fallback, not a hit."""

    module, fake = loaded_connector
    kv_cache_config = _kv_cache_config()
    scheduler_config = _config()
    worker_config = _config()
    _enable_transfer(scheduler_config, kv_cache_config)
    _enable_transfer(worker_config, kv_cache_config)
    transfer_config = module.parse_connector_extra_config(
        scheduler_config.kv_transfer_config.kv_connector_extra_config
    )
    candidate = _verified_transfer_candidate(
        tuple(range(256)), transfer_config.namespace
    )
    scheduler_resources = FakeSchedulerResources(candidate)
    worker_resources = FakeWorkerResources()
    monkeypatch.setattr(
        module,
        "create_scheduler_runtime_resources",
        lambda _config: scheduler_resources,
    )
    monkeypatch.setattr(
        module,
        "create_worker_runtime_resources",
        lambda *_args, **_kwargs: worker_resources,
    )

    scheduler = module.GptOssCacheBlendConnector(
        scheduler_config, fake.role.SCHEDULER, kv_cache_config
    )
    request = SimpleNamespace(
        request_id="partial-transfer-request",
        prompt_token_ids=list(range(256)),
        prompt_embeds=None,
        num_prompt_tokens=256,
        num_preemptions=0,
        num_external_computed_tokens=0,
    )
    group_ids = (list(range(16)), list(range(16, 32)))
    blocks = SimpleNamespace(
        get_block_ids=lambda: group_ids,
        blocks=tuple(
            [SimpleNamespace(block_id=value, is_null=False) for value in group]
            for group in group_ids
        ),
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, blocks, 0)
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={request.request_id: 128})
    )
    assert metadata.transfers == ()
    assert metadata.lookup_observations[0].kv_tokens_verified == 256

    worker = module.GptOssCacheBlendConnector(
        worker_config, fake.role.WORKER, kv_cache_config
    )
    caches = {
        f"model.layers.{index}.attn.attn": SimpleNamespace(device="cuda:0")
        for index in range(24)
    }
    worker.register_kv_caches(caches)
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(SimpleNamespace())

    stats = worker.get_kv_connector_stats()
    assert stats is not None
    reduced = stats.reduce()
    assert reduced["kv_tokens_loaded"] == 0
    assert reduced["kv_tokens_rejected"] == 256
    assert reduced["tokens_recomputed"] == 256
    assert reduced["load_fallbacks"] == 1
    assert reduced["effective_saved_prefill_fraction"] == 0.0

    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata is not None
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker_metadata)
    )
    scheduler.request_finished_all_groups(request, group_ids)
    scheduler.shutdown()
    worker.shutdown()
    assert scheduler_resources.close_calls == 1
    assert worker_resources.close_calls == 1


def test_connector_metrics_aggregate_hits_fallbacks_and_reject_bad_data(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, _fake = loaded_connector
    stats = module.GptOssCacheBlendStats()
    stats.record_lookup(
        module.CacheBlendLookupObservation(
            prompt_tokens=768,
            reusable_segments_requested=513,
            reusable_segments_hit=2,
            reusable_document_tokens_requested=768,
            kv_tokens_found=768,
            kv_tokens_verified=512,
            kv_tokens_rejected=256,
            latency_seconds=0.25,
        )
    )
    stats.record_load(
        verified_tokens=512,
        loaded_tokens=512,
        rejected_tokens=0,
        recomputed_tokens=768,
        fallback=False,
        latency_seconds=0.5,
    )
    stats.record_store(
        eligible_tokens=768,
        stored_tokens=0,
        fallback=True,
        latency_seconds=0.75,
    )

    other = module.GptOssCacheBlendStats()
    other.record_load(
        verified_tokens=0,
        loaded_tokens=0,
        rejected_tokens=0,
        recomputed_tokens=256,
        fallback=True,
        latency_seconds=1.5,
    )
    assert stats.aggregate(other) is stats
    reduced = stats.reduce()
    assert reduced["kv_tokens_loaded"] == 512
    assert reduced["tokens_recomputed"] == 1024
    assert reduced["load_fallbacks"] == 1
    assert reduced["store_fallbacks"] == 1
    assert reduced["document_hit_fraction"] == pytest.approx(2 / 513, abs=1e-6)
    assert reduced["token_hit_fraction"] == pytest.approx(2 / 3, abs=1e-6)
    assert reduced["effective_saved_prefill_fraction"] == 0.0
    assert reduced["transfer_latency_seconds"] == 1.0

    with pytest.raises(ValueError, match="lookup observation"):
        module.CacheBlendLookupObservation(
            prompt_tokens=256,
            reusable_segments_requested=1,
            reusable_segments_hit=1,
            reusable_document_tokens_requested=256,
            kv_tokens_found=255,
            kv_tokens_verified=256,
            kv_tokens_rejected=0,
            latency_seconds=0.1,
        )
    with pytest.raises(ValueError, match="lookup observation"):
        module.CacheBlendLookupObservation(
            prompt_tokens=256,
            reusable_segments_requested=2,
            reusable_segments_hit=1,
            reusable_document_tokens_requested=256,
            kv_tokens_found=257,
            kv_tokens_verified=256,
            kv_tokens_rejected=1,
            latency_seconds=0.1,
        )
    with pytest.raises(ValueError, match="loaded or rejected"):
        stats.record_load(
            verified_tokens=256,
            loaded_tokens=128,
            rejected_tokens=127,
            recomputed_tokens=256,
            fallback=True,
            latency_seconds=0.1,
        )
    with pytest.raises(ValueError, match="load fallback"):
        stats.record_load(
            verified_tokens=0,
            loaded_tokens=0,
            rejected_tokens=0,
            recomputed_tokens=0,
            fallback=1,  # type: ignore[arg-type]
            latency_seconds=0.0,
        )
    with pytest.raises(ValueError, match="store fallback"):
        stats.record_store(
            eligible_tokens=0,
            stored_tokens=0,
            fallback=1,  # type: ignore[arg-type]
            latency_seconds=0.0,
        )
    with pytest.raises(ValueError, match="exceed eligible"):
        stats.record_store(
            eligible_tokens=0,
            stored_tokens=1,
            fallback=False,
            latency_seconds=0.0,
        )
    malformed = {key: list(values) for key, values in stats.data.items()}
    malformed["requests"] = [1.5]
    with pytest.raises(ValueError, match="stats value"):
        module.GptOssCacheBlendStats(data=malformed)


def test_source_contains_the_pinned_loader_class_name() -> None:
    source = Path(
        "src/cacheblend_gpt_oss/vllm_compat/v0_19_1/connector.py"
    ).read_text(encoding="utf-8")
    assert "class GptOssCacheBlendConnector" in source
    assert "KVConnectorBase_V1, SupportsHMA" in source
    assert "return 0, False" in source


def test_connector_metrics_expose_future_position_and_selective_timers(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, _fake = loaded_connector
    stats = module.GptOssCacheBlendStats()
    stats.record_load(
        verified_tokens=0,
        loaded_tokens=0,
        rejected_tokens=0,
        recomputed_tokens=16,
        fallback=False,
        latency_seconds=0.5,
        position_correction_latency_seconds=0.2,
        selective_recomputation_latency_seconds=0.3,
    )
    reduced = stats.reduce()
    assert reduced["position_correction_latency_seconds"] == 0.2
    assert reduced["selective_recomputation_latency_seconds"] == 0.3
