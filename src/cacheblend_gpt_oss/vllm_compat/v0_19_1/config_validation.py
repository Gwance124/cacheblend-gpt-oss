"""Fail-closed validation of the pinned vLLM configuration objects.

This is deliberately separate from CUDA/device validation.  The scheduler
constructs its connector without owning model tensors, so this module checks
only facts available in ``VllmConfig`` and the finalized ``KVCacheConfig``.

The field names and KV-cache spec types are pinned to vLLM 0.19.1 commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* ``VllmConfig`` and its component configs:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L251-L326
* GPT-OSS attention construction and alternating window:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L153
* ``KVCacheGroupSpec``, ``FullAttentionSpec``, and ``SlidingWindowSpec``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/kv_cache_interface.py#L21-L193
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cacheblend_gpt_oss.targets import PINNED_TARGET

_ARCHITECTURE = "GptOssForCausalLM"
_NUM_LAYERS = 24
_NUM_QUERY_HEADS = 64
_NUM_KV_HEADS = 8
_HEAD_DIMENSION = 64
_SLIDING_WINDOW = 128
_MAX_POSITION = 131_072
_NUM_EXPERTS = 32
_ACTIVE_EXPERTS = 4
_VOCAB_SIZE = 201_088
_DEFAULT_BLOCK_SIZE = 16
_LAYER_TYPES = ("sliding_attention", "full_attention") * 12


@dataclass(frozen=True, slots=True, order=True)
class PinnedConfigIssue:
    """One bounded configuration incompatibility."""

    field: str
    expected: str
    observed: str


class UnsupportedPinnedConfigError(RuntimeError):
    """Raised before reuse when the runtime leaves the audited envelope."""

    def __init__(self, issues: tuple[PinnedConfigIssue, ...]) -> None:
        self.issues = issues
        details = "; ".join(
            f"{issue.field}: expected {issue.expected}, observed {issue.observed}"
            for issue in issues
        )
        super().__init__(f"unsupported GPT-OSS CacheBlend configuration: {details}")


def _display(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "<none>"
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, list | tuple | set | frozenset):
        return ",".join(sorted(_display(item) for item in value))
    return str(value)


def _get(value: object, name: str, default: Any = None) -> Any:
    return getattr(value, name, default)


def _mapping_get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return _get(value, name, default)


def _strict_equal(expected: object, observed: object) -> bool:
    """Compare finalized config values without Python's bool/int coercion.

    vLLM configuration fields are populated from several parsers and may be
    supplied by test doubles or launch wrappers.  Plain ``==`` would accept
    values such as ``True`` for an expected integer ``1`` (and ``0`` for an
    expected ``False``), weakening the fail-closed startup boundary.
    Numeric RoPE values intentionally allow an integral value for a float
    expectation because Hugging Face config normalization can preserve either
    representation; booleans are never numeric here.
    """

    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    if isinstance(expected, float):
        return (
            isinstance(observed, int | float)
            and not isinstance(observed, bool)
            and float(observed) == expected
        )
    if expected is None:
        return observed is None
    if type(observed) is not type(expected):
        return False
    return observed == expected


def _served_model_names(model_config: object) -> tuple[str, ...]:
    names = _get(model_config, "served_model_name")
    if isinstance(names, str):
        return (names,)
    if isinstance(names, list | tuple):
        if not all(isinstance(name, str) for name in names):
            return ()
        return tuple(names)
    model = _get(model_config, "model")
    return (model,) if isinstance(model, str) else ()


def _rope_parameters(hf_config: object) -> object:
    # vLLM's pinned ``patch_rope_parameters`` runs before GPT-OSS is
    # constructed and the model reads this finalized field directly.  Never
    # treat the legacy raw ``rope_scaling`` mapping as an equivalent input:
    # accepting it could validate values the model will not consume.
    return _get(hf_config, "rope_parameters", {})


def _layer_index(layer_name: object) -> int | None:
    if not isinstance(layer_name, str):
        return None
    prefix = "model.layers."
    suffix = ".attn.attn"
    if not layer_name.startswith(prefix) or not layer_name.endswith(suffix):
        return None
    value = layer_name[len(prefix) : -len(suffix)]
    if not value.isdigit() or len(value) > 2 or (
        len(value) > 1 and value.startswith("0")
    ):
        return None
    return int(value)


def collect_pinned_config_issues(
    vllm_config: object,
    kv_cache_config: object,
    *,
    v2_model_runner_enabled: bool,
    allow_custom_attention_backend: bool = False,
    allow_unified_kv_mode: bool = False,
) -> tuple[PinnedConfigIssue, ...]:
    """Return every static incompatibility in deterministic field order.

    ``CUSTOM`` is an explicit, repository-local vLLM plugin boundary.  It is
    permitted only for the experimental selective transfer path, whose
    connector validates the parsed transfer configuration before calling this
    function.  All ordinary serving and full-prefill transfer paths retain the
    strict pinned ``TRITON_ATTN`` requirement.
    """

    issues: list[PinnedConfigIssue] = []

    def expect(field: str, expected: object, observed: object) -> None:
        if not _strict_equal(expected, observed):
            issues.append(
                PinnedConfigIssue(
                    field=field,
                    expected=_display(expected),
                    observed=_display(observed),
                )
            )

    model = _get(vllm_config, "model_config")
    hf = _get(model, "hf_config")
    parallel = _get(vllm_config, "parallel_config")
    scheduler = _get(vllm_config, "scheduler_config")
    cache = _get(vllm_config, "cache_config")
    attention = _get(vllm_config, "attention_config")

    expect(
        "model.served_name",
        True,
        PINNED_TARGET.model_id in _served_model_names(model),
    )
    expect(
        "model.architectures",
        (_ARCHITECTURE,),
        tuple(_get(hf, "architectures", ())),
    )
    expect("model.model_type", "gpt_oss", _get(hf, "model_type"))
    expect("model.num_hidden_layers", _NUM_LAYERS, _get(hf, "num_hidden_layers"))
    expect("model.layer_types", _LAYER_TYPES, tuple(_get(hf, "layer_types", ())))
    expect(
        "model.num_attention_heads",
        _NUM_QUERY_HEADS,
        _get(hf, "num_attention_heads"),
    )
    expect(
        "model.num_key_value_heads",
        _NUM_KV_HEADS,
        _get(hf, "num_key_value_heads"),
    )
    # The pinned vLLM GPT-OSS kernel fixtures use the released model's exact
    # vocabulary size.  The correctness harness relies on this to request the
    # complete output distribution rather than a top-k proxy:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/kernels/moe/test_gpt_oss_triton_kernels.py#L195-L205
    expect("model.vocab_size", _VOCAB_SIZE, _get(hf, "vocab_size"))
    expect("model.head_dim", _HEAD_DIMENSION, _get(hf, "head_dim"))
    expect("model.sliding_window", _SLIDING_WINDOW, _get(hf, "sliding_window"))
    expect(
        "model.max_position_embeddings",
        _MAX_POSITION,
        _get(hf, "max_position_embeddings"),
    )
    expect("model.num_local_experts", _NUM_EXPERTS, _get(hf, "num_local_experts"))
    expect(
        "model.num_experts_per_tok",
        _ACTIVE_EXPERTS,
        _get(hf, "num_experts_per_tok"),
    )
    quantization = _get(hf, "quantization_config", {})
    expect("model.quantization", "mxfp4", _mapping_get(quantization, "quant_method"))
    expect("model.attention_bias", True, _get(hf, "attention_bias"))
    expect("model.disable_sliding_window", False, _get(model, "disable_sliding_window"))
    expect("model.enable_prompt_embeds", False, _get(model, "enable_prompt_embeds"))

    rope = _rope_parameters(hf)
    expect("rope.type", "yarn", _mapping_get(rope, "rope_type"))
    # Pinned GPT-OSS reads all RoPE values from ``config.rope_parameters``;
    # do not accept a stale top-level ``rope_theta`` field as proof that the
    # model will construct the same YaRN frequencies.
    expect("rope.theta", 150_000, _mapping_get(rope, "rope_theta"))
    expect("rope.factor", 32.0, _mapping_get(rope, "factor"))
    expect(
        "rope.original_max_position_embeddings",
        4096,
        _mapping_get(rope, "original_max_position_embeddings"),
    )
    expect("rope.beta_fast", 32.0, _mapping_get(rope, "beta_fast"))
    expect("rope.beta_slow", 1.0, _mapping_get(rope, "beta_slow"))
    expect("rope.truncate", False, _mapping_get(rope, "truncate"))

    expect("parallel.tensor_parallel_size", 1, _get(parallel, "tensor_parallel_size"))
    expect(
        "parallel.pipeline_parallel_size", 1, _get(parallel, "pipeline_parallel_size")
    )
    expect("parallel.data_parallel_size", 1, _get(parallel, "data_parallel_size"))
    expect(
        "parallel.prefill_context_parallel_size",
        1,
        _get(parallel, "prefill_context_parallel_size"),
    )
    expect(
        "parallel.decode_context_parallel_size",
        1,
        _get(parallel, "decode_context_parallel_size", 1),
    )
    expect("parallel.enable_dbo", False, _get(parallel, "enable_dbo"))
    expect(
        "parallel.enable_expert_parallel",
        False,
        _get(parallel, "enable_expert_parallel"),
    )

    if not allow_unified_kv_mode:
        expect(
            "scheduler.hybrid_kv_cache_manager_enabled",
            False,
            _get(scheduler, "disable_hybrid_kv_cache_manager"),
        )
    expect("runner.v2_enabled", False, v2_model_runner_enabled)
    observed_attention_backend = _display(_get(attention, "backend"))
    allowed_attention_backends = {"TRITON_ATTN"}
    if allow_custom_attention_backend:
        allowed_attention_backends.add("CUSTOM")
    if observed_attention_backend not in allowed_attention_backends:
        expected_attention_backends = "TRITON_ATTN"
        if allow_custom_attention_backend:
            expected_attention_backends = "TRITON_ATTN|CUSTOM"
        issues.append(
            PinnedConfigIssue(
                field="attention.backend",
                expected=expected_attention_backends,
                observed=observed_attention_backend,
            )
        )
    expect(
        "features.speculative_decoding",
        None,
        _get(vllm_config, "speculative_config"),
    )
    expect("features.lora", None, _get(vllm_config, "lora_config"))
    expect("cache.kv_offloading_size", None, _get(cache, "kv_offloading_size"))
    expect(
        "cache.kv_sharing_fast_prefill",
        False,
        _get(cache, "kv_sharing_fast_prefill"),
    )
    expect("cache.block_size", _DEFAULT_BLOCK_SIZE, _get(cache, "block_size"))

    groups = tuple(_get(kv_cache_config, "kv_cache_groups", ()))
    expected_group_count = 1 if allow_unified_kv_mode and len(groups) == 1 else 2
    expect("kv.groups.count", expected_group_count, len(groups))
    seen_layers: set[int] = set()
    seen_kinds: set[str] = set()
    for group_index, group in enumerate(groups):
        spec = _get(group, "kv_cache_spec")
        kind = type(spec).__name__
        seen_kinds.add(kind)
        expect(
            f"kv.groups.{group_index}.spec_type",
            True,
            kind in {"FullAttentionSpec", "SlidingWindowSpec"},
        )
        expect(
            f"kv.groups.{group_index}.block_size",
            _DEFAULT_BLOCK_SIZE,
            _get(spec, "block_size"),
        )
        expect(
            f"kv.groups.{group_index}.num_kv_heads",
            _NUM_KV_HEADS,
            _get(spec, "num_kv_heads"),
        )
        expect(
            f"kv.groups.{group_index}.head_size",
            _HEAD_DIMENSION,
            _get(spec, "head_size"),
        )
        if kind == "SlidingWindowSpec":
            expect(
                f"kv.groups.{group_index}.sliding_window",
                _SLIDING_WINDOW,
                _get(spec, "sliding_window"),
            )
        elif kind == "FullAttentionSpec":
            observed_sw = _get(spec, "sliding_window")
            if allow_unified_kv_mode and len(groups) == 1:
                if observed_sw is not None:
                    expect(
                        f"kv.groups.{group_index}.sliding_window",
                        _SLIDING_WINDOW,
                        observed_sw,
                    )
            else:
                expect(
                    f"kv.groups.{group_index}.sliding_window",
                    None,
                    observed_sw,
                )

        for layer_name in tuple(_get(group, "layer_names", ())):
            index = _layer_index(layer_name)
            if index is None:
                expect(
                    f"kv.groups.{group_index}.layer_name",
                    "model.layers.<0-23>.attn.attn",
                    layer_name,
                )
                continue
            if allow_unified_kv_mode and len(groups) == 1:
                expected_kind = "FullAttentionSpec"
            else:
                expected_kind = (
                    "SlidingWindowSpec" if index % 2 == 0 else "FullAttentionSpec"
                )
            expect(f"kv.layer.{index}.spec_type", expected_kind, kind)
            if index in seen_layers:
                expect(f"kv.layer.{index}.unique", True, False)
            seen_layers.add(index)

    if allow_unified_kv_mode and len(groups) == 1:
        expected_spec_types: set[str] = {"FullAttentionSpec"}
    else:
        expected_spec_types = {"FullAttentionSpec", "SlidingWindowSpec"}
    expect("kv.groups.spec_types", expected_spec_types, seen_kinds)
    expect("kv.layers", set(range(_NUM_LAYERS)), seen_layers)
    return tuple(issues)


def require_pinned_config(
    vllm_config: object,
    kv_cache_config: object,
    *,
    v2_model_runner_enabled: bool,
    allow_custom_attention_backend: bool = False,
    allow_unified_kv_mode: bool = False,
) -> None:
    """Raise a structured error unless all static target facts match.

    The custom backend exception is intentionally opt-in and is only used by
    the connector after it has parsed ``transfer_selective``.
    """

    issues = collect_pinned_config_issues(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=v2_model_runner_enabled,
        allow_custom_attention_backend=allow_custom_attention_backend,
        allow_unified_kv_mode=allow_unified_kv_mode,
    )
    if issues:
        raise UnsupportedPinnedConfigError(issues)


def collect_transfer_100pct_config_issues(
    vllm_config: object,
    *,
    staging_token_capacity: int,
    allow_prefix_caching: bool = False,
) -> tuple[PinnedConfigIssue, ...]:
    """Validate the stricter execution envelope for live KV transfer.

    The control-flow connector may coexist with ordinary vLLM features because
    it never reads or writes cache tensors.  The first live-transfer milestone
    is intentionally narrower: exactly one request can execute at a time, a
    prompt no larger than the staging capacity must fit in one scheduler step,
    no local prefix may be skipped, and model execution remains eager.  Longer
    prompts are still valid API requests, but the runtime marks them ineligible
    and executes ordinary (possibly chunked) full prefill without transfer.

    ``max_num_scheduled_tokens`` is the budget actually consumed by the pinned
    V1 scheduler when explicitly set; its public default is ``None``, in which
    case the scheduler uses ``max_num_batched_tokens``.  The effective budget
    and runner buffer must cover the configured staging capacity:
    https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/scheduler.py#L48-L63
    https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L103-L110

    Eager mode disables both compilation and CUDA graphs in the finalized
    ``VllmConfig``.  That keeps the connector's Python load/save hooks visible
    during the initial correctness proof:
    https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L847-L853
    """

    if (
        isinstance(staging_token_capacity, bool)
        or not isinstance(staging_token_capacity, int)
        or staging_token_capacity < 1
    ):
        raise ValueError("staging_token_capacity must be a positive integer")

    issues: list[PinnedConfigIssue] = []

    def reject(field: str, expected: object, observed: object) -> None:
        issues.append(
            PinnedConfigIssue(
                field=field,
                expected=_display(expected),
                observed=_display(observed),
            )
        )

    model = _get(vllm_config, "model_config")
    scheduler = _get(vllm_config, "scheduler_config")
    cache = _get(vllm_config, "cache_config")

    model_dtype = _display(_get(model, "dtype"))
    if model_dtype != "torch.bfloat16":
        reject("transfer.model.dtype", "torch.bfloat16", model_dtype)
    if _get(model, "enforce_eager") is not True:
        reject("transfer.model.enforce_eager", True, _get(model, "enforce_eager"))

    if not allow_prefix_caching and _get(cache, "enable_prefix_caching") is not False:
        reject(
            "transfer.cache.enable_prefix_caching",
            False,
            _get(cache, "enable_prefix_caching"),
        )
    cache_dtype = _get(cache, "cache_dtype")
    if cache_dtype not in {"auto", "bfloat16"}:
        reject("transfer.cache.cache_dtype", "auto|bfloat16", cache_dtype)

    exact_scheduler_values = (
        ("max_num_seqs", 1),
        ("long_prefill_token_threshold", 0),
        ("async_scheduling", False),
        ("scheduler_cls", None),
    )
    for field_name, expected in exact_scheduler_values:
        observed = _get(scheduler, field_name)
        if not _strict_equal(expected, observed):
            reject(f"transfer.scheduler.{field_name}", expected, observed)

    max_num_batched_tokens = _get(scheduler, "max_num_batched_tokens")
    if (
        isinstance(max_num_batched_tokens, bool)
        or not isinstance(max_num_batched_tokens, int)
        or max_num_batched_tokens < staging_token_capacity
    ):
        reject(
            "transfer.scheduler.max_num_batched_tokens",
            f">={staging_token_capacity}",
            max_num_batched_tokens,
        )

    configured_scheduled_tokens = _get(scheduler, "max_num_scheduled_tokens")
    effective_scheduled_tokens = (
        max_num_batched_tokens
        if configured_scheduled_tokens is None
        else configured_scheduled_tokens
    )
    if (
        isinstance(effective_scheduled_tokens, bool)
        or not isinstance(effective_scheduled_tokens, int)
        or effective_scheduled_tokens < staging_token_capacity
    ):
        reject(
            "transfer.scheduler.max_num_scheduled_tokens",
            f"None or >={staging_token_capacity}",
            configured_scheduled_tokens,
        )

    return tuple(issues)


def require_transfer_100pct_config(
    vllm_config: object,
    *,
    staging_token_capacity: int,
    allow_prefix_caching: bool = False,
) -> None:
    """Reject startup outside the audited live-transfer scheduler envelope."""

    issues = collect_transfer_100pct_config_issues(
        vllm_config,
        staging_token_capacity=staging_token_capacity,
        allow_prefix_caching=allow_prefix_caching,
    )
    if issues:
        raise UnsupportedPinnedConfigError(issues)


__all__ = [
    "PinnedConfigIssue",
    "UnsupportedPinnedConfigError",
    "collect_pinned_config_issues",
    "collect_transfer_100pct_config_issues",
    "require_pinned_config",
    "require_transfer_100pct_config",
]
