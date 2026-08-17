# SPDX-License-Identifier: Apache-2.0
"""Strict connector-extra configuration for the pinned transfer modes.

The version connector receives this value from vLLM 0.19.1's
``KVTransferConfig.kv_connector_extra_config``:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/kv_transfer.py#L20-L65

This module intentionally imports neither vLLM, LMCache, Torch, nor CUDA.
Empty configuration keeps the connector in its transfer-disabled control-flow
mode. ``compatibility_probe`` is also transfer-disabled and intentionally stops
startup after the connector derives the finalized model/KV digests. Transfer is
enabled only by the complete exact ``transfer_100pct`` schema; configuration
drift fails before any cache lookup or transport starts.

``disable_kv_scatter`` is an optional, explicit opt-in diagnostic switch on
``transfer_100pct``.  It defaults to ``False`` and leaves the normal path
byte-for-byte unchanged.  When set ``True`` the connector still performs
lookup, retrieval into staging, and YaRN key correction, but the worker never
copies corrected K/V into vLLM's real paged KV cache; see
:mod:`~.data_plane` and :mod:`~.worker_bridge`.  A run with this flag set can
never be mistaken for a real transfer: ``kv_tokens_loaded`` stays zero and
the suppressed run is reported through
``TransferFallbackCode.SCATTER_SUPPRESSED_DIAGNOSTIC``.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn
from urllib.parse import urlsplit

from cacheblend_gpt_oss.planner.models import CacheNamespace
from cacheblend_gpt_oss.targets import PINNED_TARGET

LMCACHE_SOURCE_COMMIT = "7f326118a2f1afc7801988dd02e3055bdf21ef6b"
LMCACHE_BLEND_PROTOCOL = "multiprocess-blend-v2"
LMCACHE_HASH_ALGORITHM = "blake3"
LMCACHE_CHUNK_SIZE = 256

MAX_LMCACHE_SERVER_URL_BYTES = 2_048
MAX_SIDECAR_PATH_BYTES = 4_096
MAX_IDENTITY_FIELD_BYTES = 1_024
MAX_STAGING_TOKEN_CAPACITY = 131_072
MAX_REQUEST_TIMEOUT_SECONDS = 300.0

_LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_FLOW_KEYS = frozenset({"mode"})
_TRANSFER_KEYS = frozenset(
    {
        "mode",
        "lmcache_server_url",
        "sidecar_path",
        "lmcache_server_attestation",
        "model_revision",
        "tokenizer_revision",
        "model_config_digest",
        "kv_cache_config_digest",
        "adapter_revision",
        "staging_token_capacity",
        "request_timeout_seconds",
        "transfer_failure_policy",
    }
)
_TRANSFER_OPTIONAL_KEYS = frozenset(
    {"transfer_evidence_path", "disable_kv_scatter", "allow_prefix_caching"}
)
_SELECTIVE_KEYS = frozenset(
    {"check_layer", "recompute_ratio", "suffix_tokens"}
)
_ATTESTATION_KEYS = frozenset(
    {
        "lmcache_version",
        "source_commit",
        "protocol",
        "hash_algorithm",
    }
)


class ConnectorTransferMode(str, Enum):
    """The only connector modes recognized by the pinned implementation."""

    CONTROL_FLOW = "control_flow"
    COMPATIBILITY_PROBE = "compatibility_probe"
    TRANSFER_100PCT = "transfer_100pct"
    TRANSFER_SELECTIVE = "transfer_selective"


class TransferFailurePolicy(str, Enum):
    """The only safe transfer failure behavior before selective reuse."""

    FULL_PREFILL = "full_prefill"


class TransferConfigErrorCode(str, Enum):
    """Stable configuration failures that never embed supplied values."""

    INVALID_EXTRA_CONFIG_TYPE = "invalid_extra_config_type"
    MODE_MISSING = "mode_missing"
    MODE_UNSUPPORTED = "mode_unsupported"
    UNKNOWN_TOP_LEVEL_KEYS = "unknown_top_level_keys"
    MISSING_TRANSFER_KEYS = "missing_transfer_keys"
    INVALID_ATTESTATION_TYPE = "invalid_attestation_type"
    UNKNOWN_ATTESTATION_KEYS = "unknown_attestation_keys"
    MISSING_ATTESTATION_KEYS = "missing_attestation_keys"
    ATTESTATION_MISMATCH = "attestation_mismatch"
    INVALID_LMCACHE_SERVER_URL = "invalid_lmcache_server_url"
    INVALID_SIDECAR_PATH = "invalid_sidecar_path"
    INVALID_TRANSFER_EVIDENCE_PATH = "invalid_transfer_evidence_path"
    INVALID_IDENTITY_FIELD = "invalid_identity_field"
    INVALID_CONFIG_DIGEST = "invalid_config_digest"
    INVALID_STAGING_TOKEN_CAPACITY = "invalid_staging_token_capacity"
    INVALID_REQUEST_TIMEOUT = "invalid_request_timeout"
    INVALID_TRANSFER_FAILURE_POLICY = "invalid_transfer_failure_policy"
    INVALID_DISABLE_KV_SCATTER = "invalid_disable_kv_scatter"
    INVALID_ALLOW_PREFIX_CACHING = "invalid_allow_prefix_caching"
    INVALID_SELECTIVE_CHECK_LAYER = "invalid_selective_check_layer"
    INVALID_SELECTIVE_RECOMPUTE_RATIO = "invalid_selective_recompute_ratio"
    INVALID_SELECTIVE_SUFFIX_TOKENS = "invalid_selective_suffix_tokens"
    INVALID_SELECTIVE_SCATTER = "invalid_selective_scatter"


class TransferConfigError(ValueError):
    """Fail-closed parse error containing only a bounded code."""

    def __init__(self, code: TransferConfigErrorCode) -> None:
        self.code = code
        super().__init__(f"connector transfer configuration rejected: {code.value}")


def _fail(code: TransferConfigErrorCode) -> NoReturn:
    raise TransferConfigError(code)


def _require_exact_keys(
    value: dict[object, object],
    expected: frozenset[str],
    *,
    unknown_code: TransferConfigErrorCode,
    missing_code: TransferConfigErrorCode,
) -> None:
    actual: set[str] = set()
    for key in value:
        if not isinstance(key, str):
            _fail(unknown_code)
        actual.add(key)
    if actual - expected:
        _fail(unknown_code)
    if expected - actual:
        _fail(missing_code)


def _require_identity_field(value: object) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        _fail(TransferConfigErrorCode.INVALID_IDENTITY_FIELD)
    if "\x00" in value or len(value.encode("utf-8")) > MAX_IDENTITY_FIELD_BYTES:
        _fail(TransferConfigErrorCode.INVALID_IDENTITY_FIELD)
    return value


def _require_config_digest(value: object) -> str:
    if not isinstance(value, str) or _LOWER_SHA256_PATTERN.fullmatch(value) is None:
        _fail(TransferConfigErrorCode.INVALID_CONFIG_DIGEST)
    return value


def _is_structurally_valid_server_url(value: str) -> bool:
    """Parse the endpoint without letting parser details escape in an error."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    return not (
        parsed.scheme != "tcp"
        or not parsed.netloc
        or hostname is None
        or not hostname
        or port is None
        or not 1 <= port <= 65_535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    )


def _require_lmcache_server_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        _fail(TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL)
    if (
        len(value.encode("utf-8")) > MAX_LMCACHE_SERVER_URL_BYTES
        or not value.startswith("tcp://")
        or "%" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        _fail(TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL)
    if not _is_structurally_valid_server_url(value):
        _fail(TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL)
    return value


def _require_sidecar_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(TransferConfigErrorCode.INVALID_SIDECAR_PATH)
    if (
        len(value.encode("utf-8")) > MAX_SIDECAR_PATH_BYTES
        or not os.path.isabs(value)
    ):
        _fail(TransferConfigErrorCode.INVALID_SIDECAR_PATH)
    return value


def _require_transfer_evidence_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(TransferConfigErrorCode.INVALID_TRANSFER_EVIDENCE_PATH)
    if (
        len(value.encode("utf-8")) > MAX_SIDECAR_PATH_BYTES
        or not os.path.isabs(value)
    ):
        _fail(TransferConfigErrorCode.INVALID_TRANSFER_EVIDENCE_PATH)
    return value


def _require_disable_kv_scatter(value: object) -> bool:
    if not isinstance(value, bool):
        _fail(TransferConfigErrorCode.INVALID_DISABLE_KV_SCATTER)
    return value


def _require_selective_check_layer(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 24
    ):
        _fail(TransferConfigErrorCode.INVALID_SELECTIVE_CHECK_LAYER)
    return value


def _require_selective_recompute_ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(TransferConfigErrorCode.INVALID_SELECTIVE_RECOMPUTE_RATIO)
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        _fail(TransferConfigErrorCode.INVALID_SELECTIVE_RECOMPUTE_RATIO)
    return ratio


def _require_selective_suffix_tokens(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 131_072
    ):
        _fail(TransferConfigErrorCode.INVALID_SELECTIVE_SUFFIX_TOKENS)
    return value


def _require_staging_capacity(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not LMCACHE_CHUNK_SIZE <= value <= MAX_STAGING_TOKEN_CAPACITY
        or value % LMCACHE_CHUNK_SIZE != 0
    ):
        _fail(TransferConfigErrorCode.INVALID_STAGING_TOKEN_CAPACITY)
    return value


def _require_request_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(TransferConfigErrorCode.INVALID_REQUEST_TIMEOUT)
    timeout = float(value)
    if (
        not math.isfinite(timeout)
        or timeout <= 0.0
        or timeout > MAX_REQUEST_TIMEOUT_SECONDS
    ):
        _fail(TransferConfigErrorCode.INVALID_REQUEST_TIMEOUT)
    return timeout


@dataclass(frozen=True, slots=True)
class PinnedLmcacheServerAttestation:
    """Explicit identity of the separately launched pinned LMCache server."""

    lmcache_version: str
    source_commit: str
    protocol: str
    hash_algorithm: str

    def __post_init__(self) -> None:
        if (
            self.lmcache_version != PINNED_TARGET.lmcache_version
            or self.source_commit != LMCACHE_SOURCE_COMMIT
            or self.protocol != LMCACHE_BLEND_PROTOCOL
            or self.hash_algorithm != LMCACHE_HASH_ALGORITHM
        ):
            _fail(TransferConfigErrorCode.ATTESTATION_MISMATCH)


@dataclass(frozen=True, slots=True)
class ControlFlowTransferConfig:
    """Transfer-disabled connector configuration used by default."""

    mode: ConnectorTransferMode = field(
        default=ConnectorTransferMode.CONTROL_FLOW,
        init=False,
    )

    @property
    def transfer_enabled(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CompatibilityProbeConfig:
    """Transfer-disabled startup mode that reports finalized config digests."""

    mode: ConnectorTransferMode = field(
        default=ConnectorTransferMode.COMPATIBILITY_PROBE,
        init=False,
    )

    @property
    def transfer_enabled(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Transfer100PctConfig:
    """Validated configuration for instrumented transfer plus full prefill."""

    lmcache_server_url: str
    sidecar_path: str
    lmcache_server_attestation: PinnedLmcacheServerAttestation
    model_revision: str
    tokenizer_revision: str
    model_config_digest: str
    kv_cache_config_digest: str
    adapter_revision: str
    staging_token_capacity: int
    request_timeout_seconds: float
    transfer_failure_policy: TransferFailurePolicy
    transfer_evidence_path: str | None = None
    disable_kv_scatter: bool = False
    allow_prefix_caching: bool = False
    mode: ConnectorTransferMode = field(
        default=ConnectorTransferMode.TRANSFER_100PCT,
        init=False,
    )
    namespace: CacheNamespace = field(init=False)

    def __post_init__(self) -> None:
        _require_lmcache_server_url(self.lmcache_server_url)
        _require_sidecar_path(self.sidecar_path)
        if not isinstance(
            self.lmcache_server_attestation, PinnedLmcacheServerAttestation
        ):
            _fail(TransferConfigErrorCode.INVALID_ATTESTATION_TYPE)
        model_revision = _require_identity_field(self.model_revision)
        tokenizer_revision = _require_identity_field(self.tokenizer_revision)
        model_digest = _require_config_digest(self.model_config_digest)
        kv_digest = _require_config_digest(self.kv_cache_config_digest)
        adapter_revision = _require_identity_field(self.adapter_revision)
        _require_disable_kv_scatter(self.disable_kv_scatter)
        if not isinstance(self.allow_prefix_caching, bool):
            _fail(TransferConfigErrorCode.INVALID_ALLOW_PREFIX_CACHING)
        _require_staging_capacity(self.staging_token_capacity)
        timeout = _require_request_timeout(self.request_timeout_seconds)
        if self.transfer_failure_policy is not TransferFailurePolicy.FULL_PREFILL:
            _fail(TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY)
        if self.transfer_evidence_path is not None:
            evidence_path = _require_transfer_evidence_path(
                self.transfer_evidence_path
            )
            if evidence_path == self.sidecar_path:
                _fail(TransferConfigErrorCode.INVALID_TRANSFER_EVIDENCE_PATH)
        object.__setattr__(self, "request_timeout_seconds", timeout)
        object.__setattr__(
            self,
            "namespace",
            CacheNamespace(
                schema_version=1,
                model_id=PINNED_TARGET.model_id,
                model_revision=model_revision,
                tokenizer_id=PINNED_TARGET.model_id,
                tokenizer_revision=tokenizer_revision,
                model_config_digest=model_digest,
                kv_cache_config_digest=kv_digest,
                adapter_revision=adapter_revision,
                vllm_version=PINNED_TARGET.vllm_version,
                lmcache_version=PINNED_TARGET.lmcache_version,
                torch_version=PINNED_TARGET.torch_version,
                cuda_runtime=PINNED_TARGET.cuda_runtime,
            ),
        )

    @property
    def transfer_enabled(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class TransferSelectiveConfig(Transfer100PctConfig):
    """Validated opt-in selective row execution configuration.

    This is a real execution mode, but it is intentionally separate from the
    validated ``transfer_100pct`` milestone.  The worker executes through the
    check layer, measures loaded-versus-fresh value differences, and only then
    applies this ratio to later layers.  Every run remains subject to the
    full-prefill correctness gate before any quality claim is made.
    """

    check_layer: int = 1
    recompute_ratio: float = 0.15
    suffix_tokens: int = 32
    mode: ConnectorTransferMode = field(
        default=ConnectorTransferMode.TRANSFER_SELECTIVE,
        init=False,
    )

    def __post_init__(self) -> None:
        Transfer100PctConfig.__post_init__(self)
        _require_selective_check_layer(self.check_layer)
        ratio = _require_selective_recompute_ratio(self.recompute_ratio)
        _require_selective_suffix_tokens(self.suffix_tokens)
        if self.disable_kv_scatter:
            _fail(TransferConfigErrorCode.INVALID_SELECTIVE_SCATTER)
        object.__setattr__(self, "recompute_ratio", ratio)


ConnectorTransferConfig = (
    ControlFlowTransferConfig
    | CompatibilityProbeConfig
    | Transfer100PctConfig
    | TransferSelectiveConfig
)


def _parse_attestation(value: object) -> PinnedLmcacheServerAttestation:
    if not isinstance(value, dict):
        _fail(TransferConfigErrorCode.INVALID_ATTESTATION_TYPE)
    _require_exact_keys(
        value,
        _ATTESTATION_KEYS,
        unknown_code=TransferConfigErrorCode.UNKNOWN_ATTESTATION_KEYS,
        missing_code=TransferConfigErrorCode.MISSING_ATTESTATION_KEYS,
    )
    return PinnedLmcacheServerAttestation(
        lmcache_version=value["lmcache_version"],
        source_commit=value["source_commit"],
        protocol=value["protocol"],
        hash_algorithm=value["hash_algorithm"],
    )


def parse_connector_extra_config(
    extra_config: object | None,
) -> ConnectorTransferConfig:
    """Parse a JSON-shaped connector extra-config dictionary fail closed."""

    if extra_config is None:
        return ControlFlowTransferConfig()
    if not isinstance(extra_config, dict):
        _fail(TransferConfigErrorCode.INVALID_EXTRA_CONFIG_TYPE)
    if not extra_config:
        return ControlFlowTransferConfig()
    mode = extra_config.get("mode")
    if mode is None:
        _fail(TransferConfigErrorCode.MODE_MISSING)
    if mode == ConnectorTransferMode.CONTROL_FLOW.value:
        _require_exact_keys(
            extra_config,
            _CONTROL_FLOW_KEYS,
            unknown_code=TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
            missing_code=TransferConfigErrorCode.MODE_MISSING,
        )
        return ControlFlowTransferConfig()
    if mode == ConnectorTransferMode.COMPATIBILITY_PROBE.value:
        _require_exact_keys(
            extra_config,
            _CONTROL_FLOW_KEYS,
            unknown_code=TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
            missing_code=TransferConfigErrorCode.MODE_MISSING,
        )
        return CompatibilityProbeConfig()
    if mode not in (
        ConnectorTransferMode.TRANSFER_100PCT.value,
        ConnectorTransferMode.TRANSFER_SELECTIVE.value,
    ):
        _fail(TransferConfigErrorCode.MODE_UNSUPPORTED)

    actual_keys = set(extra_config)
    allowed_keys = _TRANSFER_KEYS | _TRANSFER_OPTIONAL_KEYS
    if mode == ConnectorTransferMode.TRANSFER_SELECTIVE.value:
        allowed_keys |= _SELECTIVE_KEYS
    if any(not isinstance(key, str) for key in extra_config) or (
        actual_keys - allowed_keys
    ):
        _fail(TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS)
    if _TRANSFER_KEYS - actual_keys:
        _fail(TransferConfigErrorCode.MISSING_TRANSFER_KEYS)
    if mode == ConnectorTransferMode.TRANSFER_SELECTIVE.value and (
        _SELECTIVE_KEYS - actual_keys
    ):
        _fail(TransferConfigErrorCode.MISSING_TRANSFER_KEYS)
    failure_policy = extra_config["transfer_failure_policy"]
    if failure_policy != TransferFailurePolicy.FULL_PREFILL.value:
        _fail(TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY)
    config_type = (
        TransferSelectiveConfig
        if mode == ConnectorTransferMode.TRANSFER_SELECTIVE.value
        else Transfer100PctConfig
    )
    selective_kwargs = (
        {
            "check_layer": extra_config["check_layer"],
            "recompute_ratio": extra_config["recompute_ratio"],
            "suffix_tokens": extra_config["suffix_tokens"],
        }
        if config_type is TransferSelectiveConfig
        else {}
    )
    return config_type(
        lmcache_server_url=extra_config["lmcache_server_url"],
        sidecar_path=extra_config["sidecar_path"],
        lmcache_server_attestation=_parse_attestation(
            extra_config["lmcache_server_attestation"]
        ),
        model_revision=extra_config["model_revision"],
        tokenizer_revision=extra_config["tokenizer_revision"],
        model_config_digest=extra_config["model_config_digest"],
        kv_cache_config_digest=extra_config["kv_cache_config_digest"],
        adapter_revision=extra_config["adapter_revision"],
        staging_token_capacity=extra_config["staging_token_capacity"],
        request_timeout_seconds=extra_config["request_timeout_seconds"],
        transfer_failure_policy=TransferFailurePolicy.FULL_PREFILL,
        transfer_evidence_path=extra_config.get("transfer_evidence_path"),
        disable_kv_scatter=extra_config.get("disable_kv_scatter", False),
        allow_prefix_caching=extra_config.get("allow_prefix_caching", False),
        **selective_kwargs,
    )


__all__ = [
    "LMCACHE_BLEND_PROTOCOL",
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_HASH_ALGORITHM",
    "LMCACHE_SOURCE_COMMIT",
    "MAX_IDENTITY_FIELD_BYTES",
    "MAX_LMCACHE_SERVER_URL_BYTES",
    "MAX_REQUEST_TIMEOUT_SECONDS",
    "MAX_SIDECAR_PATH_BYTES",
    "MAX_STAGING_TOKEN_CAPACITY",
    "CompatibilityProbeConfig",
    "ConnectorTransferConfig",
    "ConnectorTransferMode",
    "ControlFlowTransferConfig",
    "PinnedLmcacheServerAttestation",
    "Transfer100PctConfig",
    "TransferConfigError",
    "TransferConfigErrorCode",
    "TransferFailurePolicy",
    "TransferSelectiveConfig",
    "parse_connector_extra_config",
]
