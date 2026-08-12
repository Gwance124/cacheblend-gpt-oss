# SPDX-License-Identifier: Apache-2.0
"""Strict connector-extra configuration for the 100%-recompute milestone.

The version connector receives this value from vLLM 0.19.1's
``KVTransferConfig.kv_connector_extra_config``:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/kv_transfer.py#L20-L65

This module intentionally imports neither vLLM, LMCache, Torch, nor CUDA.
Empty configuration keeps the connector in its transfer-disabled control-flow
mode.  Transfer is enabled only by the complete exact ``transfer_100pct``
schema; configuration drift fails before any cache lookup or transport starts.
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
    TRANSFER_100PCT = "transfer_100pct"


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
    INVALID_IDENTITY_FIELD = "invalid_identity_field"
    INVALID_CONFIG_DIGEST = "invalid_config_digest"
    INVALID_STAGING_TOKEN_CAPACITY = "invalid_staging_token_capacity"
    INVALID_REQUEST_TIMEOUT = "invalid_request_timeout"
    INVALID_TRANSFER_FAILURE_POLICY = "invalid_transfer_failure_policy"


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
        _require_staging_capacity(self.staging_token_capacity)
        timeout = _require_request_timeout(self.request_timeout_seconds)
        if self.transfer_failure_policy is not TransferFailurePolicy.FULL_PREFILL:
            _fail(TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY)
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


ConnectorTransferConfig = ControlFlowTransferConfig | Transfer100PctConfig


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
    if mode != ConnectorTransferMode.TRANSFER_100PCT.value:
        _fail(TransferConfigErrorCode.MODE_UNSUPPORTED)

    _require_exact_keys(
        extra_config,
        _TRANSFER_KEYS,
        unknown_code=TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
        missing_code=TransferConfigErrorCode.MISSING_TRANSFER_KEYS,
    )
    failure_policy = extra_config["transfer_failure_policy"]
    if failure_policy != TransferFailurePolicy.FULL_PREFILL.value:
        _fail(TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY)
    return Transfer100PctConfig(
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
    "ConnectorTransferConfig",
    "ConnectorTransferMode",
    "ControlFlowTransferConfig",
    "PinnedLmcacheServerAttestation",
    "Transfer100PctConfig",
    "TransferConfigError",
    "TransferConfigErrorCode",
    "TransferFailurePolicy",
    "parse_connector_extra_config",
]
