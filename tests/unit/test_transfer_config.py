from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from cacheblend_gpt_oss.targets import PINNED_TARGET
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    MAX_IDENTITY_FIELD_BYTES,
    MAX_LMCACHE_SERVER_URL_BYTES,
    MAX_REQUEST_TIMEOUT_SECONDS,
    MAX_SIDECAR_PATH_BYTES,
    CompatibilityProbeConfig,
    ConnectorTransferMode,
    ControlFlowTransferConfig,
    PinnedLmcacheServerAttestation,
    Transfer100PctConfig,
    TransferConfigError,
    TransferConfigErrorCode,
    TransferFailurePolicy,
    parse_connector_extra_config,
)


def _valid_config() -> dict[str, object]:
    return {
        "mode": "transfer_100pct",
        "lmcache_server_url": "tcp://127.0.0.1:5555",
        "sidecar_path": "/var/lib/cacheblend/sidecar.sqlite3",
        "lmcache_server_attestation": {
            "lmcache_version": "0.4.3",
            "source_commit": LMCACHE_SOURCE_COMMIT,
            "protocol": LMCACHE_BLEND_PROTOCOL,
            "hash_algorithm": LMCACHE_HASH_ALGORITHM,
        },
        "model_revision": "model-immutable-revision",
        "tokenizer_revision": "tokenizer-immutable-revision",
        "model_config_digest": "a" * 64,
        "kv_cache_config_digest": "b" * 64,
        "adapter_revision": "cacheblend-adapter-revision",
        "staging_token_capacity": 1024,
        "request_timeout_seconds": 10,
        "transfer_failure_policy": "full_prefill",
    }


def _assert_error(
    expected: TransferConfigErrorCode,
    operation: Callable[[], object],
) -> TransferConfigError:
    with pytest.raises(TransferConfigError) as caught:
        operation()
    assert caught.value.code is expected
    assert str(caught.value).endswith(expected.value)
    return caught.value


@pytest.mark.parametrize("raw", [None, {}])
def test_absent_extra_config_defaults_to_inert_control_flow(raw: object) -> None:
    parsed = parse_connector_extra_config(raw)

    assert isinstance(parsed, ControlFlowTransferConfig)
    assert parsed.mode is ConnectorTransferMode.CONTROL_FLOW
    assert not parsed.transfer_enabled


def test_explicit_control_flow_accepts_only_the_mode_key() -> None:
    parsed = parse_connector_extra_config({"mode": "control_flow"})
    assert parsed == ControlFlowTransferConfig()

    _assert_error(
        TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
        lambda: parse_connector_extra_config(
            {"mode": "control_flow", "sidecar_path": "/not-used"}
        ),
    )


def test_compatibility_probe_is_explicit_inert_and_has_no_extra_keys() -> None:
    parsed = parse_connector_extra_config({"mode": "compatibility_probe"})
    assert parsed == CompatibilityProbeConfig()
    assert parsed.mode is ConnectorTransferMode.COMPATIBILITY_PROBE
    assert not parsed.transfer_enabled

    _assert_error(
        TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
        lambda: parse_connector_extra_config(
            {"mode": "compatibility_probe", "sidecar_path": "/not-used"}
        ),
    )


def test_valid_transfer_config_is_frozen_and_builds_pinned_namespace() -> None:
    raw = _valid_config()
    parsed = parse_connector_extra_config(raw)

    assert isinstance(parsed, Transfer100PctConfig)
    assert parsed.mode is ConnectorTransferMode.TRANSFER_100PCT
    assert parsed.transfer_enabled
    assert parsed.lmcache_server_url == "tcp://127.0.0.1:5555"
    assert parsed.sidecar_path == "/var/lib/cacheblend/sidecar.sqlite3"
    assert parsed.transfer_evidence_path is None
    assert parsed.staging_token_capacity == 1024
    assert parsed.request_timeout_seconds == 10.0
    assert parsed.transfer_failure_policy is TransferFailurePolicy.FULL_PREFILL
    assert parsed.lmcache_server_attestation == PinnedLmcacheServerAttestation(
        lmcache_version="0.4.3",
        source_commit=LMCACHE_SOURCE_COMMIT,
        protocol=LMCACHE_BLEND_PROTOCOL,
        hash_algorithm=LMCACHE_HASH_ALGORITHM,
    )

    namespace = parsed.namespace
    assert namespace.schema_version == 1
    assert namespace.model_id == PINNED_TARGET.model_id
    assert namespace.model_revision == "model-immutable-revision"
    assert namespace.tokenizer_id == PINNED_TARGET.model_id
    assert namespace.tokenizer_revision == "tokenizer-immutable-revision"
    assert namespace.model_config_digest == "a" * 64
    assert namespace.kv_cache_config_digest == "b" * 64
    assert namespace.adapter_revision == "cacheblend-adapter-revision"
    assert namespace.vllm_version == PINNED_TARGET.vllm_version
    assert namespace.lmcache_version == PINNED_TARGET.lmcache_version
    assert namespace.torch_version == PINNED_TARGET.torch_version
    assert namespace.cuda_runtime == PINNED_TARGET.cuda_runtime
    with pytest.raises(FrozenInstanceError):
        parsed.staging_token_capacity = 2048  # type: ignore[misc]


def test_transfer_evidence_path_is_optional_absolute_and_separate() -> None:
    raw = _valid_config()
    raw["transfer_evidence_path"] = "/var/lib/cacheblend/transfer-evidence.json"
    parsed = parse_connector_extra_config(raw)

    assert isinstance(parsed, Transfer100PctConfig)
    assert (
        parsed.transfer_evidence_path
        == "/var/lib/cacheblend/transfer-evidence.json"
    )

    for invalid in ("relative.json", "", raw["sidecar_path"]):
        rejected = _valid_config()
        rejected["transfer_evidence_path"] = invalid
        _assert_error(
            TransferConfigErrorCode.INVALID_TRANSFER_EVIDENCE_PATH,
            lambda rejected=rejected: parse_connector_extra_config(rejected),
        )


def test_nested_input_is_copied_not_retained() -> None:
    raw = _valid_config()
    parsed = parse_connector_extra_config(raw)
    attestation = raw["lmcache_server_attestation"]
    assert isinstance(attestation, dict)
    attestation["source_commit"] = "changed-after-parse"

    assert isinstance(parsed, Transfer100PctConfig)
    assert parsed.lmcache_server_attestation.source_commit == LMCACHE_SOURCE_COMMIT


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "transfer_100pct",
        1,
        True,
    ],
)
def test_extra_config_must_be_a_dictionary(raw: object) -> None:
    _assert_error(
        TransferConfigErrorCode.INVALID_EXTRA_CONFIG_TYPE,
        lambda: parse_connector_extra_config(raw),
    )


def test_mode_is_required_and_other_modes_are_rejected() -> None:
    _assert_error(
        TransferConfigErrorCode.MODE_MISSING,
        lambda: parse_connector_extra_config({"sidecar_path": "/tmp/cache"}),
    )
    for mode in ("selective", "transfer", "TRANSFER_100PCT", 1, True):
        _assert_error(
            TransferConfigErrorCode.MODE_UNSUPPORTED,
            lambda mode=mode: parse_connector_extra_config({"mode": mode}),
        )


def test_transfer_rejects_every_missing_key_and_any_unknown_key() -> None:
    valid = _valid_config()
    for required_key in tuple(valid):
        if required_key == "mode":
            continue
        missing = dict(valid)
        del missing[required_key]
        _assert_error(
            TransferConfigErrorCode.MISSING_TRANSFER_KEYS,
            lambda missing=missing: parse_connector_extra_config(missing),
        )

    unknown = dict(valid)
    unknown["future_unsafe_option"] = True
    _assert_error(
        TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
        lambda: parse_connector_extra_config(unknown),
    )


def test_attestation_requires_exact_shape_and_pinned_values() -> None:
    for bad_value in (None, [], "attested"):
        raw = _valid_config()
        raw["lmcache_server_attestation"] = bad_value
        _assert_error(
            TransferConfigErrorCode.INVALID_ATTESTATION_TYPE,
            lambda raw=raw: parse_connector_extra_config(raw),
        )

    missing = _valid_config()
    missing_attestation = missing["lmcache_server_attestation"]
    assert isinstance(missing_attestation, dict)
    del missing_attestation["protocol"]
    _assert_error(
        TransferConfigErrorCode.MISSING_ATTESTATION_KEYS,
        lambda: parse_connector_extra_config(missing),
    )

    unknown = _valid_config()
    unknown_attestation = unknown["lmcache_server_attestation"]
    assert isinstance(unknown_attestation, dict)
    unknown_attestation["image_tag"] = "private"
    _assert_error(
        TransferConfigErrorCode.UNKNOWN_ATTESTATION_KEYS,
        lambda: parse_connector_extra_config(unknown),
    )

    for field_name in (
        "lmcache_version",
        "source_commit",
        "protocol",
        "hash_algorithm",
    ):
        mismatch = _valid_config()
        mismatch_attestation = mismatch["lmcache_server_attestation"]
        assert isinstance(mismatch_attestation, dict)
        mismatch_attestation[field_name] = "wrong-value"
        _assert_error(
            TransferConfigErrorCode.ATTESTATION_MISMATCH,
            lambda mismatch=mismatch: parse_connector_extra_config(mismatch),
        )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://127.0.0.1:5555",
        "TCP://127.0.0.1:5555",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:0",
        "tcp://127.0.0.1:65536",
        "tcp://127.0.0.1:secret-port",
        "tcp://user@127.0.0.1:5555",
        "tcp://user:password@127.0.0.1:5555",
        "tcp://127.0.0.1:5555?token=secret",
        "tcp://127.0.0.1:5555#secret",
        "tcp://127.0.0.1:5555/path",
        "tcp://127.0.0.1:5555/",
        "tcp://host%40name:5555",
        "tcp://127.0.0.1:55 55",
        True,
        5555,
    ],
)
def test_server_url_rejects_non_tcp_ambiguous_or_secret_values(url: object) -> (
    None
):
    raw = _valid_config()
    raw["lmcache_server_url"] = url
    _assert_error(
        TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL,
        lambda: parse_connector_extra_config(raw),
    )


def test_server_url_accepts_dns_and_bracketed_ipv6_and_is_bounded() -> None:
    for url in ("tcp://cache.internal:5555", "tcp://[::1]:5555"):
        raw = _valid_config()
        raw["lmcache_server_url"] = url
        parsed = parse_connector_extra_config(raw)
        assert isinstance(parsed, Transfer100PctConfig)
        assert parsed.lmcache_server_url == url

    too_long = _valid_config()
    too_long["lmcache_server_url"] = (
        "tcp://" + "a" * MAX_LMCACHE_SERVER_URL_BYTES + ":5555"
    )
    _assert_error(
        TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL,
        lambda: parse_connector_extra_config(too_long),
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "relative/sidecar.sqlite3",
        "./sidecar.sqlite3",
        True,
        10,
        "/tmp/invalid\x00suffix",
    ],
)
def test_sidecar_path_must_be_nonempty_absolute_and_bounded(path: object) -> None:
    raw = _valid_config()
    raw["sidecar_path"] = path
    _assert_error(
        TransferConfigErrorCode.INVALID_SIDECAR_PATH,
        lambda: parse_connector_extra_config(raw),
    )

    too_long = _valid_config()
    too_long["sidecar_path"] = "/" + "s" * MAX_SIDECAR_PATH_BYTES
    _assert_error(
        TransferConfigErrorCode.INVALID_SIDECAR_PATH,
        lambda: parse_connector_extra_config(too_long),
    )


@pytest.mark.parametrize(
    "field_name",
    ["model_revision", "tokenizer_revision", "adapter_revision"],
)
@pytest.mark.parametrize(
    "value",
    ["", "   ", True, 1, "revision\x00suffix"],
)
def test_identity_fields_are_bounded_nonempty_strings(
    field_name: str,
    value: object,
) -> None:
    raw = _valid_config()
    raw[field_name] = value
    _assert_error(
        TransferConfigErrorCode.INVALID_IDENTITY_FIELD,
        lambda: parse_connector_extra_config(raw),
    )


def test_identity_field_byte_limit_is_enforced() -> None:
    raw = _valid_config()
    raw["model_revision"] = "r" * (MAX_IDENTITY_FIELD_BYTES + 1)
    _assert_error(
        TransferConfigErrorCode.INVALID_IDENTITY_FIELD,
        lambda: parse_connector_extra_config(raw),
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        True,
        1,
    ],
)
@pytest.mark.parametrize(
    "field_name",
    ["model_config_digest", "kv_cache_config_digest"],
)
def test_config_digests_require_exact_lowercase_sha256(
    field_name: str,
    value: object,
) -> None:
    raw = _valid_config()
    raw[field_name] = value
    _assert_error(
        TransferConfigErrorCode.INVALID_CONFIG_DIGEST,
        lambda: parse_connector_extra_config(raw),
    )


@pytest.mark.parametrize(
    "capacity",
    [True, 255, 257, 131_328, 256.0, "1024"],
)
def test_staging_capacity_is_a_bounded_chunk_multiple(capacity: object) -> None:
    raw = _valid_config()
    raw["staging_token_capacity"] = capacity
    _assert_error(
        TransferConfigErrorCode.INVALID_STAGING_TOKEN_CAPACITY,
        lambda: parse_connector_extra_config(raw),
    )


@pytest.mark.parametrize("capacity", [256, 512, 131_072])
def test_staging_capacity_accepts_inclusive_chunk_aligned_bounds(
    capacity: int,
) -> None:
    raw = _valid_config()
    raw["staging_token_capacity"] = capacity
    parsed = parse_connector_extra_config(raw)
    assert isinstance(parsed, Transfer100PctConfig)
    assert parsed.staging_token_capacity == capacity


@pytest.mark.parametrize(
    "timeout",
    [
        True,
        0,
        -1,
        float("nan"),
        float("inf"),
        MAX_REQUEST_TIMEOUT_SECONDS + 0.001,
        "10",
    ],
)
def test_timeout_must_be_finite_positive_and_bounded(timeout: object) -> None:
    raw = _valid_config()
    raw["request_timeout_seconds"] = timeout
    _assert_error(
        TransferConfigErrorCode.INVALID_REQUEST_TIMEOUT,
        lambda: parse_connector_extra_config(raw),
    )


@pytest.mark.parametrize("timeout", [0.001, 1, MAX_REQUEST_TIMEOUT_SECONDS])
def test_timeout_accepts_positive_inclusive_upper_bound(timeout: float) -> None:
    raw = _valid_config()
    raw["request_timeout_seconds"] = timeout
    parsed = parse_connector_extra_config(raw)
    assert isinstance(parsed, Transfer100PctConfig)
    assert math.isclose(parsed.request_timeout_seconds, float(timeout))


@pytest.mark.parametrize("policy", ["fail", "recompute", "reject", None, True])
def test_transfer_failure_policy_is_full_prefill_only(policy: object) -> None:
    raw = _valid_config()
    raw["transfer_failure_policy"] = policy
    _assert_error(
        TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY,
        lambda: parse_connector_extra_config(raw),
    )


def test_direct_dataclass_construction_remains_fail_closed() -> None:
    valid = parse_connector_extra_config(_valid_config())
    assert isinstance(valid, Transfer100PctConfig)
    _assert_error(
        TransferConfigErrorCode.INVALID_STAGING_TOKEN_CAPACITY,
        lambda: replace(valid, staging_token_capacity=True),
    )
    _assert_error(
        TransferConfigErrorCode.INVALID_TRANSFER_FAILURE_POLICY,
        lambda: Transfer100PctConfig(
            lmcache_server_url=valid.lmcache_server_url,
            sidecar_path=valid.sidecar_path,
            lmcache_server_attestation=valid.lmcache_server_attestation,
            model_revision=valid.model_revision,
            tokenizer_revision=valid.tokenizer_revision,
            model_config_digest=valid.model_config_digest,
            kv_cache_config_digest=valid.kv_cache_config_digest,
            adapter_revision=valid.adapter_revision,
            staging_token_capacity=valid.staging_token_capacity,
            request_timeout_seconds=valid.request_timeout_seconds,
            transfer_failure_policy="full_prefill",  # type: ignore[arg-type]
        ),
    )


def test_errors_never_include_supplied_secrets_or_unknown_keys() -> None:
    raw = _valid_config()
    secret = "do-not-log-this-secret"
    raw["lmcache_server_url"] = f"tcp://user:{secret}@host:5555"
    error = _assert_error(
        TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL,
        lambda: parse_connector_extra_config(raw),
    )
    assert secret not in str(error)

    raw = _valid_config()
    raw["lmcache_server_url"] = f"tcp://host:{secret}"
    error = _assert_error(
        TransferConfigErrorCode.INVALID_LMCACHE_SERVER_URL,
        lambda: parse_connector_extra_config(raw),
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in str(error)

    raw = _valid_config()
    raw[f"unknown-{secret}"] = secret
    error = _assert_error(
        TransferConfigErrorCode.UNKNOWN_TOP_LEVEL_KEYS,
        lambda: parse_connector_extra_config(raw),
    )
    assert secret not in str(error)


def test_module_does_not_import_runtime_dependencies() -> None:
    source = Path(
        "src/cacheblend_gpt_oss/vllm_compat/v0_19_1/transfer_config.py"
    ).read_text(encoding="utf-8")
    assert "\nimport vllm" not in source
    assert "\nfrom vllm" not in source
    assert "\nimport lmcache" not in source
    assert "\nfrom lmcache" not in source
    assert "\nimport torch" not in source
    assert "\nfrom torch" not in source
