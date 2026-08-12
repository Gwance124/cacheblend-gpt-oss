from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest

from cacheblend_gpt_oss.planner import (
    CacheNamespace,
    CacheRecord,
    MatchPlanner,
    SegmentFingerprint,
    TokenRange,
    TokenSegment,
)
from cacheblend_gpt_oss.planner.fingerprint import SegmentFingerprinter
from cacheblend_gpt_oss.storage import (
    MAX_CACHE_KEY_BYTES,
    MAX_RECORDS_PER_TRANSACTION,
    SIDECAR_APPLICATION_ID,
    SIDECAR_SCHEMA_VERSION,
    SidecarClosedError,
    SidecarConfigurationError,
    SidecarCorruptionError,
    SidecarErrorCode,
    SidecarMode,
    SidecarModeError,
    SidecarSchemaError,
    SqliteSidecarIndex,
)


def namespace() -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="model-config-sha256",
        kv_cache_config_digest="hybrid-cache-config-sha256",
        adapter_revision="adapter-revision",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def record(
    *,
    fingerprint_byte: int = 1,
    token_ids: tuple[int, ...] = (11, 12, 13),
    source_start: int = 100,
    cache_key: str = "lmcache:record-a",
    cache_namespace: CacheNamespace | None = None,
) -> CacheRecord:
    return CacheRecord(
        namespace=cache_namespace or namespace(),
        fingerprint=SegmentFingerprint(bytes([fingerprint_byte]) * 32),
        token_ids=token_ids,
        source_range=TokenRange(source_start, source_start + len(token_ids)),
        cache_key=cache_key,
    )


def database_path(tmp_path: Path) -> Path:
    return tmp_path / "exact-token-sidecar.sqlite3"


class ConstantFingerprinter(SegmentFingerprinter):
    def __init__(self, fingerprint: SegmentFingerprint) -> None:
        self._fingerprint = fingerprint

    def fingerprint(
        self, namespace: CacheNamespace, token_ids: Iterable[int]
    ) -> SegmentFingerprint:
        del namespace, token_ids
        return self._fingerprint


def test_worker_creates_versioned_wal_database_and_persists_u64_tokens(
    tmp_path: Path,
) -> None:
    path = database_path(tmp_path)
    expected = record(token_ids=(0, (1 << 63) + 7, (1 << 64) - 1))

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        assert writer.add(expected)
        assert not writer.add(expected)
        assert writer.lookup(expected.namespace, expected.fingerprint) == (expected,)

    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA application_id").fetchone()[0] == (
            SIDECAR_APPLICATION_ID
        )
        assert raw.execute("PRAGMA user_version").fetchone()[0] == (
            SIDECAR_SCHEMA_VERSION
        )
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    with SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY) as reader:
        assert reader.lookup(expected.namespace, expected.fingerprint) == (expected,)


def test_scheduler_reads_committed_wal_updates_while_worker_is_open(
    tmp_path: Path,
) -> None:
    path = database_path(tmp_path)
    first = record(cache_key="a")
    second = record(cache_key="b", source_start=200)

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        writer.add(first)
        with SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY) as reader:
            assert reader.lookup(namespace(), first.fingerprint) == (first,)
            writer.add(second)
            assert reader.lookup(namespace(), first.fingerprint) == (first, second)


def test_collision_bucket_preserves_all_exact_token_records(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    shared_fingerprint = SegmentFingerprint(b"\x00" * 32)
    records = (
        replace(
            record(cache_key="z", token_ids=(1, 2, 3)),
            fingerprint=shared_fingerprint,
        ),
        replace(
            record(cache_key="a", token_ids=(7, 8, 9)),
            fingerprint=shared_fingerprint,
        ),
    )

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        assert writer.add_many(records) == 2

    with SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY) as reader:
        bucket = reader.lookup(namespace(), shared_fingerprint)
        assert [item.cache_key for item in bucket] == ["a", "z"]
        assert {item.token_ids for item in bucket} == {(1, 2, 3), (7, 8, 9)}

        # The sidecar implements RecordLookup; exact-token verification remains
        # the MatchPlanner's independent final gate within this collision bucket.
        plan = MatchPlanner(
            reader, fingerprinter=ConstantFingerprinter(shared_fingerprint)
        ).plan(
            namespace(), [TokenSegment.at(500, (1, 2, 3))]
        )
        assert [item.record.cache_key for item in plan.matches] == ["z"]
        assert len(plan.rejected_candidates) == 1
        assert plan.rejected_candidates[0].candidate.record.cache_key == "a"


def test_namespace_is_part_of_lookup_identity(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    expected = record()
    other_namespace = replace(namespace(), model_revision="other-revision")

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as index:
        index.add(expected)
        assert index.lookup(other_namespace, expected.fingerprint) == ()


def test_read_only_mode_rejects_writes_and_missing_database(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    with pytest.raises(SidecarConfigurationError) as missing:
        SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY)
    assert missing.value.code is SidecarErrorCode.DATABASE_NOT_FOUND

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE):
        pass
    with SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY) as reader:
        with pytest.raises(SidecarModeError) as forbidden:
            reader.add(record())
        assert forbidden.value.code is SidecarErrorCode.WRITE_FORBIDDEN


def test_closed_index_rejects_operations(tmp_path: Path) -> None:
    index = SqliteSidecarIndex(
        database_path(tmp_path), SidecarMode.WORKER_READ_WRITE
    )
    index.close()
    index.close()

    with pytest.raises(SidecarClosedError) as caught:
        index.lookup(namespace(), record().fingerprint)
    assert caught.value.code is SidecarErrorCode.CLOSED


def test_add_many_prevalidation_is_atomic(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    valid = record(cache_key="valid")
    oversized_key = "x" * (MAX_CACHE_KEY_BYTES + 1)
    invalid = replace(record(cache_key="placeholder"), cache_key=oversized_key)

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        with pytest.raises(SidecarConfigurationError) as caught:
            writer.add_many((valid, invalid))
        assert caught.value.code is SidecarErrorCode.INVALID_RECORD
        assert writer.lookup(valid.namespace, valid.fingerprint) == ()


def test_add_many_has_a_bounded_transaction_size(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    expected = record()
    oversized_batch = (expected for _ in range(MAX_RECORDS_PER_TRANSACTION + 1))

    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        with pytest.raises(SidecarConfigurationError) as caught:
            writer.add_many(oversized_batch)
        assert caught.value.code is SidecarErrorCode.INVALID_RECORD
        assert writer.lookup(expected.namespace, expected.fingerprint) == ()


def test_relative_memory_and_overlong_paths_are_rejected(tmp_path: Path) -> None:
    del tmp_path
    for bad_path in ("relative.sqlite3", ":memory:", "/" + "x" * 4_096):
        with pytest.raises(SidecarConfigurationError) as caught:
            SqliteSidecarIndex(bad_path, SidecarMode.WORKER_READ_WRITE)
        assert caught.value.code is SidecarErrorCode.INVALID_PATH


@pytest.mark.parametrize(
    ("pragma", "value"),
    [("application_id", 0), ("user_version", SIDECAR_SCHEMA_VERSION + 1)],
)
def test_wrong_application_or_schema_version_fails_closed(
    tmp_path: Path, pragma: str, value: int
) -> None:
    path = database_path(tmp_path)
    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE):
        pass

    with sqlite3.connect(path) as raw:
        raw.execute(f"PRAGMA {pragma} = {value}")
    with pytest.raises(SidecarSchemaError) as caught:
        SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY)
    assert caught.value.code is SidecarErrorCode.SCHEMA_MISMATCH


def test_unrelated_sqlite_schema_is_never_adopted(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(SidecarSchemaError) as caught:
        SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE)
    assert caught.value.code is SidecarErrorCode.SCHEMA_MISMATCH


def test_changed_table_contract_is_a_schema_mismatch(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE):
        pass

    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA writable_schema = ON")
        raw.execute(
            "UPDATE sqlite_schema SET sql = replace(sql, ?, ?) WHERE name = ?",
            ("cache_key TEXT NOT NULL", "cache_key TEXT", "sidecar_records"),
        )
        raw.execute("PRAGMA writable_schema = OFF")

    with pytest.raises(SidecarSchemaError) as caught:
        SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY)
    assert caught.value.code is SidecarErrorCode.SCHEMA_MISMATCH


def test_tampered_exact_tokens_fail_checksum_verification(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    expected = record()
    with SqliteSidecarIndex(path, SidecarMode.WORKER_READ_WRITE) as writer:
        writer.add(expected)

    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA ignore_check_constraints = ON")
        token_blob = raw.execute(
            "SELECT token_ids FROM sidecar_records"
        ).fetchone()[0]
        tampered = token_blob[:-1] + bytes([token_blob[-1] ^ 1])
        raw.execute("UPDATE sidecar_records SET token_ids = ?", (tampered,))

    with SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY) as reader:
        with pytest.raises(SidecarCorruptionError) as caught:
            reader.lookup(expected.namespace, expected.fingerprint)
        assert caught.value.code is SidecarErrorCode.RECORD_CORRUPT


def test_non_sqlite_file_fails_as_corrupt(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(SidecarCorruptionError) as caught:
        SqliteSidecarIndex(path, SidecarMode.SCHEDULER_READ_ONLY)
    assert caught.value.code is SidecarErrorCode.DATABASE_CORRUPT
