# SPDX-License-Identifier: Apache-2.0
"""Persistent exact-token sidecar index backed by stdlib SQLite.

LMCache's rolling-hash match is only a candidate.  This sidecar persists the
complete :class:`~cacheblend_gpt_oss.planner.models.CacheRecord` needed by the
planner to verify namespace, strong fingerprint, and exact token equality
before any KV transfer is accepted.

SQLite is used because it is in the Python standard library and provides the
atomic commits and WAL snapshots required by the deployment topology: one
read-write worker process publishes records while scheduler processes open the
same database with SQLite's enforced ``mode=ro`` URI.  There are no vLLM,
LMCache, Torch, or CUDA imports in this module.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    SegmentFingerprint,
    TokenRange,
    normalize_token_ids,
)

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_APPLICATION_ID = 0x4342474F  # ASCII "CBGO", within SQLite's int32 range.
SIDECAR_TOKEN_ENCODING = "u64be-counted-v1"
SIDECAR_CHECKSUM_ENCODING = "sha256-record-v1"

MAX_SIDECAR_PATH_BYTES = 4_096
MAX_NAMESPACE_FIELD_BYTES = 1_024
MAX_CACHE_KEY_BYTES = 2_048
MAX_TOKENS_PER_RECORD = 131_072
MAX_TOKEN_POSITION = 131_072
MAX_COLLISION_BUCKET_RECORDS = 1_024
MAX_RECORDS_PER_TRANSACTION = 1_024
MAX_TOKENS_PER_TRANSACTION = 1_048_576

_BUSY_TIMEOUT_MILLISECONDS = 5_000
_TOKEN_BLOB_MAGIC = b"CBT1"
_TOKEN_BLOB_HEADER_BYTES = len(_TOKEN_BLOB_MAGIC) + 8
_NAMESPACE_KEY_DOMAIN = b"cacheblend-gpt-oss\x00sidecar-namespace\x00v1\x00"
_RECORD_CHECKSUM_DOMAIN = b"cacheblend-gpt-oss\x00sidecar-record\x00v1\x00"

_METADATA_VALUES = {
    "schema_version": str(SIDECAR_SCHEMA_VERSION),
    "token_encoding": SIDECAR_TOKEN_ENCODING,
    "record_checksum": SIDECAR_CHECKSUM_ENCODING,
}

_CREATE_METADATA_SQL = """
CREATE TABLE sidecar_metadata (
    key TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID
"""

_CREATE_RECORDS_SQL = f"""
CREATE TABLE sidecar_records (
    record_id INTEGER PRIMARY KEY,
    namespace_key BLOB NOT NULL
        CHECK(typeof(namespace_key) = 'blob' AND length(namespace_key) = 32),
    namespace_schema_version INTEGER NOT NULL
        CHECK(namespace_schema_version BETWEEN 1 AND 2147483647),
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    tokenizer_id TEXT NOT NULL,
    tokenizer_revision TEXT NOT NULL,
    model_config_digest TEXT NOT NULL,
    kv_cache_config_digest TEXT NOT NULL,
    adapter_revision TEXT NOT NULL,
    vllm_version TEXT NOT NULL,
    lmcache_version TEXT NOT NULL,
    torch_version TEXT NOT NULL,
    cuda_runtime TEXT NOT NULL,
    fingerprint BLOB NOT NULL
        CHECK(typeof(fingerprint) = 'blob' AND length(fingerprint) = 32),
    cache_key TEXT NOT NULL,
    source_start INTEGER NOT NULL
        CHECK(source_start BETWEEN 0 AND {MAX_TOKEN_POSITION}),
    source_end INTEGER NOT NULL
        CHECK(source_end BETWEEN 1 AND {MAX_TOKEN_POSITION}),
    token_count INTEGER NOT NULL
        CHECK(token_count BETWEEN 1 AND {MAX_TOKENS_PER_RECORD}),
    token_ids BLOB NOT NULL
        CHECK(
            typeof(token_ids) = 'blob'
            AND length(token_ids) = {_TOKEN_BLOB_HEADER_BYTES} + token_count * 8
        ),
    record_checksum BLOB NOT NULL UNIQUE
        CHECK(typeof(record_checksum) = 'blob' AND length(record_checksum) = 32),
    CHECK(source_end - source_start = token_count)
)
"""

_CREATE_LOOKUP_INDEX_SQL = """
CREATE INDEX sidecar_lookup_idx ON sidecar_records (
    namespace_key,
    fingerprint,
    cache_key,
    source_start,
    source_end
)
"""

_EXPECTED_SCHEMA_SQL = {
    "sidecar_metadata": _CREATE_METADATA_SQL.strip(),
    "sidecar_records": _CREATE_RECORDS_SQL.strip(),
    "sidecar_lookup_idx": _CREATE_LOOKUP_INDEX_SQL.strip(),
}

_RECORD_COLUMN_NAMES = (
    "namespace_key",
    "namespace_schema_version",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "model_config_digest",
    "kv_cache_config_digest",
    "adapter_revision",
    "vllm_version",
    "lmcache_version",
    "torch_version",
    "cuda_runtime",
    "fingerprint",
    "cache_key",
    "source_start",
    "source_end",
    "token_count",
    "token_ids",
    "record_checksum",
)

_SELECT_RECORD_COLUMNS = ", ".join(_RECORD_COLUMN_NAMES)
_INSERT_PLACEHOLDERS = ", ".join("?" for _ in _RECORD_COLUMN_NAMES)
_INSERT_RECORD_SQL = (
    f"INSERT OR IGNORE INTO sidecar_records ({_SELECT_RECORD_COLUMNS}) "
    f"VALUES ({_INSERT_PLACEHOLDERS})"
)

_NAMESPACE_TEXT_COLUMNS = (
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "model_config_digest",
    "kv_cache_config_digest",
    "adapter_revision",
    "vllm_version",
    "lmcache_version",
    "torch_version",
    "cuda_runtime",
)

_EXPECTED_RECORD_COLUMNS = (
    ("record_id", "INTEGER", 1),
    ("namespace_key", "BLOB", 0),
    ("namespace_schema_version", "INTEGER", 0),
    *((name, "TEXT", 0) for name in _NAMESPACE_TEXT_COLUMNS),
    ("fingerprint", "BLOB", 0),
    ("cache_key", "TEXT", 0),
    ("source_start", "INTEGER", 0),
    ("source_end", "INTEGER", 0),
    ("token_count", "INTEGER", 0),
    ("token_ids", "BLOB", 0),
    ("record_checksum", "BLOB", 0),
)


class SidecarMode(str, Enum):
    """The only two process roles allowed to open the sidecar."""

    SCHEDULER_READ_ONLY = "scheduler_read_only"
    WORKER_READ_WRITE = "worker_read_write"


class SidecarErrorCode(str, Enum):
    """Bounded failure codes suitable for logs and metrics."""

    INVALID_MODE = "invalid_mode"
    INVALID_PATH = "invalid_path"
    DATABASE_NOT_FOUND = "database_not_found"
    OPEN_FAILED = "open_failed"
    CLOSED = "closed"
    WRITE_FORBIDDEN = "write_forbidden"
    INVALID_NAMESPACE = "invalid_namespace"
    INVALID_FINGERPRINT = "invalid_fingerprint"
    INVALID_RECORD = "invalid_record"
    SCHEMA_MISMATCH = "schema_mismatch"
    DATABASE_CORRUPT = "database_corrupt"
    RECORD_CORRUPT = "record_corrupt"
    CHECKSUM_COLLISION = "checksum_collision"
    COLLISION_BUCKET_LIMIT = "collision_bucket_limit"
    SQLITE_OPERATION_FAILED = "sqlite_operation_failed"


class SidecarError(RuntimeError):
    """Base fail-closed sidecar error carrying only a bounded code."""

    def __init__(self, code: SidecarErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class SidecarConfigurationError(SidecarError):
    """An input or filesystem target violates a bounded sidecar contract."""


class SidecarModeError(SidecarError):
    """The process role attempted an operation it is not allowed to perform."""


class SidecarClosedError(SidecarError):
    """An operation was attempted after the connection was closed."""


class SidecarSchemaError(SidecarError):
    """The file is SQLite, but does not have the exact supported schema."""


class SidecarCorruptionError(SidecarError):
    """Database structure or persisted record content failed verification."""


class SidecarOperationError(SidecarError):
    """A bounded SQLite operation failed without evidence of corruption."""


def _raise_configuration(code: SidecarErrorCode) -> NoReturn:
    raise SidecarConfigurationError(code)


def _raise_schema() -> NoReturn:
    raise SidecarSchemaError(SidecarErrorCode.SCHEMA_MISMATCH)


def _raise_corrupt(
    code: SidecarErrorCode = SidecarErrorCode.RECORD_CORRUPT,
) -> NoReturn:
    raise SidecarCorruptionError(code)


def _translate_sqlite_error(error: sqlite3.DatabaseError) -> NoReturn:
    message = str(error).lower()
    corruption_markers = (
        "database disk image is malformed",
        "file is not a database",
        "malformed database schema",
        "database corruption",
    )
    if any(marker in message for marker in corruption_markers):
        raise SidecarCorruptionError(SidecarErrorCode.DATABASE_CORRUPT) from error
    raise SidecarOperationError(SidecarErrorCode.SQLITE_OPERATION_FAILED) from error


def _validated_path(path: str | os.PathLike[str], mode: SidecarMode) -> Path:
    if not isinstance(mode, SidecarMode):
        _raise_configuration(SidecarErrorCode.INVALID_MODE)
    try:
        raw_path = os.fspath(path)
    except TypeError as error:
        raise SidecarConfigurationError(SidecarErrorCode.INVALID_PATH) from error
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        _raise_configuration(SidecarErrorCode.INVALID_PATH)
    try:
        encoded_path = raw_path.encode("utf-8")
    except UnicodeError as error:
        raise SidecarConfigurationError(SidecarErrorCode.INVALID_PATH) from error
    if len(encoded_path) > MAX_SIDECAR_PATH_BYTES or raw_path == ":memory:":
        _raise_configuration(SidecarErrorCode.INVALID_PATH)

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        _raise_configuration(SidecarErrorCode.INVALID_PATH)
    try:
        resolved = candidate.resolve(strict=False)
        if len(os.fspath(resolved).encode("utf-8")) > MAX_SIDECAR_PATH_BYTES:
            _raise_configuration(SidecarErrorCode.INVALID_PATH)
        if mode is SidecarMode.SCHEDULER_READ_ONLY:
            if not resolved.exists():
                _raise_configuration(SidecarErrorCode.DATABASE_NOT_FOUND)
            if not resolved.is_file():
                _raise_configuration(SidecarErrorCode.INVALID_PATH)
        else:
            if not resolved.parent.is_dir():
                _raise_configuration(SidecarErrorCode.INVALID_PATH)
            if resolved.exists() and not resolved.is_file():
                _raise_configuration(SidecarErrorCode.INVALID_PATH)
    except OSError as error:
        raise SidecarConfigurationError(SidecarErrorCode.INVALID_PATH) from error
    return resolved


def _bounded_text(value: object, *, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise SidecarConfigurationError(SidecarErrorCode.INVALID_RECORD) from error
    if len(encoded) > maximum_bytes:
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)
    return value


def _validate_namespace(namespace: CacheNamespace) -> None:
    if not isinstance(namespace, CacheNamespace):
        _raise_configuration(SidecarErrorCode.INVALID_NAMESPACE)
    if (
        isinstance(namespace.schema_version, bool)
        or not isinstance(namespace.schema_version, int)
        or not 1 <= namespace.schema_version <= 2_147_483_647
    ):
        _raise_configuration(SidecarErrorCode.INVALID_NAMESPACE)
    expected_names = ("schema_version", *_NAMESPACE_TEXT_COLUMNS)
    canonical_fields = namespace.canonical_fields()
    if tuple(name for name, _ in canonical_fields) != expected_names:
        _raise_configuration(SidecarErrorCode.INVALID_NAMESPACE)
    for name, value in canonical_fields:
        if name != "schema_version":
            try:
                _bounded_text(value, maximum_bytes=MAX_NAMESPACE_FIELD_BYTES)
            except SidecarConfigurationError as error:
                raise SidecarConfigurationError(
                    SidecarErrorCode.INVALID_NAMESPACE
                ) from error


def _validate_fingerprint(fingerprint: SegmentFingerprint) -> None:
    if not isinstance(fingerprint, SegmentFingerprint):
        _raise_configuration(SidecarErrorCode.INVALID_FINGERPRINT)
    if not isinstance(fingerprint.digest, bytes) or len(fingerprint.digest) != 32:
        _raise_configuration(SidecarErrorCode.INVALID_FINGERPRINT)


def _validate_record(record: CacheRecord) -> None:
    if not isinstance(record, CacheRecord):
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)
    _validate_namespace(record.namespace)
    _validate_fingerprint(record.fingerprint)
    _bounded_text(record.cache_key, maximum_bytes=MAX_CACHE_KEY_BYTES)
    try:
        token_ids = normalize_token_ids(record.token_ids)
    except (TypeError, ValueError) as error:
        raise SidecarConfigurationError(SidecarErrorCode.INVALID_RECORD) from error
    if not 1 <= len(token_ids) <= MAX_TOKENS_PER_RECORD:
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)
    if not isinstance(record.source_range, TokenRange):
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)
    if (
        record.source_range.start < 0
        or record.source_range.end > MAX_TOKEN_POSITION
        or len(record.source_range) != len(token_ids)
    ):
        _raise_configuration(SidecarErrorCode.INVALID_RECORD)


class _Digest(Protocol):
    def update(self, value: bytes) -> None:
        """Add bytes to a streaming digest."""


def _update_field(digest: _Digest, name: str, value: bytes) -> None:
    name_bytes = name.encode("ascii")
    digest.update(struct.pack(">I", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _namespace_key(namespace: CacheNamespace) -> bytes:
    digest = hashlib.sha256()
    digest.update(_NAMESPACE_KEY_DOMAIN)
    for name, value in namespace.canonical_fields():
        _update_field(digest, name, value.encode("utf-8"))
    return digest.digest()


def _encode_token_ids(token_ids: tuple[int, ...]) -> bytes:
    output = bytearray(_TOKEN_BLOB_MAGIC)
    output.extend(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        output.extend(struct.pack(">Q", token_id))
    return bytes(output)


def _decode_token_ids(value: bytes, expected_count: int) -> tuple[int, ...]:
    if (
        len(value) < _TOKEN_BLOB_HEADER_BYTES
        or value[: len(_TOKEN_BLOB_MAGIC)] != _TOKEN_BLOB_MAGIC
    ):
        _raise_corrupt()
    (encoded_count,) = struct.unpack(
        ">Q", value[len(_TOKEN_BLOB_MAGIC) : _TOKEN_BLOB_HEADER_BYTES]
    )
    if (
        encoded_count != expected_count
        or not 1 <= encoded_count <= MAX_TOKENS_PER_RECORD
    ):
        _raise_corrupt()
    expected_bytes = _TOKEN_BLOB_HEADER_BYTES + encoded_count * 8
    if len(value) != expected_bytes:
        _raise_corrupt()
    token_payload = value[_TOKEN_BLOB_HEADER_BYTES:]
    return tuple(item[0] for item in struct.iter_unpack(">Q", token_payload))


def _record_checksum(
    namespace: CacheNamespace,
    fingerprint: bytes,
    cache_key: str,
    source_start: int,
    source_end: int,
    token_blob: bytes,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(_RECORD_CHECKSUM_DOMAIN)
    for name, value in namespace.canonical_fields():
        _update_field(digest, name, value.encode("utf-8"))
    _update_field(digest, "fingerprint", fingerprint)
    _update_field(digest, "cache_key", cache_key.encode("utf-8"))
    _update_field(digest, "source_start", struct.pack(">Q", source_start))
    _update_field(digest, "source_end", struct.pack(">Q", source_end))
    _update_field(digest, "token_ids", token_blob)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class _EncodedRecord:
    record: CacheRecord
    namespace_key: bytes
    token_blob: bytes
    checksum: bytes

    @classmethod
    def from_record(cls, record: CacheRecord) -> _EncodedRecord:
        _validate_record(record)
        token_blob = _encode_token_ids(record.token_ids)
        return cls(
            record=record,
            namespace_key=_namespace_key(record.namespace),
            token_blob=token_blob,
            checksum=_record_checksum(
                record.namespace,
                record.fingerprint.digest,
                record.cache_key,
                record.source_range.start,
                record.source_range.end,
                token_blob,
            ),
        )

    def database_values(self) -> tuple[object, ...]:
        namespace = self.record.namespace
        return (
            self.namespace_key,
            namespace.schema_version,
            namespace.model_id,
            namespace.model_revision,
            namespace.tokenizer_id,
            namespace.tokenizer_revision,
            namespace.model_config_digest,
            namespace.kv_cache_config_digest,
            namespace.adapter_revision,
            namespace.vllm_version,
            namespace.lmcache_version,
            namespace.torch_version,
            namespace.cuda_runtime,
            self.record.fingerprint.digest,
            self.record.cache_key,
            self.record.source_range.start,
            self.record.source_range.end,
            len(self.record.token_ids),
            self.token_blob,
            self.checksum,
        )


def _db_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_corrupt()
    return value


def _db_text(value: object) -> str:
    if not isinstance(value, str):
        _raise_corrupt()
    return value


def _db_blob(value: object) -> bytes:
    if not isinstance(value, bytes):
        _raise_corrupt()
    return value


def _namespace_from_row(row: sqlite3.Row) -> CacheNamespace:
    try:
        namespace = CacheNamespace(
            schema_version=_db_int(row["namespace_schema_version"]),
            model_id=_db_text(row["model_id"]),
            model_revision=_db_text(row["model_revision"]),
            tokenizer_id=_db_text(row["tokenizer_id"]),
            tokenizer_revision=_db_text(row["tokenizer_revision"]),
            model_config_digest=_db_text(row["model_config_digest"]),
            kv_cache_config_digest=_db_text(row["kv_cache_config_digest"]),
            adapter_revision=_db_text(row["adapter_revision"]),
            vllm_version=_db_text(row["vllm_version"]),
            lmcache_version=_db_text(row["lmcache_version"]),
            torch_version=_db_text(row["torch_version"]),
            cuda_runtime=_db_text(row["cuda_runtime"]),
        )
        _validate_namespace(namespace)
    except (TypeError, ValueError, SidecarConfigurationError) as error:
        raise SidecarCorruptionError(SidecarErrorCode.RECORD_CORRUPT) from error
    return namespace


def _record_from_row(
    row: sqlite3.Row,
    expected_namespace: CacheNamespace,
    expected_fingerprint: SegmentFingerprint,
) -> CacheRecord:
    namespace = _namespace_from_row(row)
    namespace_key = _db_blob(row["namespace_key"])
    if namespace_key != _namespace_key(namespace) or namespace != expected_namespace:
        _raise_corrupt()

    fingerprint_bytes = _db_blob(row["fingerprint"])
    if fingerprint_bytes != expected_fingerprint.digest:
        _raise_corrupt()
    token_count = _db_int(row["token_count"])
    token_ids = _decode_token_ids(_db_blob(row["token_ids"]), token_count)
    source_start = _db_int(row["source_start"])
    source_end = _db_int(row["source_end"])
    cache_key = _db_text(row["cache_key"])
    if (
        source_start < 0
        or source_end > MAX_TOKEN_POSITION
        or source_end - source_start != token_count
    ):
        _raise_corrupt()
    try:
        _bounded_text(cache_key, maximum_bytes=MAX_CACHE_KEY_BYTES)
    except SidecarConfigurationError as error:
        raise SidecarCorruptionError(SidecarErrorCode.RECORD_CORRUPT) from error
    checksum = _db_blob(row["record_checksum"])
    expected_checksum = _record_checksum(
        namespace,
        fingerprint_bytes,
        cache_key,
        source_start,
        source_end,
        _db_blob(row["token_ids"]),
    )
    if checksum != expected_checksum:
        _raise_corrupt()

    try:
        record = CacheRecord(
            namespace=namespace,
            fingerprint=SegmentFingerprint(fingerprint_bytes),
            token_ids=token_ids,
            source_range=TokenRange(source_start, source_end),
            cache_key=cache_key,
        )
        _validate_record(record)
    except (TypeError, ValueError, SidecarConfigurationError) as error:
        raise SidecarCorruptionError(SidecarErrorCode.RECORD_CORRUPT) from error
    return record


class SqliteSidecarIndex:
    """Persistent ``RecordLookup`` with explicit scheduler/worker modes."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        mode: SidecarMode,
    ) -> None:
        self._mode = mode
        self._path = _validated_path(path, mode)
        self._lock = RLock()
        self._closed = False
        self._connection = self._open_connection()
        try:
            self._prepare_database()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @property
    def mode(self) -> SidecarMode:
        return self._mode

    @property
    def path(self) -> Path:
        return self._path

    @property
    def closed(self) -> bool:
        return self._closed

    def _open_connection(self) -> sqlite3.Connection:
        try:
            if self._mode is SidecarMode.SCHEDULER_READ_ONLY:
                connection = sqlite3.connect(
                    f"{self._path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=_BUSY_TIMEOUT_MILLISECONDS / 1_000,
                    isolation_level=None,
                    check_same_thread=False,
                )
            else:
                connection = sqlite3.connect(
                    self._path,
                    timeout=_BUSY_TIMEOUT_MILLISECONDS / 1_000,
                    isolation_level=None,
                    check_same_thread=False,
                )
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.DatabaseError as error:
            if self._mode is SidecarMode.SCHEDULER_READ_ONLY:
                raise SidecarOperationError(SidecarErrorCode.OPEN_FAILED) from error
            _translate_sqlite_error(error)

    def _prepare_database(self) -> None:
        try:
            self._connection.execute(
                f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MILLISECONDS}"
            )
            self._connection.execute("PRAGMA trusted_schema = OFF")
            if self._mode is SidecarMode.SCHEDULER_READ_ONLY:
                self._connection.execute("PRAGMA query_only = ON")
                self._validate_schema()
                return

            self._connection.execute("PRAGMA foreign_keys = ON")
            if self._database_is_empty():
                journal_mode = self._pragma_text("PRAGMA journal_mode = WAL")
                if journal_mode.lower() != "wal":
                    raise SidecarOperationError(
                        SidecarErrorCode.SQLITE_OPERATION_FAILED
                    )
                self._connection.execute("PRAGMA synchronous = FULL")
                self._initialize_schema()
            self._validate_schema()
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA wal_autocheckpoint = 1000")
        except sqlite3.DatabaseError as error:
            _translate_sqlite_error(error)

    def _database_is_empty(self) -> bool:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%'"
        ).fetchone()
        if row is None:
            _raise_corrupt(SidecarErrorCode.DATABASE_CORRUPT)
        return _db_int(row[0]) == 0

    def _initialize_schema(self) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            # Another worker may have completed initialization while this
            # connection waited for its transaction lock.
            if not self._database_is_empty():
                self._connection.execute("COMMIT")
                return
            self._connection.execute(_CREATE_METADATA_SQL)
            self._connection.execute(_CREATE_RECORDS_SQL)
            self._connection.execute(_CREATE_LOOKUP_INDEX_SQL)
            self._connection.executemany(
                "INSERT INTO sidecar_metadata (key, value) VALUES (?, ?)",
                tuple(sorted(_METADATA_VALUES.items())),
            )
            self._connection.execute(
                f"PRAGMA application_id = {SIDECAR_APPLICATION_ID}"
            )
            self._connection.execute(
                f"PRAGMA user_version = {SIDECAR_SCHEMA_VERSION}"
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _pragma_int(self, statement: str) -> int:
        row = self._connection.execute(statement).fetchone()
        if row is None:
            _raise_corrupt(SidecarErrorCode.DATABASE_CORRUPT)
        return _db_int(row[0])

    def _pragma_text(self, statement: str) -> str:
        row = self._connection.execute(statement).fetchone()
        if row is None:
            _raise_corrupt(SidecarErrorCode.DATABASE_CORRUPT)
        return _db_text(row[0])

    def _validate_schema(self) -> None:
        if self._database_is_empty():
            _raise_schema()
        if self._pragma_int("PRAGMA application_id") != SIDECAR_APPLICATION_ID:
            _raise_schema()
        if self._pragma_int("PRAGMA user_version") != SIDECAR_SCHEMA_VERSION:
            _raise_schema()
        if self._pragma_text("PRAGMA journal_mode").lower() != "wal":
            _raise_schema()

        objects = {
            (_db_text(row[0]), _db_text(row[1]))
            for row in self._connection.execute(
                "SELECT type, name FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if objects != {
            ("table", "sidecar_metadata"),
            ("table", "sidecar_records"),
            ("index", "sidecar_lookup_idx"),
        }:
            _raise_schema()
        for object_name, expected_sql in _EXPECTED_SCHEMA_SQL.items():
            row = self._connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name = ?",
                (object_name,),
            ).fetchone()
            if (
                row is None
                or not isinstance(row[0], str)
                or row[0].strip() != expected_sql
            ):
                _raise_schema()

        metadata_columns = tuple(
            (_db_text(row[1]), _db_text(row[2]), _db_int(row[5]))
            for row in self._connection.execute(
                "PRAGMA table_info(sidecar_metadata)"
            ).fetchall()
        )
        if metadata_columns != (("key", "TEXT", 1), ("value", "TEXT", 0)):
            _raise_schema()

        record_columns = tuple(
            (_db_text(row[1]), _db_text(row[2]), _db_int(row[5]))
            for row in self._connection.execute(
                "PRAGMA table_info(sidecar_records)"
            ).fetchall()
        )
        if record_columns != _EXPECTED_RECORD_COLUMNS:
            _raise_schema()

        lookup_columns = tuple(
            _db_text(row[2])
            for row in self._connection.execute(
                "PRAGMA index_info(sidecar_lookup_idx)"
            ).fetchall()
        )
        if lookup_columns != (
            "namespace_key",
            "fingerprint",
            "cache_key",
            "source_start",
            "source_end",
        ):
            _raise_schema()

        unique_indexes = {
            _db_text(row[1]): _db_int(row[2])
            for row in self._connection.execute(
                "PRAGMA index_list(sidecar_records)"
            ).fetchall()
        }
        checksum_indexes = tuple(
            index_name
            for index_name, is_unique in unique_indexes.items()
            if is_unique == 1
            and tuple(
                _db_text(row[2])
                for row in self._connection.execute(
                    f"PRAGMA index_info('{index_name.replace(chr(39), chr(39) * 2)}')"
                ).fetchall()
            )
            == ("record_checksum",)
        )
        if len(checksum_indexes) != 1:
            _raise_schema()

        metadata_rows = self._connection.execute(
            "SELECT key, value FROM sidecar_metadata ORDER BY key"
        ).fetchall()
        metadata = {
            _db_text(row[0]): _db_text(row[1]) for row in metadata_rows
        }
        if metadata != _METADATA_VALUES:
            _raise_schema()

        quick_check = tuple(
            _db_text(row[0])
            for row in self._connection.execute("PRAGMA quick_check").fetchall()
        )
        if quick_check != ("ok",):
            _raise_corrupt(SidecarErrorCode.DATABASE_CORRUPT)

    def _require_open(self) -> None:
        if self._closed:
            raise SidecarClosedError(SidecarErrorCode.CLOSED)

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def add(self, record: CacheRecord) -> bool:
        """Atomically add one record; return ``False`` for an exact duplicate."""

        return self.add_many((record,)) == 1

    def add_many(self, records: Iterable[CacheRecord]) -> int:
        """Atomically add multiple records after prevalidating the whole batch."""

        with self._lock:
            self._require_open()
            if self._mode is not SidecarMode.WORKER_READ_WRITE:
                raise SidecarModeError(SidecarErrorCode.WRITE_FORBIDDEN)
            encoded_records: list[_EncodedRecord] = []
            total_tokens = 0
            for record in records:
                if len(encoded_records) >= MAX_RECORDS_PER_TRANSACTION:
                    _raise_configuration(SidecarErrorCode.INVALID_RECORD)
                encoded = _EncodedRecord.from_record(record)
                total_tokens += len(encoded.record.token_ids)
                if total_tokens > MAX_TOKENS_PER_TRANSACTION:
                    _raise_configuration(SidecarErrorCode.INVALID_RECORD)
                encoded_records.append(encoded)
            inserted = 0
            try:
                with self._transaction(write=True):
                    for encoded in encoded_records:
                        cursor = self._connection.execute(
                            _INSERT_RECORD_SQL, encoded.database_values()
                        )
                        if cursor.rowcount == 1:
                            inserted += 1
                            continue
                        rows = self._connection.execute(
                            f"SELECT {_SELECT_RECORD_COLUMNS} "
                            "FROM sidecar_records WHERE record_checksum = ? LIMIT 2",
                            (encoded.checksum,),
                        ).fetchall()
                        if len(rows) != 1:
                            _raise_corrupt()
                        existing = _record_from_row(
                            rows[0],
                            encoded.record.namespace,
                            encoded.record.fingerprint,
                        )
                        if existing != encoded.record:
                            _raise_corrupt(SidecarErrorCode.CHECKSUM_COLLISION)
            except sqlite3.DatabaseError as error:
                _translate_sqlite_error(error)
            return inserted

    def lookup(
        self,
        namespace: CacheNamespace,
        fingerprint: SegmentFingerprint,
    ) -> Sequence[CacheRecord]:
        """Return an exact persisted collision bucket in deterministic order."""

        with self._lock:
            self._require_open()
            _validate_namespace(namespace)
            _validate_fingerprint(fingerprint)
            namespace_key = _namespace_key(namespace)
            try:
                with self._transaction(write=False):
                    rows = self._connection.execute(
                        f"SELECT {_SELECT_RECORD_COLUMNS} FROM sidecar_records "
                        "WHERE namespace_key = ? AND fingerprint = ? "
                        "ORDER BY cache_key, source_start, source_end, record_id "
                        "LIMIT ?",
                        (
                            namespace_key,
                            fingerprint.digest,
                            MAX_COLLISION_BUCKET_RECORDS + 1,
                        ),
                    ).fetchall()
                    if len(rows) > MAX_COLLISION_BUCKET_RECORDS:
                        raise SidecarCorruptionError(
                            SidecarErrorCode.COLLISION_BUCKET_LIMIT
                        )
                    records = tuple(
                        _record_from_row(row, namespace, fingerprint) for row in rows
                    )
            except sqlite3.DatabaseError as error:
                _translate_sqlite_error(error)
            return records

    def close(self) -> None:
        """Close the local SQLite handle; repeated calls are harmless."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.close()
            except sqlite3.DatabaseError as error:
                self._closed = True
                _translate_sqlite_error(error)
            self._closed = True

    def __enter__(self) -> SqliteSidecarIndex:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def open_sidecar_index(
    path: str | os.PathLike[str], mode: SidecarMode
) -> SqliteSidecarIndex:
    """Open the exact schema with an explicit scheduler or worker role."""

    return SqliteSidecarIndex(path, mode)
