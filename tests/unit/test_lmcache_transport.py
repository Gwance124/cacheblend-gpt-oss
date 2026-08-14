"""CPU-only contract tests for the pinned LMCache Blend V2 adapter."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from cacheblend_gpt_oss.planner import (
    SHA256_FINGERPRINTER,
    CacheNamespace,
    InMemoryRecordIndex,
    MatchPlanner,
    TokenRange,
    TokenSegment,
    build_cache_record,
)
from cacheblend_gpt_oss.storage import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_CHUNK_SIZE,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    LMCACHE_VERSION,
    LmcacheBlendTransport,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheCloseError,
    LmcacheConfigurationError,
    LmcacheDependencyError,
    LmcacheLifecycleError,
    LmcacheOperationError,
    LmcacheProtocolError,
    LmcacheRequest,
    LmcacheRetrieveReceipt,
    LmcacheServerAttestation,
    LmcacheStagingLayout,
    LmcacheStagingRegistration,
    LmcacheTransportState,
    VerifiedLmcacheCandidate,
)
from cacheblend_gpt_oss.storage import lmcache_v0_4_3 as runtime_module


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


def attestation() -> LmcacheServerAttestation:
    return LmcacheServerAttestation(
        lmcache_version=LMCACHE_VERSION,
        source_commit=LMCACHE_SOURCE_COMMIT,
        protocol=LMCACHE_BLEND_PROTOCOL,
        hash_algorithm=LMCACHE_HASH_ALGORITHM,
    )


def config(**changes: object) -> LmcacheBlendTransportConfig:
    values: dict[str, object] = {
        "namespace": namespace(),
        "server_attestation": attestation(),
    }
    values.update(changes)
    return LmcacheBlendTransportConfig(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FakeKey:
    model_name: str
    world_size: int
    worker_id: int | None
    token_ids: tuple[int, ...]
    start: int
    end: int
    request_id: str


@dataclass(frozen=True, slots=True)
class FakeMatch:
    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    hash: bytes


@dataclass(frozen=True, slots=True)
class FakeWrapper:
    shape: tuple[int, int, int, int]
    dtype: str


@dataclass(slots=True)
class FakeFuture:
    response: object = None
    error: BaseException | None = None

    def result(self, timeout: float | None = None) -> object:
        assert timeout is not None and timeout > 0
        if self.error is not None:
            raise self.error
        return self.response


@dataclass(frozen=True, slots=True)
class SubmittedCall:
    request: str
    payloads: tuple[object, ...]
    response_cls: object | None


class FakeMessageQueue:
    def __init__(self) -> None:
        self.responses: defaultdict[str, deque[FakeFuture]] = defaultdict(deque)
        self.calls: list[SubmittedCall] = []
        self.closed = False
        self.close_error: BaseException | None = None

    def enqueue(
        self,
        request: LmcacheRequest,
        response: object = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.responses[request.value].append(FakeFuture(response, error))

    def submit_request(
        self,
        request_type: object,
        request_payloads: list[object],
        response_cls: object | None = None,
    ) -> FakeFuture:
        request = str(request_type)
        self.calls.append(SubmittedCall(request, tuple(request_payloads), response_cls))
        if not self.responses[request]:
            raise AssertionError(f"no fake response queued for {request}")
        return self.responses[request].popleft()

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeBindings:
    lmcache_version = LMCACHE_VERSION

    def __init__(
        self,
        *,
        schema_error: bool = False,
        chunk_hashes: object | None = None,
        hash_error: BaseException | None = None,
    ) -> None:
        self.schema_error = schema_error
        self.chunk_hashes = chunk_hashes
        self.hash_error = hash_error
        self.schema_validations = 0
        self.ordinary_waits = 0
        self.cuda_waits = 0
        self.hash_calls: list[tuple[int, ...]] = []

    def validate_protocol_schema(self) -> None:
        self.schema_validations += 1
        if self.schema_error:
            raise LmcacheProtocolError("fake schema mismatch")

    def compute_chunk_hashes(
        self, token_ids: tuple[int, ...]
    ) -> tuple[bytes, ...]:
        self.hash_calls.append(token_ids)
        if self.hash_error is not None:
            raise self.hash_error
        if self.chunk_hashes is not None:
            return cast(tuple[bytes, ...], self.chunk_hashes)
        return tuple(
            bytes(((chunk_index + 1) % 256,)) * 32
            for chunk_index in range(len(token_ids) // LMCACHE_CHUNK_SIZE)
        )

    def request_type(self, request: LmcacheRequest) -> object:
        return request.value

    def response_class(self, request: LmcacheRequest) -> object | None:
        return f"response:{request.value}"

    def make_key(
        self,
        *,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: tuple[int, ...],
        start: int,
        end: int,
        request_id: str,
    ) -> object:
        return FakeKey(
            model_name,
            world_size,
            worker_id,
            token_ids,
            start,
            end,
            request_id,
        )

    def parse_match(self, raw_match: object) -> tuple[int, int, int, int, bytes]:
        if not isinstance(raw_match, FakeMatch):
            raise LmcacheProtocolError("not a FakeMatch")
        return (
            raw_match.old_st,
            raw_match.old_ed,
            raw_match.cur_st,
            raw_match.cur_ed,
            raw_match.hash,
        )

    def make_match(
        self, *, old_st: int, old_ed: int, cur_st: int, cur_ed: int, hash: bytes
    ) -> object:
        return FakeMatch(old_st, old_ed, cur_st, cur_ed, hash)

    def validate_kv_cache_payload(
        self,
        payload: list[object],
        *,
        expected_shape: tuple[int, int, int, int],
        expected_dtype_name: str,
    ) -> None:
        if len(payload) != 1 or not isinstance(payload[0], FakeWrapper):
            raise LmcacheConfigurationError("invalid fake wrapper")
        if payload[0].shape != expected_shape:
            raise LmcacheConfigurationError("invalid fake shape")
        if payload[0].dtype != expected_dtype_name:
            raise LmcacheConfigurationError("invalid fake dtype")

    def wait(self, future: runtime_module.MessageFuture, timeout: float) -> object:
        self.ordinary_waits += 1
        return future.result(timeout)

    def wait_cuda(
        self, future: runtime_module.MessageFuture, timeout: float
    ) -> object:
        self.cuda_waits += 1
        response = future.result(timeout)
        if (
            not isinstance(response, tuple)
            or len(response) != 2
            or not isinstance(response[0], bytes)
            or not response[0]
            or not isinstance(response[1], bool)
        ):
            raise LmcacheProtocolError("invalid fake CUDA response")
        return response[1]


def registration() -> LmcacheStagingRegistration:
    layout = LmcacheStagingLayout(
        layer_count=24,
        token_capacity=1024,
        kv_width=512,
        dtype_name="torch.bfloat16",
    )
    return LmcacheStagingRegistration(
        instance_id=71,
        kv_cache_payload=(FakeWrapper(layout.shape, layout.dtype_name),),
        layout=layout,
    )


def opened_transport(
    *,
    queue: FakeMessageQueue | None = None,
    bindings: FakeBindings | None = None,
) -> tuple[LmcacheBlendTransport, FakeMessageQueue, FakeBindings]:
    queue = queue or FakeMessageQueue()
    bindings = bindings or FakeBindings()
    queue.enqueue(LmcacheRequest.PING, True)
    queue.enqueue(LmcacheRequest.GET_CHUNK_SIZE, LMCACHE_CHUNK_SIZE)
    transport = LmcacheBlendTransport(config(), bindings, queue)
    transport.open()
    return transport, queue, bindings


def registered_transport(
    *, bindings: FakeBindings | None = None
) -> tuple[
    LmcacheBlendTransport, FakeMessageQueue, FakeBindings
]:
    transport, queue, bindings = opened_transport(bindings=bindings)
    queue.enqueue(LmcacheRequest.REGISTER, None)
    transport.register_staging_buffer(registration())
    return transport, queue, bindings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "other/model"),
        ("tokenizer_id", "other/tokenizer"),
        ("vllm_version", "0.20.0"),
        ("lmcache_version", "0.4.4"),
        ("torch_version", "2.11.0+cu128"),
        ("cuda_runtime", "12.9"),
    ],
)
def test_config_rejects_incompatible_cache_namespace(field: str, value: str) -> None:
    incompatible = replace(namespace(), **{field: value})

    with pytest.raises(LmcacheConfigurationError):
        config(namespace=incompatible)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lmcache_version", "0.4.4"),
        ("source_commit", "main"),
        ("protocol", "legacy-blend"),
        ("hash_algorithm", "builtin"),
    ],
)
def test_config_requires_exact_server_attestation(field: str, value: str) -> None:
    bad_attestation = replace(attestation(), **{field: value})

    with pytest.raises(LmcacheConfigurationError):
        config(server_attestation=bad_attestation)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"world_size": 2}, "world_size=1"),
        ({"worker_id": 1}, "worker_id=0"),
        ({"chunk_size": 128}, "chunk size"),
        ({"request_timeout_seconds": 0.0}, "finite positive"),
    ],
)
def test_config_rejects_unaudited_protocol_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(LmcacheConfigurationError, match=message):
        config(**changes)


def test_lmcache_value_objects_reject_untyped_descriptors() -> None:
    with pytest.raises(LmcacheConfigurationError):
        LmcacheServerAttestation(
            lmcache_version=True,  # type: ignore[arg-type]
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        )
    with pytest.raises(LmcacheConfigurationError):
        LmcacheBlendTransportConfig(
            namespace=object(),  # type: ignore[arg-type]
            server_attestation=attestation(),
        )
    with pytest.raises(LmcacheConfigurationError):
        LmcacheStagingLayout(
            layer_count=24,
            token_capacity=1024,
            kv_width=512,
            dtype_name=True,  # type: ignore[arg-type]
        )
    layout = LmcacheStagingLayout(
        layer_count=24,
        token_capacity=1024,
        kv_width=512,
        dtype_name="torch.bfloat16",
    )
    with pytest.raises(LmcacheConfigurationError):
        LmcacheStagingRegistration(
            instance_id=1,
            kv_cache_payload=[],  # type: ignore[arg-type]
            layout=layout,
        )
    with pytest.raises(LmcacheProtocolError):
        LmcacheCandidate(
            source_relative_range=object(),  # type: ignore[arg-type]
            target_range=TokenRange(0, LMCACHE_CHUNK_SIZE),
            storage_hash=b"h" * 32,
            storage_model_name="model",
            query_digest=b"q" * 32,
        )
    with pytest.raises(LmcacheProtocolError):
        LmcacheRetrieveReceipt(True, 1)  # type: ignore[arg-type]

    with pytest.raises(LmcacheConfigurationError):
        runtime_module.validate_buffer_range(
            start=0,
            length=1,
            capacity=True,  # type: ignore[arg-type]
            field_name="buffer",
        )


def test_storage_model_name_is_stable_and_covers_namespace() -> None:
    first = config()
    same = config()
    changed = config(namespace=replace(namespace(), model_revision="other-revision"))

    assert first.storage_model_name == same.storage_model_name
    assert first.storage_model_name != changed.storage_model_name
    assert first.storage_model_name.startswith("openai/gpt-oss-20b#cacheblend-v1-")


def test_constructor_uses_injected_bindings_without_runtime_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(name: str) -> NoReturn:
        raise AssertionError(f"unexpected runtime import: {name}")

    monkeypatch.setattr(runtime_module, "import_module", unexpected_import)
    bindings = FakeBindings()
    transport = LmcacheBlendTransport(config(), bindings, FakeMessageQueue())

    assert transport.state is LmcacheTransportState.CREATED
    assert bindings.schema_validations == 1


def test_local_protocol_schema_mismatch_fails_construction() -> None:
    with pytest.raises(LmcacheProtocolError, match="fake schema mismatch"):
        LmcacheBlendTransport(
            config(), FakeBindings(schema_error=True), FakeMessageQueue()
        )


def test_lazy_loader_reports_missing_lmcache_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> NoReturn:
        assert name == "lmcache"
        raise runtime_module.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_module, "version", missing_distribution)

    with pytest.raises(LmcacheDependencyError, match="not installed"):
        runtime_module.load_lmcache_v0_4_3_bindings()


def test_lazy_loader_constructs_the_exact_pinned_token_hasher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[int, str]] = []
    hashers: list[FakeTokenHasher] = []

    class FakeTokenHasher:
        def __init__(self, *, chunk_size: int, hash_algorithm: str) -> None:
            constructed.append((chunk_size, hash_algorithm))
            hashers.append(self)
            self.chunk_size = chunk_size
            self.hash_algorithm_name = hash_algorithm
            self.none_hash = b"random-process-seed"

        @staticmethod
        def hash_func(value: object) -> bytes:
            assert value == (0, (0,), None)
            return b"d" * 32

        def compute_chunk_hashes(self, token_ids: list[int]) -> list[bytes]:
            assert token_ids == list(range(LMCACHE_CHUNK_SIZE))
            return [b"p" * 32]

    modules = {
        "lmcache.v1.multiprocess.custom_types": SimpleNamespace(
            IPCCacheEngineKey=FakeKey,
            CBMatchResult=FakeMatch,
            CudaIPCWrapper=FakeWrapper,
        ),
        "lmcache.v1.multiprocess.protocol": SimpleNamespace(
            RequestType=object,
            get_payload_classes=lambda _request: [],
            get_response_class=lambda _request: None,
        ),
        "lmcache.v1.multiprocess.token_hasher": SimpleNamespace(
            TokenHasher=FakeTokenHasher
        ),
    }
    monkeypatch.setattr(runtime_module, "version", lambda _name: LMCACHE_VERSION)
    monkeypatch.setattr(runtime_module, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        runtime_module._RuntimeLmcacheBindings,
        "validate_protocol_schema",
        lambda _self: None,
    )

    bindings = runtime_module.load_lmcache_v0_4_3_bindings()

    assert constructed == [(LMCACHE_CHUNK_SIZE, LMCACHE_HASH_ALGORITHM)]
    assert hashers[0].none_hash == b"d" * 32
    assert tuple(bindings.compute_chunk_hashes(tuple(range(LMCACHE_CHUNK_SIZE)))) == (
        b"p" * 32,
    )
    hashers[0].hash_algorithm_name = "drifted"
    with pytest.raises(LmcacheProtocolError, match="drifted"):
        bindings.compute_chunk_hashes(tuple(range(LMCACHE_CHUNK_SIZE)))


def test_open_probes_ping_and_exact_chunk_size() -> None:
    transport, queue, bindings = opened_transport()

    assert transport.state is LmcacheTransportState.READY
    assert [call.request for call in queue.calls] == [
        LmcacheRequest.PING.value,
        LmcacheRequest.GET_CHUNK_SIZE.value,
    ]
    assert bindings.ordinary_waits == 2


def test_open_chunk_mismatch_fails_closed() -> None:
    queue = FakeMessageQueue()
    queue.enqueue(LmcacheRequest.PING, True)
    queue.enqueue(LmcacheRequest.GET_CHUNK_SIZE, 128)
    transport = LmcacheBlendTransport(config(), FakeBindings(), queue)

    with pytest.raises(LmcacheOperationError, match="chunk size"):
        transport.open()

    assert transport.state is LmcacheTransportState.FAILED
    with pytest.raises(LmcacheLifecycleError):
        transport.lookup_candidates(range(512), request_id="after-failure")


def test_register_uses_strong_model_namespace_and_exact_layout() -> None:
    transport, queue, _ = opened_transport()
    queue.enqueue(LmcacheRequest.REGISTER, None)

    transport.register_staging_buffer(registration())

    assert transport.state is LmcacheTransportState.REGISTERED
    call = queue.calls[-1]
    assert call.request == LmcacheRequest.REGISTER.value
    instance_id, wrappers, model_name, world_size = call.payloads
    assert instance_id == 71
    assert wrappers == list(registration().kv_cache_payload)
    assert model_name == transport.config.storage_model_name
    assert world_size == 1


def test_lookup_returns_untrusted_candidates_and_rankless_key() -> None:
    transport, queue, _ = opened_transport()
    document = tuple(range(LMCACHE_CHUNK_SIZE))
    query = (91, 92, 93, *document)
    storage_hash = b"h" * 32
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        [FakeMatch(0, LMCACHE_CHUNK_SIZE, 3, 3 + LMCACHE_CHUNK_SIZE, storage_hash)],
    )

    candidates = transport.lookup_candidates(query, request_id="lookup")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert isinstance(candidate, LmcacheCandidate)
    assert not isinstance(candidate, VerifiedLmcacheCandidate)
    assert candidate.target_range.start == 3
    assert candidate.cache_key.endswith(storage_hash.hex())
    key = queue.calls[-1].payloads[0]
    assert isinstance(key, FakeKey)
    assert key.model_name == transport.config.storage_model_name
    assert key.world_size == 1
    assert key.worker_id is None
    assert key.start == 0 and key.end == len(query)


@pytest.mark.parametrize(
    "match",
    [
        FakeMatch(0, 128, 0, 128, b"h" * 32),
        FakeMatch(0, 256, -1, 255, b"h" * 32),
        FakeMatch(0, 256, 0, 128, b"h" * 32),
        FakeMatch(0, 256, 300, 556, b"h" * 32),
        FakeMatch(0, 256, 0, 256, b"short"),
    ],
)
def test_lookup_rejects_malformed_candidate_response(match: FakeMatch) -> None:
    transport, queue, _ = opened_transport()
    queue.enqueue(LmcacheRequest.LOOKUP, [match])

    with pytest.raises(LmcacheProtocolError, match="invalid candidate"):
        transport.lookup_candidates(range(512), request_id="malformed")

    assert transport.state is LmcacheTransportState.FAILED


def test_lookup_transport_failure_is_not_silently_converted_to_miss() -> None:
    transport, queue, _ = opened_transport()
    queue.enqueue(LmcacheRequest.LOOKUP, error=TimeoutError("server timeout"))

    with pytest.raises(LmcacheOperationError, match="request failed"):
        transport.lookup_candidates(range(512), request_id="timeout")

    assert transport.state is LmcacheTransportState.FAILED


def test_already_wrapped_operation_failure_still_poison_transport() -> None:
    transport, queue, _ = opened_transport()
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        error=LmcacheOperationError("already wrapped operation failure"),
    )

    with pytest.raises(LmcacheOperationError, match="already wrapped"):
        transport.lookup_candidates(range(512), request_id="wrapped-error")

    assert transport.state is LmcacheTransportState.FAILED
    with pytest.raises(LmcacheLifecycleError):
        transport.lookup_candidates(range(512), request_id="after-wrapped")


def verified_candidate(
    transport: LmcacheBlendTransport,
    queue: FakeMessageQueue,
) -> tuple[tuple[int, ...], VerifiedLmcacheCandidate]:
    document = tuple(range(LMCACHE_CHUNK_SIZE))
    query = (9001, 9002, 9003, *document)
    storage_hash = b"v" * 32
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        [FakeMatch(0, LMCACHE_CHUNK_SIZE, 3, 3 + LMCACHE_CHUNK_SIZE, storage_hash)],
    )
    candidate = transport.lookup_candidates(query, request_id="candidate")[0]
    record = build_cache_record(
        transport.config.namespace,
        TokenSegment.at(100, document),
        candidate.cache_key,
    )
    plan = MatchPlanner(InMemoryRecordIndex([record])).plan(
        transport.config.namespace,
        [TokenSegment.at(3, document)],
    )
    return query, VerifiedLmcacheCandidate.bind(
        candidate,
        plan.matches[0],
        expected_namespace=transport.config.namespace,
    )


def test_candidate_binding_requires_storage_hash_bound_sidecar_record() -> None:
    transport, queue, _ = opened_transport()
    document = tuple(range(LMCACHE_CHUNK_SIZE))
    query = document
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        [FakeMatch(0, LMCACHE_CHUNK_SIZE, 0, LMCACHE_CHUNK_SIZE, b"a" * 32)],
    )
    candidate = transport.lookup_candidates(query, request_id="lookup")[0]
    wrong_record = build_cache_record(
        transport.config.namespace,
        TokenSegment.at(20, document),
        "different-storage-key",
    )
    plan = MatchPlanner(InMemoryRecordIndex([wrong_record])).plan(
        transport.config.namespace,
        [TokenSegment.at(0, document)],
    )

    with pytest.raises(LmcacheProtocolError, match="storage hash"):
        VerifiedLmcacheCandidate.bind(
            candidate,
            plan.matches[0],
            expected_namespace=transport.config.namespace,
        )


def test_retrieve_requires_verified_candidate_and_builds_exact_v2_request() -> None:
    transport, queue, bindings = registered_transport()
    query, verified = verified_candidate(transport, queue)
    queue.enqueue(LmcacheRequest.RETRIEVE, (b"server-event", True))

    receipt = transport.retrieve_precomputed(
        query,
        [verified],
        buffer_offset=10,
        event_ipc_handle=b"client-event",
        request_id="retrieve",
    )

    assert receipt.retrieved_tokens == LMCACHE_CHUNK_SIZE
    assert receipt.retrieved_chunks == 1
    assert bindings.cuda_waits == 1
    call = queue.calls[-1]
    assert call.request == LmcacheRequest.RETRIEVE.value
    key, matches, offset, instance_id, event = call.payloads
    assert isinstance(key, FakeKey) and key.worker_id == 0
    assert matches == [
        FakeMatch(0, LMCACHE_CHUNK_SIZE, 3, 3 + LMCACHE_CHUNK_SIZE, b"v" * 32)
    ]
    assert offset == 10
    assert instance_id == 71
    assert event == b"client-event"


def test_retrieve_rejects_raw_candidate_before_rpc() -> None:
    transport, queue, _ = registered_transport()
    document = tuple(range(LMCACHE_CHUNK_SIZE))
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        [FakeMatch(0, 256, 0, 256, b"x" * 32)],
    )
    candidate = transport.lookup_candidates(document, request_id="raw")[0]
    calls_before = len(queue.calls)

    with pytest.raises(LmcacheConfigurationError, match="independently verified"):
        transport.retrieve_precomputed(
            document,
            [candidate],  # type: ignore[list-item]
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="raw-retrieve",
        )

    assert len(queue.calls) == calls_before


def test_retrieve_rejects_candidate_replayed_against_different_query() -> None:
    transport, queue, _ = registered_transport()
    query, verified = verified_candidate(transport, queue)

    with pytest.raises(LmcacheProtocolError, match="different lookup query"):
        transport.retrieve_precomputed(
            (*query[:-1], 999_999),
            [verified],
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="replay",
        )


def test_precomputed_store_requires_complete_chunks_and_waits_for_cuda() -> None:
    transport, queue, bindings = registered_transport()
    document = tuple(range(LMCACHE_CHUNK_SIZE))
    queue.enqueue(LmcacheRequest.STORE_PRECOMPUTED, (b"server-event", True))

    receipt = transport.store_precomputed(
        document,
        cache_namespace=transport.config.namespace,
        document_source_range=TokenRange(1000, 1000 + LMCACHE_CHUNK_SIZE),
        buffer_offset=16,
        event_ipc_handle=b"client-event",
        request_id="store",
    )

    assert receipt.stored_tokens == LMCACHE_CHUNK_SIZE
    assert receipt.stored_chunks == 1
    assert receipt.candidate_lookup_required
    assert len(receipt.sidecar_records) == 1
    record = receipt.sidecar_records[0]
    assert record.namespace == transport.config.namespace
    assert record.token_ids == document
    assert record.source_range == TokenRange(1000, 1000 + LMCACHE_CHUNK_SIZE)
    assert record.cache_key == "lmcache:0.4.3:blake3:" + (b"\x01" * 32).hex()
    assert record.fingerprint == SHA256_FINGERPRINTER.fingerprint(
        transport.config.namespace, document
    )
    assert bindings.hash_calls == [document]
    assert bindings.cuda_waits == 1
    key, offset, instance_id, event = queue.calls[-1].payloads
    assert isinstance(key, FakeKey) and key.worker_id == 0
    assert key.start == 0 and key.end == LMCACHE_CHUNK_SIZE
    assert offset == 16 and instance_id == 71 and event == b"client-event"


def test_precomputed_store_derives_each_exact_chunk_and_keeps_positions_separate(
) -> None:
    first_hash = b"a" * 32
    second_hash = b"b" * 32
    bindings = FakeBindings(chunk_hashes=(first_hash, second_hash))
    transport, queue, _ = registered_transport(bindings=bindings)
    document = tuple(range(2 * LMCACHE_CHUNK_SIZE))
    absolute_source = TokenRange(1000, 1000 + len(document))
    queue.enqueue(LmcacheRequest.STORE_PRECOMPUTED, (b"server-event", True))

    receipt = transport.store_precomputed(
        document,
        cache_namespace=transport.config.namespace,
        document_source_range=absolute_source,
        buffer_offset=0,
        event_ipc_handle=b"client-event",
        request_id="two-chunk-store",
    )

    assert [record.cache_key for record in receipt.sidecar_records] == [
        "lmcache:0.4.3:blake3:" + first_hash.hex(),
        "lmcache:0.4.3:blake3:" + second_hash.hex(),
    ]
    assert [record.source_range for record in receipt.sidecar_records] == [
        TokenRange(1000, 1256),
        TokenRange(1256, 1512),
    ]
    assert [record.token_ids for record in receipt.sidecar_records] == [
        document[:LMCACHE_CHUNK_SIZE],
        document[LMCACHE_CHUNK_SIZE:],
    ]

    moved_start = 17
    moved_chunk = document[:LMCACHE_CHUNK_SIZE]
    query = (*range(moved_start), *moved_chunk)
    queue.enqueue(
        LmcacheRequest.LOOKUP,
        [
            FakeMatch(
                0,
                LMCACHE_CHUNK_SIZE,
                moved_start,
                moved_start + LMCACHE_CHUNK_SIZE,
                first_hash,
            )
        ],
    )
    candidate = transport.lookup_candidates(query, request_id="moved-lookup")[0]
    plan = MatchPlanner(InMemoryRecordIndex(receipt.sidecar_records)).plan(
        transport.config.namespace,
        [TokenSegment.at(moved_start, moved_chunk)],
    )
    verified = VerifiedLmcacheCandidate.bind(
        candidate,
        plan.matches[0],
        expected_namespace=transport.config.namespace,
    )

    assert candidate.source_relative_range == TokenRange(0, LMCACHE_CHUNK_SIZE)
    assert verified.match.record.source_range == TokenRange(1000, 1256)
    assert verified.match.position_delta == moved_start - 1000


def test_precomputed_store_preserves_valid_hash_collision_bucket_records() -> None:
    colliding_hash = b"c" * 32
    bindings = FakeBindings(chunk_hashes=(colliding_hash, colliding_hash))
    transport, queue, _ = registered_transport(bindings=bindings)
    document = tuple(range(2 * LMCACHE_CHUNK_SIZE))
    queue.enqueue(LmcacheRequest.STORE_PRECOMPUTED, (b"server-event", True))

    receipt = transport.store_precomputed(
        document,
        cache_namespace=transport.config.namespace,
        document_source_range=TokenRange(2000, 2000 + len(document)),
        buffer_offset=0,
        event_ipc_handle=b"client-event",
        request_id="collision-store",
    )

    first, second = receipt.sidecar_records
    assert first.cache_key == second.cache_key
    assert first.token_ids != second.token_ids
    assert first.fingerprint != second.fingerprint


def test_precomputed_store_rejects_partial_chunk_without_rpc() -> None:
    transport, queue, bindings = registered_transport()
    calls_before = len(queue.calls)

    with pytest.raises(LmcacheConfigurationError, match="complete chunks"):
        transport.store_precomputed(
            range(LMCACHE_CHUNK_SIZE - 1),
            cache_namespace=transport.config.namespace,
            document_source_range=TokenRange(1000, 1255),
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="partial",
        )

    assert len(queue.calls) == calls_before
    assert bindings.hash_calls == []


@pytest.mark.parametrize(
    "document_source_range",
    [
        TokenRange(1000, 1255),
        TokenRange(131_000, 131_256),
    ],
)
def test_precomputed_store_rejects_unpublishable_source_range_before_rpc(
    document_source_range: TokenRange,
) -> None:
    transport, queue, bindings = registered_transport()
    calls_before = len(queue.calls)

    with pytest.raises(LmcacheConfigurationError, match="document_source_range"):
        transport.store_precomputed(
            range(LMCACHE_CHUNK_SIZE),
            cache_namespace=transport.config.namespace,
            document_source_range=document_source_range,
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="bad-source-range",
        )

    assert len(queue.calls) == calls_before
    assert bindings.hash_calls == []


def test_precomputed_store_rejects_namespace_mismatch_before_rpc() -> None:
    transport, queue, bindings = registered_transport()
    calls_before = len(queue.calls)
    other_namespace = replace(namespace(), model_revision="other-revision")

    with pytest.raises(LmcacheConfigurationError, match="namespace"):
        transport.store_precomputed(
            range(LMCACHE_CHUNK_SIZE),
            cache_namespace=other_namespace,
            document_source_range=TokenRange(1000, 1256),
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="wrong-namespace",
        )

    assert len(queue.calls) == calls_before
    assert bindings.hash_calls == []


def test_precomputed_store_rejects_tokens_outside_pinned_hasher_wire_domain() -> None:
    transport, queue, bindings = registered_transport()
    calls_before = len(queue.calls)
    document = [0] * LMCACHE_CHUNK_SIZE
    document[-1] = 1 << 32

    with pytest.raises(LmcacheConfigurationError, match="unsigned 32 bits"):
        transport.store_precomputed(
            document,
            cache_namespace=transport.config.namespace,
            document_source_range=TokenRange(1000, 1256),
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="token-wire-domain",
        )

    assert len(queue.calls) == calls_before
    assert bindings.hash_calls == []


@pytest.mark.parametrize(
    "chunk_hashes",
    [
        (),
        (b"a" * 32, b"b" * 32),
        (b"short",),
        (bytearray(b"a" * 32),),
        b"a" * 32,
    ],
)
def test_precomputed_store_fails_closed_on_hash_output_drift(
    chunk_hashes: object,
) -> None:
    bindings = FakeBindings(chunk_hashes=chunk_hashes)
    transport, queue, _ = registered_transport(bindings=bindings)
    queue.enqueue(LmcacheRequest.STORE_PRECOMPUTED, (b"server-event", True))

    with pytest.raises(LmcacheProtocolError, match="TokenHasher"):
        transport.store_precomputed(
            range(LMCACHE_CHUNK_SIZE),
            cache_namespace=transport.config.namespace,
            document_source_range=TokenRange(1000, 1256),
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="hash-drift",
        )

    assert queue.calls[-1].request == LmcacheRequest.STORE_PRECOMPUTED.value
    assert bindings.hash_calls == [tuple(range(LMCACHE_CHUNK_SIZE))]
    assert transport.state is LmcacheTransportState.FAILED


def test_final_store_is_not_claimed_as_non_prefix_candidate_registration() -> None:
    transport, queue, _ = registered_transport()
    queue.enqueue(LmcacheRequest.STORE_FINAL, (b"server-event", True))

    receipt = transport.store_final(
        range(LMCACHE_CHUNK_SIZE),
        buffer_offset=0,
        event_ipc_handle=b"client-event",
        request_id="final",
    )

    assert not receipt.candidate_lookup_required
    assert receipt.sidecar_records == ()
    assert queue.calls[-1].request == LmcacheRequest.STORE_FINAL.value


def test_rejected_transfer_fails_closed() -> None:
    transport, queue, bindings = registered_transport()
    queue.enqueue(LmcacheRequest.STORE_PRECOMPUTED, (b"server-event", False))

    with pytest.raises(LmcacheOperationError, match="reported failure"):
        transport.store_precomputed(
            range(LMCACHE_CHUNK_SIZE),
            cache_namespace=transport.config.namespace,
            document_source_range=TokenRange(1000, 1256),
            buffer_offset=0,
            event_ipc_handle=b"event",
            request_id="rejected",
        )

    assert transport.state is LmcacheTransportState.FAILED
    assert bindings.hash_calls == []


def test_close_unregisters_then_closes_and_is_idempotent() -> None:
    transport, queue, _ = registered_transport()
    queue.enqueue(LmcacheRequest.UNREGISTER, None)

    transport.close()
    transport.close()

    assert transport.state is LmcacheTransportState.CLOSED
    assert queue.closed
    assert queue.calls[-1].request == LmcacheRequest.UNREGISTER.value


def test_close_still_closes_mq_when_unregister_fails() -> None:
    transport, queue, _ = registered_transport()
    queue.enqueue(
        LmcacheRequest.UNREGISTER,
        error=TimeoutError("unregister timeout"),
    )

    with pytest.raises(LmcacheCloseError, match="unregister failed"):
        transport.close()

    assert transport.state is LmcacheTransportState.CLOSED
    assert queue.closed


def test_operations_after_close_are_rejected() -> None:
    transport, queue, _ = opened_transport()
    transport.close()

    with pytest.raises(LmcacheLifecycleError, match="closed"):
        transport.lookup_candidates(range(512), request_id="closed")
    assert queue.closed
