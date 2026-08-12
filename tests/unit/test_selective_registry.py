from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from hashlib import sha256
from pathlib import Path

import pytest

from cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_registry import (
    CUSTOM_ATTENTION_BACKEND_NAME,
    GPT_OSS_MODEL_ARCHITECTURE,
    SelectiveExtensionRegistrar,
    SelectiveGateEvidence,
    SelectivePrerequisites,
    SelectiveRegistrationError,
    SelectiveRegistrationErrorCode,
    SelectiveRegistrationSpec,
)


def _ready() -> SelectivePrerequisites:
    return SelectivePrerequisites(
        True,
        True,
        True,
        True,
        SelectiveGateEvidence(
            runtime_digest="a" * 64,
            full_prefill_digest="b" * 64,
            transfer_digest="c" * 64,
            yarn_digest="d" * 64,
            hybrid_sink_digest="e" * 64,
        ),
    )


def _spec(suffix: str = "one") -> SelectiveRegistrationSpec:
    return SelectiveRegistrationSpec(
        prerequisites=_ready(),
        model_class_path=(
            "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_model:"
            f"GptOssSelectiveModel{suffix}"
        ),
        attention_backend_class_path=(
            "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
            f"GptOssSelectiveBackend{suffix}"
        ),
    )


class FakeRegistries:
    def __init__(self) -> None:
        self.models: list[tuple[str, str]] = []
        self.backends: list[tuple[object, str]] = []

    def model(self, architecture: str, class_path: str) -> None:
        self.models.append((architecture, class_path))

    def backend(self, token: object, class_path: str) -> None:
        self.backends.append((token, class_path))


FakeBackendEnum = Enum(
    "AttentionBackendEnum",
    {CUSTOM_ATTENTION_BACKEND_NAME: CUSTOM_ATTENTION_BACKEND_NAME},
    module="vllm.v1.attention.backends.registry",
)


def test_valid_registration_is_idempotent_and_binds_custom_backend() -> None:
    fakes = FakeRegistries()
    registrar = SelectiveExtensionRegistrar()
    spec = _spec()

    first = registrar.register(
        spec,
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
    )
    second = registrar.register(
        spec,
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
    )

    assert first.registered and not first.already_registered
    assert not second.registered and second.already_registered
    assert fakes.models == [(GPT_OSS_MODEL_ARCHITECTURE, spec.model_class_path)]
    assert fakes.backends == [
        (CUSTOM_ATTENTION_BACKEND_NAME, spec.attention_backend_class_path)
    ]


def test_registration_conflict_is_rejected_without_second_call() -> None:
    fakes = FakeRegistries()
    registrar = SelectiveExtensionRegistrar()
    registrar.register(
        _spec("one"),
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
    )

    with pytest.raises(SelectiveRegistrationError) as error:
        registrar.register(
            _spec("two"),
            model_register=fakes.model,
            backend_register=fakes.backend,
            backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
        )

    assert error.value.code is SelectiveRegistrationErrorCode.REGISTRATION_CONFLICT
    assert len(fakes.models) == 1
    assert len(fakes.backends) == 1


def test_changed_evidence_digest_is_a_registration_conflict() -> None:
    fakes = FakeRegistries()
    registrar = SelectiveExtensionRegistrar()
    first = _spec()
    registrar.register(
        first,
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
    )
    assert first.prerequisites.evidence is not None
    changed_evidence = replace(
        first.prerequisites.evidence,
        transfer_digest="f" * 64,
    )
    changed = SelectiveRegistrationSpec(
        prerequisites=replace(first.prerequisites, evidence=changed_evidence),
        model_class_path=first.model_class_path,
        attention_backend_class_path=first.attention_backend_class_path,
    )

    with pytest.raises(SelectiveRegistrationError) as error:
        registrar.register(
            changed,
            model_register=fakes.model,
            backend_register=fakes.backend,
            backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
        )
    assert error.value.code is SelectiveRegistrationErrorCode.REGISTRATION_CONFLICT
    assert len(fakes.models) == 1
    assert len(fakes.backends) == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SelectiveRegistrationSpec(
            prerequisites=SelectivePrerequisites(False, True, True, True),
            model_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.model:GptOssModel"
            ),
            attention_backend_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.backend.GptOssBackend"
            ),
        ),
        lambda: SelectiveRegistrationSpec(
            prerequisites=_ready(),
            model_class_path="vllm.model_executor.models.gpt_oss:GptOssForCausalLM",
            attention_backend_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.backend.GptOssBackend"
            ),
        ),
        lambda: SelectiveRegistrationSpec(
            prerequisites=_ready(),
            model_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.model:GptOssModel"
            ),
            attention_backend_class_path="vllm.v1.attention.backends.TritonAttention",
        ),
    ],
)
def test_registration_spec_rejects_unproven_or_external_paths(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(SelectiveRegistrationError) as error:
        factory()

    assert error.value.code in {
        SelectiveRegistrationErrorCode.INVALID_PREREQUISITES,
        SelectiveRegistrationErrorCode.INVALID_MODEL_CLASS_PATH,
        SelectiveRegistrationErrorCode.INVALID_BACKEND_CLASS_PATH,
    }


def test_partial_registration_fails_closed_for_process_lifetime() -> None:
    fakes = FakeRegistries()

    def failing_backend(token: object, path: str) -> None:
        del token, path
        raise RuntimeError("fake backend failure")

    registrar = SelectiveExtensionRegistrar()
    with pytest.raises(SelectiveRegistrationError) as first:
        registrar.register(
            _spec(),
            model_register=fakes.model,
            backend_register=failing_backend,
            backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
        )
    assert first.value.code is SelectiveRegistrationErrorCode.PARTIAL_REGISTRATION

    with pytest.raises(SelectiveRegistrationError) as second:
        registrar.register(
            _spec(),
            model_register=fakes.model,
            backend_register=fakes.backend,
            backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
        )
    assert second.value.code is SelectiveRegistrationErrorCode.PARTIAL_REGISTRATION
    assert fakes.models == []


def test_non_custom_backend_token_is_rejected_before_registry_mutation() -> None:
    fakes = FakeRegistries()
    registrar = SelectiveExtensionRegistrar()

    with pytest.raises(SelectiveRegistrationError) as error:
        registrar.register(
            _spec(),
            model_register=fakes.model,
            backend_register=fakes.backend,
            backend_token="TRITON_ATTN",
        )

    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_BACKEND_TOKEN
    assert fakes.backends == []
    assert fakes.models == []

    receipt = registrar.register(
        _spec(),
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=CUSTOM_ATTENTION_BACKEND_NAME,
    )
    assert receipt.registered


@pytest.mark.parametrize(
    "token",
    [
        FakeBackendEnum.CUSTOM,
    ],
)
def test_enum_shaped_custom_token_is_accepted(token: object) -> None:
    fakes = FakeRegistries()
    receipt = SelectiveExtensionRegistrar().register(
        _spec(),
        model_register=fakes.model,
        backend_register=fakes.backend,
        backend_token=token,
    )

    assert receipt.registered
    assert fakes.models == [(GPT_OSS_MODEL_ARCHITECTURE, _spec().model_class_path)]


def test_unrelated_enum_with_custom_value_is_rejected() -> None:
    class UnrelatedBackendEnum(str, Enum):
        CUSTOM = CUSTOM_ATTENTION_BACKEND_NAME

    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveExtensionRegistrar().register(
            _spec(),
            model_register=FakeRegistries().model,
            backend_register=FakeRegistries().backend,
            backend_token=UnrelatedBackendEnum.CUSTOM,
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_BACKEND_TOKEN


def test_nested_model_class_name_is_rejected() -> None:
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveRegistrationSpec(
            prerequisites=_ready(),
            model_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_model:"
                "Outer.Inner"
            ),
            attention_backend_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
                "GptOssSelectiveBackend"
            ),
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_MODEL_CLASS_PATH


def test_arbitrary_object_with_custom_attributes_is_rejected() -> None:
    token = type("FakeObject", (), {"name": CUSTOM_ATTENTION_BACKEND_NAME})()
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveExtensionRegistrar().register(
            _spec(),
            model_register=FakeRegistries().model,
            backend_register=FakeRegistries().backend,
            backend_token=token,
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_BACKEND_TOKEN


def test_all_true_prerequisites_without_bound_gpu_evidence_are_not_ready() -> None:
    prerequisites = SelectivePrerequisites(True, True, True, True)
    assert not prerequisites.ready
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveRegistrationSpec(
            prerequisites=prerequisites,
            model_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.model:GptOssModel"
            ),
            attention_backend_class_path=(
                "cacheblend_gpt_oss.vllm_compat.v0_19_1.backend.GptOssBackend"
            ),
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_PREREQUISITES


@pytest.mark.parametrize("field", ["runtime_digest", "transfer_digest"])
def test_gate_evidence_requires_lowercase_sha256_digests(field: str) -> None:
    values = {
        "runtime_digest": "a" * 64,
        "full_prefill_digest": "b" * 64,
        "transfer_digest": "c" * 64,
        "yarn_digest": "d" * 64,
        "hybrid_sink_digest": "e" * 64,
    }
    values[field] = "not-a-digest"
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveGateEvidence(**values)
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_EVIDENCE


def test_gate_evidence_can_be_derived_from_regular_artifacts(tmp_path: Path) -> None:
    paths = []
    for index in range(5):
        path = tmp_path / f"artifact-{index}.json"
        path.write_bytes(f"artifact-{index}".encode("ascii"))
        paths.append(path)

    evidence = SelectiveGateEvidence.from_artifact_paths(
        runtime=paths[0],
        full_prefill=paths[1],
        transfer=paths[2],
        yarn=paths[3],
        hybrid_sink=paths[4],
    )

    assert evidence.runtime_digest == sha256(b"artifact-0").hexdigest()
    assert evidence.full_prefill_digest == sha256(b"artifact-1").hexdigest()
    assert evidence.transfer_digest == sha256(b"artifact-2").hexdigest()
    assert evidence.yarn_digest == sha256(b"artifact-3").hexdigest()
    assert evidence.hybrid_sink_digest == sha256(b"artifact-4").hexdigest()
    assert SelectiveGateEvidence.from_dict(evidence.to_dict()) == evidence


def test_gate_evidence_decoder_rejects_schema_and_extra_keys() -> None:
    values = _ready().evidence
    assert values is not None
    encoded = values.to_dict()
    encoded["unexpected"] = True
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveGateEvidence.from_dict(encoded)
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_EVIDENCE

    encoded = values.to_dict()
    encoded["schema_version"] = 2
    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveGateEvidence.from_dict(encoded)
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_EVIDENCE


@pytest.mark.parametrize("bad_kind", ["missing", "directory", "symlink"])
def test_gate_evidence_artifact_failures_are_bounded(
    tmp_path: Path, bad_kind: str
) -> None:
    paths = []
    for index in range(5):
        path = tmp_path / f"artifact-{index}"
        path.write_bytes(b"ok")
        paths.append(path)
    if bad_kind == "missing":
        paths[2] = tmp_path / "missing"
    elif bad_kind == "directory":
        paths[2].unlink()
        paths[2].mkdir()
    else:
        target = tmp_path / "target"
        target.write_bytes(b"ok")
        paths[2].unlink()
        paths[2].symlink_to(target)

    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveGateEvidence.from_artifact_paths(
            runtime=paths[0],
            full_prefill=paths[1],
            transfer=paths[2],
            yarn=paths[3],
            hybrid_sink=paths[4],
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_EVIDENCE


def test_gate_evidence_rejects_relative_artifact_paths(tmp_path: Path) -> None:
    paths = []
    for index in range(5):
        path = tmp_path / f"artifact-{index}"
        path.write_bytes(b"ok")
        paths.append(path)

    with pytest.raises(SelectiveRegistrationError) as error:
        SelectiveGateEvidence.from_artifact_paths(
            runtime=Path("relative-runtime.txt"),
            full_prefill=paths[1],
            transfer=paths[2],
            yarn=paths[3],
            hybrid_sink=paths[4],
        )
    assert error.value.code is SelectiveRegistrationErrorCode.INVALID_EVIDENCE
