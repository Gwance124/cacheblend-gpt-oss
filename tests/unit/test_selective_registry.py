from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

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
