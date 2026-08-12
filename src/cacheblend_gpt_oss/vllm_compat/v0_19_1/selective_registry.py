# SPDX-License-Identifier: Apache-2.0
"""Fail-closed registration boundary for the future M6 selective extension.

The pinned vLLM 0.19.1 source explicitly provides these two public seams:

* ``ModelRegistry.register_model`` accepts a lazy ``<module>:<class>`` path:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/registry.py#L894-L938
* ``AttentionBackendEnum.CUSTOM`` and ``register_backend`` accept a dotted
  backend class path:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/registry.py#L34-L118
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/registry.py#L205-L262

This module only guards those calls.  It does not provide the model override,
attention implementation, or a ``vllm.general_plugins`` entry point.  The
current connector remains the 100%-recompute path.  A future plugin must supply
real class paths and an immutable proof that full-prefill equivalence, transfer,
YaRN correction, hybrid groups, and sink behavior have passed on the pinned
target before this registrar is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, NoReturn, Protocol

GPT_OSS_MODEL_ARCHITECTURE = "GptOssForCausalLM"
CUSTOM_ATTENTION_BACKEND_NAME = "CUSTOM"
_PACKAGE_PREFIX = "cacheblend_gpt_oss.vllm_compat.v0_19_1."


class SelectiveRegistrationErrorCode(str, Enum):
    """Bounded failures suitable for startup diagnostics."""

    INVALID_PREREQUISITES = "invalid_prerequisites"
    INVALID_MODEL_ARCHITECTURE = "invalid_model_architecture"
    INVALID_MODEL_CLASS_PATH = "invalid_model_class_path"
    INVALID_BACKEND_CLASS_PATH = "invalid_backend_class_path"
    REGISTRATION_CONFLICT = "registration_conflict"
    PARTIAL_REGISTRATION = "partial_registration"


class SelectiveRegistrationError(RuntimeError):
    """Fail-closed registration error without exposing path contents."""

    def __init__(self, code: SelectiveRegistrationErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend selective registration failure: {code.value}")


def _fail(code: SelectiveRegistrationErrorCode) -> NoReturn:
    raise SelectiveRegistrationError(code)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class SelectivePrerequisites:
    """Immutable proof inputs required before registering selective classes."""

    full_prefill_equivalence: bool
    transfer_100pct: bool
    yarn_correction: bool
    hybrid_groups_and_sinks: bool

    def __post_init__(self) -> None:
        if not all(
            _is_bool(value)
            for value in (
                self.full_prefill_equivalence,
                self.transfer_100pct,
                self.yarn_correction,
                self.hybrid_groups_and_sinks,
            )
        ):
            _fail(SelectiveRegistrationErrorCode.INVALID_PREREQUISITES)

    @property
    def ready(self) -> bool:
        return (
            self.full_prefill_equivalence
            and self.transfer_100pct
            and self.yarn_correction
            and self.hybrid_groups_and_sinks
        )


def _valid_model_class_path(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_PACKAGE_PREFIX):
        return False
    module_name, separator, class_name = value.partition(":")
    return bool(
        separator
        and module_name
        and class_name
        and ":" not in class_name
        and all(part.isidentifier() for part in module_name.split("."))
        and all(part.isidentifier() for part in class_name.split("."))
    )


def _valid_backend_class_path(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_PACKAGE_PREFIX):
        return False
    return all(part.isidentifier() for part in value.split("."))


@dataclass(frozen=True, slots=True)
class SelectiveRegistrationSpec:
    """Class paths and proof required for one registration attempt."""

    prerequisites: SelectivePrerequisites
    model_class_path: str
    attention_backend_class_path: str
    model_architecture: str = GPT_OSS_MODEL_ARCHITECTURE

    def __post_init__(self) -> None:
        if self.model_architecture != GPT_OSS_MODEL_ARCHITECTURE:
            _fail(SelectiveRegistrationErrorCode.INVALID_MODEL_ARCHITECTURE)
        if not isinstance(self.prerequisites, SelectivePrerequisites):
            _fail(SelectiveRegistrationErrorCode.INVALID_PREREQUISITES)
        if not self.prerequisites.ready:
            _fail(SelectiveRegistrationErrorCode.INVALID_PREREQUISITES)
        if not _valid_model_class_path(self.model_class_path):
            _fail(SelectiveRegistrationErrorCode.INVALID_MODEL_CLASS_PATH)
        if not _valid_backend_class_path(self.attention_backend_class_path):
            _fail(SelectiveRegistrationErrorCode.INVALID_BACKEND_CLASS_PATH)

    @property
    def signature(self) -> tuple[str, str, str]:
        return (
            self.model_architecture,
            self.model_class_path,
            self.attention_backend_class_path,
        )


@dataclass(frozen=True, slots=True)
class SelectiveRegistrationReceipt:
    """Bounded result of an idempotent registration call."""

    registered: bool
    already_registered: bool


class ModelRegister(Protocol):
    def __call__(self, model_architecture: str, model_class_path: str) -> None: ...


class BackendRegister(Protocol):
    def __call__(self, backend: object, backend_class_path: str) -> Any: ...


class SelectiveExtensionRegistrar:
    """One-process idempotent registrar used by a future general plugin."""

    def __init__(self) -> None:
        self._signature: tuple[str, str, str] | None = None
        self._partial = False

    def register(
        self,
        spec: SelectiveRegistrationSpec,
        *,
        model_register: ModelRegister | None = None,
        backend_register: BackendRegister | None = None,
        backend_token: object | None = None,
    ) -> SelectiveRegistrationReceipt:
        if not isinstance(spec, SelectiveRegistrationSpec):
            _fail(SelectiveRegistrationErrorCode.INVALID_PREREQUISITES)
        if self._partial:
            _fail(SelectiveRegistrationErrorCode.PARTIAL_REGISTRATION)

        signature = spec.signature
        if self._signature is not None:
            if self._signature != signature:
                _fail(SelectiveRegistrationErrorCode.REGISTRATION_CONFLICT)
            return SelectiveRegistrationReceipt(
                registered=False,
                already_registered=True,
            )

        try:
            if (
                model_register is None
                or backend_register is None
                or backend_token is None
            ):
                model_register, backend_register, backend_token = _load_vllm_registries(
                    model_register=model_register,
                    backend_register=backend_register,
                    backend_token=backend_token,
                )
            # Register the backend first so a model lookup cannot resolve a
            # selective model while CUSTOM is still unbound in this process.
            assert backend_register is not None
            assert backend_token is not None
            backend_register(backend_token, spec.attention_backend_class_path)
            assert model_register is not None
            model_register(spec.model_architecture, spec.model_class_path)
        except Exception as error:
            self._partial = True
            if isinstance(error, SelectiveRegistrationError):
                raise
            _fail(SelectiveRegistrationErrorCode.PARTIAL_REGISTRATION)

        self._signature = signature
        return SelectiveRegistrationReceipt(registered=True, already_registered=False)


def _load_vllm_registries(
    *,
    model_register: ModelRegister | None,
    backend_register: BackendRegister | None,
    backend_token: object | None,
) -> tuple[ModelRegister, BackendRegister, object]:
    """Import pinned vLLM registries lazily, never on CPU package import."""

    if model_register is None:
        models = import_module("vllm.model_executor.models")
        model_register = models.ModelRegistry.register_model
    if backend_register is None or backend_token is None:
        registry = import_module("vllm.v1.attention.backends.registry")
        if backend_register is None:
            backend_register = registry.register_backend
        if backend_token is None:
            backend_token = registry.AttentionBackendEnum.CUSTOM
    return model_register, backend_register, backend_token


_DEFAULT_REGISTRAR = SelectiveExtensionRegistrar()


def register_selective_extension(
    spec: SelectiveRegistrationSpec,
    *,
    model_register: ModelRegister | None = None,
    backend_register: BackendRegister | None = None,
    backend_token: object | None = None,
) -> SelectiveRegistrationReceipt:
    """Register one gated spec through the process-local default registrar."""

    return _DEFAULT_REGISTRAR.register(
        spec,
        model_register=model_register,
        backend_register=backend_register,
        backend_token=backend_token,
    )


__all__ = [
    "CUSTOM_ATTENTION_BACKEND_NAME",
    "GPT_OSS_MODEL_ARCHITECTURE",
    "SelectiveExtensionRegistrar",
    "SelectivePrerequisites",
    "SelectiveRegistrationError",
    "SelectiveRegistrationErrorCode",
    "SelectiveRegistrationReceipt",
    "SelectiveRegistrationSpec",
    "register_selective_extension",
]
