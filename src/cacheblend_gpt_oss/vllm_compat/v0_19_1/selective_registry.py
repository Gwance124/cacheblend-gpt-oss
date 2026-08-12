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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

GPT_OSS_MODEL_ARCHITECTURE = "GptOssForCausalLM"
CUSTOM_ATTENTION_BACKEND_NAME = "CUSTOM"
_PACKAGE_PREFIX = "cacheblend_gpt_oss.vllm_compat.v0_19_1."
SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION = 1
_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "runtime_digest",
        "full_prefill_digest",
        "transfer_digest",
        "yarn_digest",
        "hybrid_sink_digest",
    }
)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class SelectiveRegistrationErrorCode(str, Enum):
    """Bounded failures suitable for startup diagnostics."""

    INVALID_PREREQUISITES = "invalid_prerequisites"
    INVALID_EVIDENCE = "invalid_evidence"
    INVALID_MODEL_ARCHITECTURE = "invalid_model_architecture"
    INVALID_MODEL_CLASS_PATH = "invalid_model_class_path"
    INVALID_BACKEND_CLASS_PATH = "invalid_backend_class_path"
    INVALID_BACKEND_TOKEN = "invalid_backend_token"
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
class SelectiveGateEvidence:
    """Immutable digests binding registration to pinned GPU artifacts.

    The booleans in :class:`SelectivePrerequisites` are assertions about the
    artifacts; these digests make those assertions auditable and prevent a
    future plugin entry point from being enabled with four unbound ``True``
    values. The files themselves stay outside the runtime package and are
    supplied by the solab-g3 experiment hand-off.
    """

    runtime_digest: str
    full_prefill_digest: str
    transfer_digest: str
    yarn_digest: str
    hybrid_sink_digest: str

    def to_dict(self) -> dict[str, object]:
        """Return the strict, identifier-free handoff representation."""

        return {
            "schema_version": SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION,
            "runtime_digest": self.runtime_digest,
            "full_prefill_digest": self.full_prefill_digest,
            "transfer_digest": self.transfer_digest,
            "yarn_digest": self.yarn_digest,
            "hybrid_sink_digest": self.hybrid_sink_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> SelectiveGateEvidence:
        """Decode one canonical evidence handoff without trusting extra keys."""

        if (
            not isinstance(value, Mapping)
            or frozenset(value) != _EVIDENCE_KEYS
            or isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION
        ):
            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
        fields = (
            "runtime_digest",
            "full_prefill_digest",
            "transfer_digest",
            "yarn_digest",
            "hybrid_sink_digest",
        )
        values = tuple(value.get(field) for field in fields)
        if any(not _is_digest(item) for item in values):
            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
        return cls(*cast(tuple[str, ...], values))

    @classmethod
    def from_artifact_paths(
        cls,
        *,
        runtime: Path,
        full_prefill: Path,
        transfer: Path,
        yarn: Path,
        hybrid_sink: Path,
    ) -> SelectiveGateEvidence:
        """Bind evidence to exact immutable files from the GPU hand-off.

        The registrar only consumes digests, but accepting operator-entered
        hex strings alone would make the proof bundle unauditable.  This
        helper hashes regular files with bounded size, rejects symlinks and
        missing paths, and maps every filesystem failure to the same bounded
        ``INVALID_EVIDENCE`` code.  Artifact semantics remain the caller's
        responsibility and are reviewed separately before registration.
        """

        paths = (runtime, full_prefill, transfer, yarn, hybrid_sink)
        digests: list[str] = []
        try:
            for path in paths:
                if (
                    not isinstance(path, Path)
                    or not path.is_absolute()
                    or path.is_symlink()
                ):
                    _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
                stat = path.stat()
                if (
                    not path.is_file()
                    or stat.st_size <= 0
                    or stat.st_size > 64 * 1024 * 1024
                ):
                    _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
                digest = sha256()
                with path.open("rb") as handle:
                    remaining = stat.st_size
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
                        digest.update(chunk)
                        remaining -= len(chunk)
                after = path.stat()
                if (
                    after.st_size != stat.st_size
                    or after.st_mtime_ns != stat.st_mtime_ns
                    or after.st_ino != stat.st_ino
                    or after.st_dev != stat.st_dev
                ):
                    _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
                digests.append(digest.hexdigest())
        except SelectiveRegistrationError:
            raise
        except (OSError, ValueError, TypeError):
            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)
        return cls(*digests)

    def __post_init__(self) -> None:
        if not all(
            _is_digest(value)
            for value in (
                self.runtime_digest,
                self.full_prefill_digest,
                self.transfer_digest,
                self.yarn_digest,
                self.hybrid_sink_digest,
            )
        ):
            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)


@dataclass(frozen=True, slots=True)
class SelectivePrerequisites:
    """Immutable proof inputs required before registering selective classes."""

    full_prefill_equivalence: bool
    transfer_100pct: bool
    yarn_correction: bool
    hybrid_groups_and_sinks: bool
    evidence: SelectiveGateEvidence | None = None

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
        if self.evidence is not None and not isinstance(
            self.evidence, SelectiveGateEvidence
        ):
            _fail(SelectiveRegistrationErrorCode.INVALID_EVIDENCE)

    @property
    def ready(self) -> bool:
        return (
            self.full_prefill_equivalence
            and self.transfer_100pct
            and self.yarn_correction
            and self.hybrid_groups_and_sinks
            and self.evidence is not None
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
        and class_name.isidentifier()
    )


def _valid_backend_class_path(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_PACKAGE_PREFIX):
        return False
    return all(part.isidentifier() for part in value.split("."))


def _is_custom_backend_token(value: object) -> bool:
    """Accept only the pinned registry's ``CUSTOM`` selector.

    The real vLLM value is an enum member, while CPU tests and dependency-
    injected callers may use the stable string name.  Checking both the enum
    name and value keeps this module free of a top-level vLLM import without
    allowing an arbitrary backend selector to reach ``register_backend``.
    """

    if type(value) is str and value == CUSTOM_ATTENTION_BACKEND_NAME:
        return True
    if not isinstance(value, Enum):
        return False
    enum_type = type(value)
    # The pinned token is defined by vLLM's registry, rather than by a generic
    # enum protocol.  Accepting any unrelated enum with a coincidental
    # ``CUSTOM`` value would let dependency-injected callers select a backend
    # that the registry does not understand.
    return (
        enum_type.__module__ == "vllm.v1.attention.backends.registry"
        and enum_type.__name__ == "AttentionBackendEnum"
        and (
            value.name == CUSTOM_ATTENTION_BACKEND_NAME
            or value.value == CUSTOM_ATTENTION_BACKEND_NAME
        )
    )


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
    def signature(self) -> tuple[str, str, str, tuple[str, ...]]:
        assert self.prerequisites.evidence is not None
        evidence = self.prerequisites.evidence
        return (
            self.model_architecture,
            self.model_class_path,
            self.attention_backend_class_path,
            (
                evidence.runtime_digest,
                evidence.full_prefill_digest,
                evidence.transfer_digest,
                evidence.yarn_digest,
                evidence.hybrid_sink_digest,
            ),
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
        self._signature: tuple[str, str, str, tuple[str, ...]] | None = None
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

        # Validate an injected selector before entering the partial-
        # registration transaction.  A caller typo must not poison an
        # otherwise unused registrar; only a failure after a registry call is
        # treated as an irreversible partial registration.
        if backend_token is not None and not _is_custom_backend_token(backend_token):
            _fail(SelectiveRegistrationErrorCode.INVALID_BACKEND_TOKEN)

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
            if not _is_custom_backend_token(backend_token):
                _fail(SelectiveRegistrationErrorCode.INVALID_BACKEND_TOKEN)
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
    "SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION",
    "SelectiveExtensionRegistrar",
    "SelectiveGateEvidence",
    "SelectivePrerequisites",
    "SelectiveRegistrationError",
    "SelectiveRegistrationErrorCode",
    "SelectiveRegistrationReceipt",
    "SelectiveRegistrationSpec",
    "register_selective_extension",
]
