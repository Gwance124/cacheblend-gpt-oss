#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one fresh CacheBlend artifact against a frozen probability gate.

This CLI assumes the separately-owned
``cacheblend_gpt_oss.correctness.probability_ensemble`` module exposes the
interface documented in ``freeze_probability_ensemble.py`` plus
``evaluate_probability_candidate(ensemble, candidate)``.  Its policy
constants and probability comparison logic are never duplicated here.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    artifact_digest,
    read_artifact,
    read_transfer_evidence,
    transfer_evidence_digest,
    validate_transfer_evidence_binding,
)

_BASELINE_COUNT = 5
_PROBABILITY_MODULE = "cacheblend_gpt_oss.correctness.probability_ensemble"
_REQUIRED_CALLABLES = (
    "build_probability_baseline_ensemble",
    "canonical_manifest_bytes",
    "evaluate_probability_candidate",
    "manifest_digest",
    "manifest_from_dict",
    "manifest_to_dict",
)
_REQUIRED_SYMBOLS = (
    "ProbabilityEnsembleStatus",
    "ProbabilityBaselineComparison",
    "ProbabilityCandidateComparison",
)


def _load_probability_module() -> ModuleType:
    try:
        module = importlib.import_module(_PROBABILITY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _PROBABILITY_MODULE:
            raise RuntimeError(
                "probability-aware gate unavailable: "
                f"{_PROBABILITY_MODULE} is not present; expected API is "
                "build_probability_baseline_ensemble, "
                "evaluate_probability_candidate, canonical_manifest_bytes, "
                "manifest_digest, manifest_from_dict, manifest_to_dict, "
                "ProbabilityEnsembleStatus, ProbabilityBaselineComparison, "
                "and ProbabilityCandidateComparison"
            ) from exc
        raise

    missing = [
        name
        for name in (*_REQUIRED_CALLABLES, *_REQUIRED_SYMBOLS)
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "probability-aware gate module has an incomplete assumed API: "
            + ", ".join(sorted(missing))
        )
    if any(not callable(getattr(module, name)) for name in _REQUIRED_CALLABLES):
        raise RuntimeError(
            "probability-aware gate module has non-callable required helpers"
        )
    return module


def _status_value(status: object) -> object:
    return status.value if isinstance(status, Enum) else status


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return _status_value(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_value(item) for item in value]
    if hasattr(value, "_asdict"):
        return _json_value(value._asdict())  # type: ignore[attr-defined]
    if hasattr(value, "__dict__"):
        return _json_value(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        return _json_value(
            {
                key: getattr(value, key)
                for key in slots
                if isinstance(key, str) and hasattr(value, key)
            }
        )
    raise TypeError(f"probability comparison is not JSON serializable: {type(value)!r}")


def _records(value: object, names: tuple[str, ...]) -> list[object]:
    for name in names:
        records = getattr(value, name, None)
        if records is not None:
            return list(records)
    return []


def _read_manifest(module: ModuleType, path: Path) -> object:
    return module.manifest_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _require_exact_manifest(
    module: ModuleType,
    frozen_manifest: object,
    rebuilt_manifest: object,
) -> str:
    frozen_bytes = module.canonical_manifest_bytes(frozen_manifest)
    rebuilt_bytes = module.canonical_manifest_bytes(rebuilt_manifest)
    if frozen_bytes != rebuilt_bytes:
        raise ValueError(
            "frozen probability ensemble manifest does not bind these baselines"
        )
    frozen_digest = module.manifest_digest(frozen_manifest)
    rebuilt_digest = module.manifest_digest(rebuilt_manifest)
    if frozen_digest != rebuilt_digest:
        raise ValueError(
            "frozen probability ensemble manifest digest does not bind these baselines"
        )
    return frozen_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly five times in the frozen manifest's order",
    )
    parser.add_argument(
        "--excluded-candidate",
        type=Path,
        required=True,
        help="strict-v1 pilot bound into the v2 manifest and excluded",
    )
    parser.add_argument("--cacheblend", type=Path, required=True)
    parser.add_argument(
        "--transfer-evidence",
        type=Path,
        required=True,
        help="required all-layer transfer sidecar bound to the candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != _BASELINE_COUNT:
        parser.error(f"--baseline must be supplied exactly {_BASELINE_COUNT} times")
    if args.output.exists():
        raise FileExistsError("probability ensemble verdict output already exists")

    module = _load_probability_module()
    frozen_manifest = _read_manifest(module, args.manifest)
    baselines = tuple(read_artifact(path) for path in args.baseline)
    excluded_candidate = read_artifact(args.excluded_candidate)
    ensemble = module.build_probability_baseline_ensemble(
        baselines,
        excluded_candidates=(excluded_candidate,),
    )
    manifest = ensemble.manifest
    manifest_digest_value = _require_exact_manifest(
        module,
        frozen_manifest,
        manifest,
    )

    candidate = read_artifact(args.cacheblend)
    transfer = read_transfer_evidence(args.transfer_evidence)
    validate_transfer_evidence_binding(candidate, transfer)
    verdict = module.evaluate_probability_candidate(ensemble, candidate)

    status = _status_value(verdict.status)
    passed = bool(getattr(verdict, "passed", status == "PASS"))
    failure_reasons = _json_value(getattr(verdict, "failure_reasons", ()))
    comparisons = [
        _json_value(record)
        for record in _records(
            verdict,
            ("comparisons", "candidate_comparisons", "probability_comparisons"),
        )
    ]
    candidate_digest = getattr(verdict, "candidate_artifact_digest", None)
    if candidate_digest is None:
        candidate_digest = artifact_digest(candidate)
    report = {
        "schema_version": 1,
        "status": status,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "candidate_artifact_digest": candidate_digest,
        "manifest_digest": manifest_digest_value,
        "manifest_rebuilt_and_bound": True,
        "transfer_evidence_digest": transfer_evidence_digest(transfer),
        "transfer_evidence_bound": True,
        "comparisons": comparisons,
        "verdict": _json_value(verdict),
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
