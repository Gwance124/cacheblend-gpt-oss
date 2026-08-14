#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze the five-baseline input for the probability-aware prospective gate.

The probability policy is intentionally owned by
``cacheblend_gpt_oss.correctness.probability_ensemble``.  This CLI assumes that
module exposes these symbols:

* ``build_probability_baseline_ensemble(baselines, excluded_candidates=...)``;
* ``canonical_manifest_bytes(manifest)``;
* ``manifest_digest(manifest)``;
* ``manifest_from_dict(data)`` and ``manifest_to_dict(manifest)``;
* ``ProbabilityEnsembleStatus``;
* ``ProbabilityBaselineComparison`` and ``ProbabilityCandidateComparison``
  record types.

The last three symbols are interface markers for the status and comparison
records.  The CLI serializes the records structurally, so their concrete
probability metrics remain owned by the module.
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

from cacheblend_gpt_oss.correctness import read_artifact  # noqa: E402

_BASELINE_COUNT = 5
_PROBABILITY_MODULE = "cacheblend_gpt_oss.correctness.probability_ensemble"
_REQUIRED_CALLABLES = (
    "build_probability_baseline_ensemble",
    "canonical_manifest_bytes",
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
    """Load and validate the separately-owned probability gate interface."""

    try:
        module = importlib.import_module(_PROBABILITY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _PROBABILITY_MODULE:
            raise RuntimeError(
                "probability-aware gate unavailable: "
                f"{_PROBABILITY_MODULE} is not present; expected API is "
                "build_probability_baseline_ensemble, canonical_manifest_bytes, "
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
    """Convert module-owned records to JSON without knowing their metrics."""

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
    del module
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(module: ModuleType, path: Path, manifest: object) -> None:
    encoded = module.canonical_manifest_bytes(manifest)
    if not isinstance(encoded, bytes):
        raise TypeError("canonical_manifest_bytes must return bytes")
    with path.open("xb") as output:
        output.write(encoded + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly five times in canonical capture order",
    )
    parser.add_argument(
        "--excluded-candidate",
        type=Path,
        required=True,
        help="immutable strict-v1 pilot candidate, ineligible for v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != _BASELINE_COUNT:
        parser.error(f"--baseline must be supplied exactly {_BASELINE_COUNT} times")
    if args.output.exists():
        raise FileExistsError("probability ensemble manifest output already exists")

    module = _load_probability_module()
    baselines = tuple(read_artifact(path) for path in args.baseline)
    excluded_candidate = read_artifact(args.excluded_candidate)
    ensemble = module.build_probability_baseline_ensemble(
        baselines,
        excluded_candidates=(excluded_candidate,),
    )
    manifest = ensemble.manifest
    _write_manifest(module, args.output, manifest)

    status = _status_value(ensemble.status)
    report = {
        "schema_version": 1,
        "manifest": _json_value(module.manifest_to_dict(manifest)),
        "manifest_digest": module.manifest_digest(manifest),
        "status": status,
        "stable": status == "PASS",
        "baseline_comparisons": [
            _json_value(record)
            for record in _records(
                ensemble,
                ("comparisons", "baseline_comparisons", "pairwise_comparisons"),
            )
        ],
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
