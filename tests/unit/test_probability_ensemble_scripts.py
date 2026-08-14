# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the probability-aware prospective-gate CLIs."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_freeze = runpy.run_path(
    "scripts/freeze_probability_ensemble.py",
    run_name="cacheblend_probability_freeze_script_test",
)
_evaluate = runpy.run_path(
    "scripts/evaluate_probability_ensemble.py",
    run_name="cacheblend_probability_evaluate_script_test",
)


class _Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class _Comparison:
    left_index: int
    right_index: int
    probability_error: float


@dataclass(frozen=True)
class _CandidateComparison:
    baseline_index: int
    probability_error: float


@dataclass(frozen=True)
class _Artifact:
    digest: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fake_probability_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict_status: _Status = _Status.PASS,
) -> ModuleType:
    module = ModuleType(
        "cacheblend_gpt_oss.correctness.probability_ensemble"
    )

    def manifest_to_dict(manifest: dict[str, Any]) -> dict[str, Any]:
        return dict(manifest)

    def manifest_from_dict(data: object) -> dict[str, Any]:
        assert isinstance(data, dict)
        return dict(data)

    def manifest_digest(manifest: dict[str, Any]) -> str:
        return hashlib.sha256(_canonical(manifest)).hexdigest()

    def build_probability_baseline_ensemble(
        baselines: tuple[_Artifact, ...],
        *,
        excluded_candidates: tuple[_Artifact, ...],
    ) -> SimpleNamespace:
        manifest = {
            "policy_version": "probability-test-v1",
            "artifact_digests": [artifact.digest for artifact in baselines],
            "excluded_candidate_artifact_digests": [
                artifact.digest for artifact in excluded_candidates
            ],
        }
        comparisons = tuple(
            _Comparison(left, right, 0.01)
            for left in range(5)
            for right in range(left + 1, 5)
        )
        return SimpleNamespace(
            baselines=baselines,
            manifest=manifest,
            status=_Status.PASS,
            comparisons=comparisons,
        )

    def evaluate_probability_candidate(
        ensemble: SimpleNamespace,
        candidate: _Artifact,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status=verdict_status,
            passed=verdict_status is _Status.PASS,
            failure_reasons=()
            if verdict_status is _Status.PASS
            else ("probability_gate_failed",),
            candidate_artifact_digest=candidate.digest,
            comparisons=tuple(
                _CandidateComparison(index, 0.02) for index in range(5)
            ),
        )

    module.build_probability_baseline_ensemble = (  # type: ignore[attr-defined]
        build_probability_baseline_ensemble
    )
    module.evaluate_probability_candidate = (  # type: ignore[attr-defined]
        evaluate_probability_candidate
    )
    module.canonical_manifest_bytes = (  # type: ignore[attr-defined]
        _canonical
    )
    module.manifest_digest = manifest_digest  # type: ignore[attr-defined]
    module.manifest_from_dict = manifest_from_dict  # type: ignore[attr-defined]
    module.manifest_to_dict = manifest_to_dict  # type: ignore[attr-defined]
    module.ProbabilityEnsembleStatus = _Status  # type: ignore[attr-defined]
    module.ProbabilityBaselineComparison = _Comparison  # type: ignore[attr-defined]
    module.ProbabilityCandidateComparison = (  # type: ignore[attr-defined]
        _CandidateComparison
    )
    monkeypatch.setitem(sys.modules, _freeze["_PROBABILITY_MODULE"], module)
    monkeypatch.setitem(sys.modules, _evaluate["_PROBABILITY_MODULE"], module)
    return module


def _patch_artifacts(namespace: dict[str, Any]) -> dict[str, _Artifact]:
    artifacts = {
        f"artifact-{index}.json": _Artifact(f"{index:064x}")
        for index in range(1, 8)
    }
    namespace["main"].__globals__["read_artifact"] = (  # type: ignore[index]
        lambda path: artifacts[path.name]
    )
    return artifacts


def _baseline_args(paths: list[Path]) -> list[str]:
    arguments: list[str] = []
    for path in paths:
        arguments.extend(("--baseline", str(path)))
    return arguments


def test_freeze_requires_exactly_five_and_has_no_threshold_flags() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/freeze_probability_ensemble.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--baseline" in result.stdout
    assert "--excluded-candidate" in result.stdout
    assert "--output" in result.stdout
    assert "threshold" not in result.stdout


def test_freeze_writes_canonical_manifest_create_only_and_prints_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _fake_probability_module(monkeypatch)
    artifacts = _patch_artifacts(_freeze)
    output = tmp_path / "probability-manifest.json"
    baselines = [tmp_path / f"artifact-{index}.json" for index in range(1, 6)]
    for path in baselines:
        path.touch()
    excluded_candidate = tmp_path / "artifact-6.json"
    excluded_candidate.touch()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_probability_ensemble.py",
            *_baseline_args(baselines),
            "--excluded-candidate",
            str(excluded_candidate),
            "--output",
            str(output),
        ],
    )
    assert _freeze["main"]() == 0
    report = json.loads(capsys.readouterr().out)
    expected_manifest = {
        "policy_version": "probability-test-v1",
        "artifact_digests": [artifacts[path.name].digest for path in baselines],
        "excluded_candidate_artifact_digests": [
            artifacts[excluded_candidate.name].digest
        ],
    }
    assert json.loads(output.read_text(encoding="utf-8")) == expected_manifest
    assert report["manifest"] == expected_manifest
    assert report["status"] == "PASS"
    assert len(report["baseline_comparisons"]) == 10

    with pytest.raises(FileExistsError, match="manifest output already exists"):
        _freeze["main"]()
    assert module is sys.modules[_freeze["_PROBABILITY_MODULE"]]


def test_evaluate_rebuilds_manifest_binds_transfer_and_returns_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_probability_module(monkeypatch, verdict_status=_Status.FAIL)
    artifacts = _patch_artifacts(_evaluate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "policy_version": "probability-test-v1",
                "artifact_digests": [f"{index:064x}" for index in range(1, 6)],
                "excluded_candidate_artifact_digests": [f"{6:064x}"],
            }
        ),
        encoding="utf-8",
    )
    baselines = [tmp_path / f"artifact-{index}.json" for index in range(1, 6)]
    for path in baselines:
        path.touch()
    excluded_candidate_path = tmp_path / "artifact-6.json"
    excluded_candidate_path.touch()
    candidate_path = tmp_path / "artifact-7.json"
    candidate_path.touch()
    evidence_path = tmp_path / "transfer-evidence.json"
    evidence_path.write_text("{}", encoding="utf-8")
    verdict_path = tmp_path / "verdict.json"

    transfer = object()
    calls: list[tuple[_Artifact, object]] = []
    evaluate_globals = _evaluate["main"].__globals__
    evaluate_globals["read_transfer_evidence"] = lambda path: transfer
    evaluate_globals["validate_transfer_evidence_binding"] = (
        lambda candidate, evidence: calls.append((candidate, evidence))
    )
    evaluate_globals["transfer_evidence_digest"] = lambda evidence: "e" * 64

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_probability_ensemble.py",
            "--manifest",
            str(manifest),
            *_baseline_args(baselines),
            "--excluded-candidate",
            str(excluded_candidate_path),
            "--cacheblend",
            str(candidate_path),
            "--transfer-evidence",
            str(evidence_path),
            "--output",
            str(verdict_path),
        ],
    )
    assert _evaluate["main"]() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert report["passed"] is False
    assert report["manifest_rebuilt_and_bound"] is True
    assert report["transfer_evidence_bound"] is True
    assert report["transfer_evidence_digest"] == "e" * 64
    assert len(report["comparisons"]) == 5
    assert calls == [(artifacts[candidate_path.name], transfer)]


def test_evaluate_requires_transfer_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_probability_module(monkeypatch)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_probability_ensemble.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--baseline",
            str(tmp_path / "one.json"),
            "--baseline",
            str(tmp_path / "two.json"),
            "--baseline",
            str(tmp_path / "three.json"),
            "--baseline",
            str(tmp_path / "four.json"),
            "--baseline",
            str(tmp_path / "five.json"),
            "--cacheblend",
            str(tmp_path / "candidate.json"),
            "--excluded-candidate",
            str(tmp_path / "pilot.json"),
            "--output",
            str(tmp_path / "verdict.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--transfer-evidence" in result.stderr
