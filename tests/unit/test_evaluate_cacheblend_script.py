"""CPU-only contract tests for the correctness-evaluator CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.correctness import CorrectnessCase

_evaluator = runpy.run_path(
    "scripts/evaluate_cacheblend_correctness.py",
    run_name="cacheblend_evaluator_test",
)


def test_evaluator_requires_independent_transfer_evidence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_cacheblend_correctness.py",
            "--reference",
            "reference.json",
            "--cacheblend",
            "cacheblend.json",
            "--tolerance",
            "tolerance.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--transfer-evidence" in result.stderr


def test_evaluator_documents_explicit_cache_miss_exception() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_cacheblend_correctness.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--allow-cache-miss-no-transfer" in result.stdout


def test_selective_evaluator_does_not_require_transfer_evidence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_cacheblend_correctness.py",
            "--reference",
            "reference.json",
            "--cacheblend",
            "cacheblend.json",
            "--tolerance",
            "tolerance.json",
            "--mode",
            "cacheblend_selective",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 2
    assert "--transfer-evidence is required" not in result.stderr


def test_no_transfer_exception_is_limited_to_zero_transfer_cache_miss() -> None:
    validate = _evaluator["_validate_no_transfer_cache_miss"]
    cache_miss = SimpleNamespace(
        prompt=SimpleNamespace(case=CorrectnessCase.CACHE_MISS),
        connector=SimpleNamespace(
            kv_tokens_found=0,
            kv_tokens_loaded=0,
            kv_tokens_rejected=0,
        ),
    )
    validate(cache_miss)

    moved_hit = SimpleNamespace(
        prompt=SimpleNamespace(case=CorrectnessCase.MOVED_DOCUMENT),
        connector=cache_miss.connector,
    )
    with pytest.raises(ValueError, match="CACHE_MISS"):
        validate(moved_hit)
