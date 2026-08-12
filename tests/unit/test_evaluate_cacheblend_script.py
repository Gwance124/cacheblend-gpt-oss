"""CPU-only contract tests for the correctness-evaluator CLI."""

from __future__ import annotations

import subprocess
import sys


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
