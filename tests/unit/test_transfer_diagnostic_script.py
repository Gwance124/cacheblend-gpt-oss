"""CPU-only tests for the solab-g3 transfer diagnostic report."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("local-m85-check-transfer-diag.sh").resolve()


def _run_diagnostic(
    tmp_path: Path,
    log_lines: list[str],
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "vllm-server.log").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _lookup(request: str, prefix_cached_tokens: int) -> str:
    return (
        "CACHEBLEND_TRANSFER_DIAG lookup "
        f"request={request} prompt_tokens=1024 "
        f"prefix_cached_tokens={prefix_cached_tokens} "
        "lookup_status=miss should_transfer=False verified_candidates=0"
    )


def test_reports_failed_gate_when_every_prefix_count_is_zero(tmp_path: Path) -> None:
    result = _run_diagnostic(
        tmp_path,
        [
            _lookup("request-1", 0),
            "CACHEBLEND_TRANSFER_DIAG alloc request=request-1 num_external_tokens=0",
            _lookup("request-2", 0),
            "CACHEBLEND_DECODE_DIAG finished=2 decode_steps=15",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "PREFIX_CACHED_ZERO_EVENTS=2" in result.stdout
    assert "PREFIX_CACHED_NONZERO_EVENTS=0" in result.stdout
    assert "PREFIX_CACHE_GATE=FAIL_ALL_LOOKUPS_ZERO" in result.stdout
    report = (tmp_path / "prefix-cache-diagnostic.txt").read_text(encoding="utf-8")
    assert "PREFIX_CACHE_GATE=FAIL_ALL_LOOKUPS_ZERO" in report


def test_reports_passed_gate_when_any_prefix_count_is_nonzero(tmp_path: Path) -> None:
    result = _run_diagnostic(
        tmp_path,
        [
            _lookup("request-1", 0),
            _lookup("request-2", 768),
            "Prefix cache hit rate: 50.0%",
            "CACHEBLEND_DECODE_DIAG finished=2 decode_steps=15",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "PREFIX_CACHED_MIN=0" in result.stdout
    assert "PREFIX_CACHED_MAX=768" in result.stdout
    assert "PREFIX_CACHED_SUM=768" in result.stdout
    assert "PREFIX_CACHE_GATE=PASS_NONZERO_REUSE_OBSERVED" in result.stdout


def test_excerpt_limit_does_not_abort_large_diagnostic_log(tmp_path: Path) -> None:
    result = _run_diagnostic(
        tmp_path,
        [_lookup(f"request-{index}", 0) for index in range(25)],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("CACHEBLEND_TRANSFER_DIAG lookup") == 20
    assert "LOOKUP_EVENTS=25" in result.stdout
    assert "PREFIX_CACHE_GATE=FAIL_ALL_LOOKUPS_ZERO" in result.stdout


@pytest.mark.parametrize(
    ("log_lines", "expected_gate"),
    [
        (
            ["CACHEBLEND_DECODE_DIAG finished=1 decode_steps=5"],
            "INCONCLUSIVE_NO_LOOKUP_DIAGNOSTICS",
        ),
        (
            ["CACHEBLEND_TRANSFER_DIAG lookup request=request-1"],
            "INCONCLUSIVE_MALFORMED_LOOKUP_DIAGNOSTICS",
        ),
    ],
)
def test_returns_two_for_inconclusive_diagnostics(
    tmp_path: Path,
    log_lines: list[str],
    expected_gate: str,
) -> None:
    result = _run_diagnostic(tmp_path, log_lines)

    assert result.returncode == 2
    assert f"PREFIX_CACHE_GATE={expected_gate}" in result.stdout
    report = (tmp_path / "prefix-cache-diagnostic.txt").read_text(encoding="utf-8")
    assert f"PREFIX_CACHE_GATE={expected_gate}" in report
