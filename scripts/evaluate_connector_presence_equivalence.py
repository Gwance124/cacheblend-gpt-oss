#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a deterministic long-context connector-presence A/B/A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
RESPONSES_CONTRACT = "cacheblend-gpt-oss-hybrid-flag-responses-v1"
MODEL_ID = "openai/gpt-oss-20b"
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0}
FILLER_UNITS = [" alpha", " beta", " gamma"]
FILLER_REPETITIONS_PER_TURN = 20_000
WARMUP_PAYLOAD = {
    "model": MODEL_ID,
    "prompt": "Isolated connector-presence warmup.",
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
}
_CONNECTOR_COUNTER_KEYS = frozenset(
    {
        "requests",
        "reusable_document_tokens_requested",
        "kv_tokens_found",
        "kv_tokens_loaded",
        "kv_tokens_rejected",
        "tokens_recomputed",
        "prefill_tokens_avoided",
    }
)
_STORE_COUNTER_KEYS = frozenset(
    {"store_tokens_eligible", "store_tokens_completed", "store_fallbacks"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-a", type=Path, required=True)
    parser.add_argument("--baseline-b", type=Path, required=True)
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--latency-ratio-limit", type=float, default=2.0)
    parser.add_argument("--minimum-final-input-tokens", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _positive_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"invalid {name}")
    return float(value)


def _counter_mapping(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"invalid {name} schema")
    return {
        key: _nonnegative_integer(value.get(key), f"{name} {key}")
        for key in expected_keys
    }


def _read_responses(
    path: Path,
    *,
    expected_mode: str,
    connector_expected: bool,
) -> dict[str, Any]:
    artifact = _read_json(path)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("contract") != RESPONSES_CONTRACT
        or artifact.get("mode") != expected_mode
        or artifact.get("model") != MODEL_ID
        or artifact.get("sampling") != SAMPLING
        or artifact.get("stable_replay_ids") is not True
    ):
        raise ValueError(f"invalid connector-presence Responses artifact: {path}")

    workload = artifact.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("filler_units_per_turn") != FILLER_UNITS
        or workload.get("filler_repetitions_per_turn") != FILLER_REPETITIONS_PER_TURN
    ):
        raise ValueError(f"invalid long-context workload identity: {path}")
    warmup = artifact.get("warmup")
    if (
        not isinstance(warmup, dict)
        or set(warmup)
        != {
            "performed",
            "request_digest",
            "connector_counters",
            "connector_store_counters",
        }
        or warmup.get("performed") is not True
        or warmup.get("request_digest") != _digest(WARMUP_PAYLOAD)
    ):
        raise ValueError(f"invalid warmup identity: {path}")
    if connector_expected:
        warmup_connector = _counter_mapping(
            warmup.get("connector_counters"),
            _CONNECTOR_COUNTER_KEYS,
            "warmup connector counters",
        )
        warmup_stores = _counter_mapping(
            warmup.get("connector_store_counters"),
            _STORE_COUNTER_KEYS,
            "warmup connector store counters",
        )
        if (
            warmup_connector["requests"] != 1
            or warmup_connector["reusable_document_tokens_requested"] != 0
            or warmup_connector["kv_tokens_found"] != 0
            or warmup_connector["kv_tokens_loaded"] != 0
            or warmup_connector["kv_tokens_rejected"] != 0
            or warmup_connector["tokens_recomputed"] <= 0
            or warmup_connector["prefill_tokens_avoided"] != 0
            or any(warmup_stores.values())
        ):
            raise ValueError(f"warmup connector counters do not reconcile: {path}")
    elif (
        warmup.get("connector_counters") is not None
        or warmup.get("connector_store_counters") is not None
    ):
        raise ValueError(f"baseline warmup contains connector counters: {path}")

    turns = artifact.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        raise ValueError(f"Responses artifact must contain three turns: {path}")
    input_tokens: list[int] = []
    cached_tokens: list[int] = []
    elapsed_seconds: list[float] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"invalid Responses turn: {path}")
        if turn.get("output_digest") != _digest(turn.get("canonical_output")):
            raise ValueError(f"Responses output digest mismatch: {path}")
        request_digest = turn.get("request_digest")
        if (
            not isinstance(request_digest, str)
            or len(request_digest) != 64
            or any(character not in "0123456789abcdef" for character in request_digest)
        ):
            raise ValueError(f"invalid Responses request digest: {path}")
        usage = turn.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"invalid Responses usage: {path}")
        counts = {
            key: _nonnegative_integer(usage.get(key), f"usage {key}")
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
                "tool_output_tokens",
            )
        }
        if (
            counts["total_tokens"] != counts["input_tokens"] + counts["output_tokens"]
            or counts["cached_tokens"] > counts["input_tokens"]
        ):
            raise ValueError(f"Responses usage does not reconcile: {path}")
        input_tokens.append(counts["input_tokens"])
        cached_tokens.append(counts["cached_tokens"])
        elapsed_seconds.append(
            _positive_number(turn.get("elapsed_seconds"), "turn latency")
        )

    if not input_tokens[0] < input_tokens[1] < input_tokens[2]:
        raise ValueError(f"long-context input tokens are not increasing: {path}")
    if workload.get("input_tokens_per_turn") != input_tokens:
        raise ValueError(f"workload token counts do not reconcile: {path}")
    prefix_cache = artifact.get("prefix_cache")
    reuse_observed = any(value > 0 for value in cached_tokens[1:])
    if (
        not isinstance(prefix_cache, dict)
        or prefix_cache.get("cached_tokens_per_turn") != cached_tokens
        or prefix_cache.get("reuse_observed_after_cold_turn") is not reuse_observed
    ):
        raise ValueError(f"invalid prefix-cache evidence: {path}")
    total_elapsed = _positive_number(
        artifact.get("total_elapsed_seconds"), "total latency"
    )
    if not math.isclose(
        total_elapsed,
        sum(elapsed_seconds),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Responses latency total does not reconcile: {path}")

    if connector_expected:
        connector = _counter_mapping(
            artifact.get("connector_counters"),
            _CONNECTOR_COUNTER_KEYS,
            "connector counters",
        )
        stores = _counter_mapping(
            artifact.get("connector_store_counters"),
            _STORE_COUNTER_KEYS,
            "connector store counters",
        )
        if (
            connector["requests"] != 3
            or connector["kv_tokens_found"]
            > connector["reusable_document_tokens_requested"]
            or connector["kv_tokens_loaded"] != 0
            or connector["prefill_tokens_avoided"] != 0
            or connector["kv_tokens_found"]
            != connector["kv_tokens_loaded"] + connector["kv_tokens_rejected"]
            or connector["tokens_recomputed"] <= 0
            or stores["store_fallbacks"] != 0
            or stores["store_tokens_completed"] > stores["store_tokens_eligible"]
        ):
            raise ValueError(f"connector counters do not reconcile: {path}")
    elif (
        artifact.get("connector_counters") is not None
        or artifact.get("connector_store_counters") is not None
    ):
        raise ValueError(f"baseline unexpectedly contains connector counters: {path}")
    return artifact


def _signature(artifact: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "request_digest": turn["request_digest"],
            "output_digest": turn["output_digest"],
            "usage": turn["usage"],
        }
        for turn in artifact["turns"]
    ]


def _turn_latencies(artifact: dict[str, Any]) -> list[float]:
    return [float(turn["elapsed_seconds"]) for turn in artifact["turns"]]


def evaluate(
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
    connector: dict[str, Any],
    *,
    latency_ratio_limit: float,
    minimum_final_input_tokens: int,
) -> tuple[dict[str, object], int]:
    if not math.isfinite(latency_ratio_limit) or latency_ratio_limit <= 1:
        raise ValueError("latency ratio limit must be finite and greater than one")
    if minimum_final_input_tokens <= 0:
        raise ValueError("minimum final input tokens must be positive")

    baseline_a_signature = _signature(baseline_a)
    baseline_b_signature = _signature(baseline_b)
    connector_signature = _signature(connector)
    baseline_outputs_stable = baseline_a_signature == baseline_b_signature
    connector_outputs_match = (
        connector_signature == baseline_a_signature == baseline_b_signature
    )
    prefix_reuse_all = all(
        artifact["prefix_cache"].get("reuse_observed_after_cold_turn") is True
        for artifact in (baseline_a, baseline_b, connector)
    )
    final_input_tokens = [
        int(artifact["workload"]["input_tokens_per_turn"][-1])
        for artifact in (baseline_a, baseline_b, connector)
    ]
    long_context_reached = min(final_input_tokens) >= minimum_final_input_tokens

    baseline_a_turns = _turn_latencies(baseline_a)
    baseline_b_turns = _turn_latencies(baseline_b)
    connector_turns = _turn_latencies(connector)
    baseline_turn_means = [
        (left + right) / 2
        for left, right in zip(baseline_a_turns, baseline_b_turns, strict=True)
    ]
    baseline_turn_spreads = [
        max(left, right) / min(left, right)
        for left, right in zip(baseline_a_turns, baseline_b_turns, strict=True)
    ]
    connector_turn_ratios = [
        candidate / baseline
        for candidate, baseline in zip(
            connector_turns,
            baseline_turn_means,
            strict=True,
        )
    ]
    baseline_totals = [
        float(baseline_a["total_elapsed_seconds"]),
        float(baseline_b["total_elapsed_seconds"]),
    ]
    baseline_total_mean = sum(baseline_totals) / 2
    baseline_total_spread = max(baseline_totals) / min(baseline_totals)
    connector_total = float(connector["total_elapsed_seconds"])
    connector_total_ratio = connector_total / baseline_total_mean
    baseline_latency_stable = (
        baseline_total_spread <= latency_ratio_limit
        and max(baseline_turn_spreads) <= latency_ratio_limit
    )
    connector_latency_within_limit = (
        connector_total_ratio <= latency_ratio_limit
        and max(connector_turn_ratios) <= latency_ratio_limit
    )

    if not baseline_outputs_stable:
        status = "INCONCLUSIVE_BASELINE_OUTPUT_UNSTABLE"
        exit_status = 2
    elif not long_context_reached:
        status = "FAIL_LONG_CONTEXT_NOT_REACHED"
        exit_status = 1
    elif not prefix_reuse_all:
        status = "FAIL_PREFIX_REUSE_MISSING"
        exit_status = 1
    elif not connector_outputs_match:
        status = "FAIL_CONNECTOR_OUTPUT_DIVERGED"
        exit_status = 1
    elif not baseline_latency_stable:
        status = "INCONCLUSIVE_BASELINE_LATENCY_UNSTABLE"
        exit_status = 2
    elif not connector_latency_within_limit:
        status = "FAIL_CONNECTOR_LATENCY_DIVERGED"
        exit_status = 1
    else:
        status = "PASS_CONNECTOR_PRESENCE_WITHIN_LIMIT"
        exit_status = 0

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": status,
        "passed": exit_status == 0,
        "baseline_outputs_stable": baseline_outputs_stable,
        "connector_outputs_match": connector_outputs_match,
        "prefix_reuse_all": prefix_reuse_all,
        "long_context_reached": long_context_reached,
        "final_input_tokens": {
            "baseline_a": final_input_tokens[0],
            "baseline_b": final_input_tokens[1],
            "connector": final_input_tokens[2],
            "minimum": minimum_final_input_tokens,
        },
        "latency": {
            "ratio_limit": latency_ratio_limit,
            "baseline_total_seconds": baseline_totals,
            "baseline_total_mean_seconds": baseline_total_mean,
            "baseline_total_spread_ratio": baseline_total_spread,
            "connector_total_seconds": connector_total,
            "connector_total_ratio": connector_total_ratio,
            "baseline_turn_seconds": [baseline_a_turns, baseline_b_turns],
            "baseline_turn_mean_seconds": baseline_turn_means,
            "baseline_turn_spread_ratios": baseline_turn_spreads,
            "connector_turn_seconds": connector_turns,
            "connector_turn_ratios": connector_turn_ratios,
            "baseline_stable": baseline_latency_stable,
            "connector_within_limit": connector_latency_within_limit,
        },
        "connector_warmup": connector["warmup"],
        "connector_counters": connector["connector_counters"],
        "connector_store_counters": connector["connector_store_counters"],
        "response_signatures": {
            "baseline_a": baseline_a_signature,
            "baseline_b": baseline_b_signature,
            "connector": connector_signature,
        },
    }
    return report, exit_status


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("connector-presence verdict output already exists")
    report, exit_status = evaluate(
        _read_responses(
            args.baseline_a,
            expected_mode="baseline",
            connector_expected=False,
        ),
        _read_responses(
            args.baseline_b,
            expected_mode="baseline",
            connector_expected=False,
        ),
        _read_responses(
            args.connector,
            expected_mode="connector",
            connector_expected=True,
        ),
        latency_ratio_limit=args.latency_ratio_limit,
        minimum_final_input_tokens=args.minimum_final_input_tokens,
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
