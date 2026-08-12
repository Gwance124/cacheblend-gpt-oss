"""CPU-only tests for selective-ratio experiment artifacts."""

from __future__ import annotations

import json

import pytest

from cacheblend_gpt_oss.gpt_oss.selective_policy import (
    CacheBlendSelectionPolicy,
    SelectionMeasurement,
    SelectionPolicyError,
)
from cacheblend_gpt_oss.gpt_oss.selective_policy_io import (
    SELECTION_SWEEP_KIND,
    SelectionSweepIoError,
    SelectionSweepIoErrorCode,
    canonical_selection_sweep_bytes,
    read_selection_sweep,
    selection_sweep_digest,
    selection_sweep_from_dict,
    selection_sweep_to_dict,
    write_selection_sweep,
)
from cacheblend_gpt_oss.planner.models import TokenRange


def _sweep(*, measured: bool = False):
    sweep = CacheBlendSelectionPolicy().sweep(
        prompt_tokens=12,
        cache_ranges=(TokenRange(2, 10),),
        importance_scores=[float(index) for index in range(12)],
        check_layer=1,
        recompute_ratios=(1.0, 0.5, 0.0),
        suffix_tokens=2,
    )
    if not measured:
        return sweep
    return sweep.with_measurements(
        (
            SelectionMeasurement(0.0, 0.0, 0.4),
            SelectionMeasurement(0.1, 0.01, 0.2),
            SelectionMeasurement(0.5, 0.1, 0.1),
        )
    )


def test_round_trip_preserves_work_only_artifact() -> None:
    sweep = _sweep()
    parsed = selection_sweep_from_dict(selection_sweep_to_dict(sweep))
    assert parsed == sweep
    assert parsed.work_curve == sweep.work_curve
    with pytest.raises(SelectionPolicyError) as caught:
        _ = parsed.error_curve
    assert caught.value.code.value == "measurement_mismatch"


def test_round_trip_preserves_explicit_measurements_and_digest() -> None:
    sweep = _sweep(measured=True)
    payload = selection_sweep_to_dict(sweep)
    assert payload["kind"] == SELECTION_SWEEP_KIND
    parsed = selection_sweep_from_dict(payload)
    assert parsed == sweep
    assert parsed.error_curve == sweep.error_curve
    assert parsed.latency_curve == sweep.latency_curve
    assert selection_sweep_digest(parsed) == selection_sweep_digest(sweep)
    assert canonical_selection_sweep_bytes(parsed).endswith(b"}")


def test_writer_is_create_only_and_reader_is_canonical(tmp_path) -> None:
    path = tmp_path / "selection-sweep.json"
    sweep = _sweep(measured=True)
    write_selection_sweep(path, sweep)
    assert read_selection_sweep(path) == sweep
    with pytest.raises(SelectionSweepIoError) as caught:
        write_selection_sweep(path, sweep)
    assert caught.value.code is SelectionSweepIoErrorCode.FILE_EXISTS


def test_mixed_measurements_are_rejected() -> None:
    payload = selection_sweep_to_dict(_sweep())
    assert isinstance(payload["points"], list)
    payload["points"][0]["measurement"] = {
        "max_abs_error": 0.1,
        "mean_abs_error": 0.01,
        "selective_latency_seconds": 0.2,
    }
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is SelectionSweepIoErrorCode.INCOMPLETE_MEASUREMENTS


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("kind", "other", SelectionSweepIoErrorCode.INVALID_SCHEMA),
        ("schema_version", 2, SelectionSweepIoErrorCode.INVALID_SCHEMA),
    ),
)
def test_root_schema_is_exact(field: str, value: object, code) -> None:
    payload = selection_sweep_to_dict(_sweep())
    payload[field] = value
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is code


def test_unknown_point_field_and_range_tampering_fail_closed() -> None:
    payload = selection_sweep_to_dict(_sweep())
    payload["points"][0]["unexpected"] = 1  # type: ignore[index]
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is SelectionSweepIoErrorCode.INVALID_POINT

    payload = selection_sweep_to_dict(_sweep())
    payload["points"][0]["recompute_ranges"] = [[0, 1], [0, 2]]  # type: ignore[index]
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is SelectionSweepIoErrorCode.INVALID_RANGE

    payload = selection_sweep_to_dict(_sweep())
    payload["points"][1]["selected_cached_rows"] = []  # type: ignore[index]
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is SelectionSweepIoErrorCode.POINT_MISMATCH


def test_inconsistent_context_and_invalid_json_are_bounded(tmp_path) -> None:
    payload = selection_sweep_to_dict(_sweep())
    payload["points"][1]["check_layer"] = 2  # type: ignore[index]
    with pytest.raises(SelectionSweepIoError) as caught:
        selection_sweep_from_dict(payload)
    assert caught.value.code is SelectionSweepIoErrorCode.INCONSISTENT_SWEEP

    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SelectionSweepIoError) as caught:
        read_selection_sweep(path)
    assert caught.value.code is SelectionSweepIoErrorCode.INVALID_JSON


def test_artifact_does_not_contain_prompt_or_request_identifiers() -> None:
    rendered = json.dumps(selection_sweep_to_dict(_sweep()), sort_keys=True)
    assert "prompt_tokens" in rendered
    assert "request_id" not in rendered
    assert "token_ids" not in rendered
    assert "fingerprint" not in rendered
