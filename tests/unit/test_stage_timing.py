"""CPU-only tests for dependency-free worker phase telemetry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cacheblend_gpt_oss.vllm_compat.v0_19_1.stage_timing import (
    DataPlanePhaseTiming,
)


def test_data_plane_phase_timing_is_bounded_and_immutable() -> None:
    timing = DataPlanePhaseTiming(
        0.1,
        0.2,
        0.3,
        59_904,
        96,
        input_materialization_latency_seconds=0.001,
        span_validation_latency_seconds=0.02,
        tensor_validation_latency_seconds=0.03,
        range_validation_latency_seconds=0.004,
        block_plan_latency_seconds=0.01,
        block_index_view_latency_seconds=0.02,
        legacy_view_latency_seconds=0.0,
    )

    assert timing.total_latency_seconds == pytest.approx(0.6)
    assert timing.preparation_subphase_latency_seconds == pytest.approx(0.085)
    assert timing.submitted_copy_operations == 96
    with pytest.raises(FrozenInstanceError):
        timing.prepared_copy_operations = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"prepare_latency_seconds": -0.1},
        {"enqueue_latency_seconds": float("nan")},
        {"synchronize_latency_seconds": True},
        {"span_validation_latency_seconds": -0.1},
        {"prepared_copy_operations": -1},
        {"prepared_copy_operations": True},
        {"submitted_copy_operations": -1},
        {"submitted_copy_operations": True},
        {
            "prepared_copy_operations": 1,
            "submitted_copy_operations": 2,
        },
        {
            "prepare_latency_seconds": 0.1,
            "span_validation_latency_seconds": 0.11,
        },
    ),
)
def test_data_plane_phase_timing_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="data-plane"):
        DataPlanePhaseTiming(**kwargs)  # type: ignore[arg-type]
