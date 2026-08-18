"""CPU-only tests for dependency-free worker phase telemetry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cacheblend_gpt_oss.vllm_compat.v0_19_1.stage_timing import (
    DataPlanePhaseTiming,
)


def test_data_plane_phase_timing_is_bounded_and_immutable() -> None:
    timing = DataPlanePhaseTiming(0.1, 0.2, 0.3, 59_904)

    assert timing.total_latency_seconds == pytest.approx(0.6)
    with pytest.raises(FrozenInstanceError):
        timing.prepared_copy_operations = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"prepare_latency_seconds": -0.1},
        {"enqueue_latency_seconds": float("nan")},
        {"synchronize_latency_seconds": True},
        {"prepared_copy_operations": -1},
        {"prepared_copy_operations": True},
    ),
)
def test_data_plane_phase_timing_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="data-plane"):
        DataPlanePhaseTiming(**kwargs)  # type: ignore[arg-type]
