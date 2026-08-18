# SPDX-License-Identifier: Apache-2.0
"""Dependency-free timing values for the pinned worker data plane."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class DataPlanePhaseTiming:
    """Wall-time phases for one fully prepared data-plane copy batch."""

    prepare_latency_seconds: float = 0.0
    enqueue_latency_seconds: float = 0.0
    synchronize_latency_seconds: float = 0.0
    prepared_copy_operations: int = 0
    submitted_copy_operations: int = 0
    input_materialization_latency_seconds: float = 0.0
    span_validation_latency_seconds: float = 0.0
    tensor_validation_latency_seconds: float = 0.0
    range_validation_latency_seconds: float = 0.0
    block_plan_latency_seconds: float = 0.0
    block_index_view_latency_seconds: float = 0.0
    block_index_construction_latency_seconds: float = 0.0
    block_index_validation_latency_seconds: float = 0.0
    staging_view_construction_latency_seconds: float = 0.0
    staging_view_validation_latency_seconds: float = 0.0
    legacy_view_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        latencies = (
            self.prepare_latency_seconds,
            self.enqueue_latency_seconds,
            self.synchronize_latency_seconds,
            self.input_materialization_latency_seconds,
            self.span_validation_latency_seconds,
            self.tensor_validation_latency_seconds,
            self.range_validation_latency_seconds,
            self.block_plan_latency_seconds,
            self.block_index_view_latency_seconds,
            self.block_index_construction_latency_seconds,
            self.block_index_validation_latency_seconds,
            self.staging_view_construction_latency_seconds,
            self.staging_view_validation_latency_seconds,
            self.legacy_view_latency_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0.0
            for value in latencies
        ):
            raise ValueError("invalid data-plane phase latency")
        operation_counts = (
            self.prepared_copy_operations,
            self.submitted_copy_operations,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in operation_counts
        ):
            raise ValueError("invalid data-plane prepared-copy count")
        if self.submitted_copy_operations > self.prepared_copy_operations:
            raise ValueError("invalid data-plane submitted-copy count")
        tolerance = max(1e-9, self.prepare_latency_seconds * 1e-6)
        if (
            self.preparation_subphase_latency_seconds
            > self.prepare_latency_seconds + tolerance
        ):
            raise ValueError("data-plane preparation subphases exceed preparation")
        block_tolerance = max(1e-9, self.block_index_view_latency_seconds * 1e-6)
        if (
            self.block_index_view_subphase_latency_seconds
            > self.block_index_view_latency_seconds + block_tolerance
        ):
            raise ValueError("data-plane block-index/view subphases exceed envelope")

    @property
    def total_latency_seconds(self) -> float:
        return (
            self.prepare_latency_seconds
            + self.enqueue_latency_seconds
            + self.synchronize_latency_seconds
        )

    @property
    def preparation_subphase_latency_seconds(self) -> float:
        """Return nested preparation time without double-counting the envelope."""

        return (
            self.input_materialization_latency_seconds
            + self.span_validation_latency_seconds
            + self.tensor_validation_latency_seconds
            + self.range_validation_latency_seconds
            + self.block_plan_latency_seconds
            + self.block_index_view_latency_seconds
            + self.legacy_view_latency_seconds
        )

    @property
    def block_index_view_subphase_latency_seconds(self) -> float:
        """Return nested index/view time without double-counting its envelope."""

        return (
            self.block_index_construction_latency_seconds
            + self.block_index_validation_latency_seconds
            + self.staging_view_construction_latency_seconds
            + self.staging_view_validation_latency_seconds
        )


__all__ = ["DataPlanePhaseTiming"]
