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

    def __post_init__(self) -> None:
        latencies = (
            self.prepare_latency_seconds,
            self.enqueue_latency_seconds,
            self.synchronize_latency_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0.0
            for value in latencies
        ):
            raise ValueError("invalid data-plane phase latency")
        if (
            isinstance(self.prepared_copy_operations, bool)
            or not isinstance(self.prepared_copy_operations, int)
            or self.prepared_copy_operations < 0
        ):
            raise ValueError("invalid data-plane prepared-copy count")

    @property
    def total_latency_seconds(self) -> float:
        return (
            self.prepare_latency_seconds
            + self.enqueue_latency_seconds
            + self.synchronize_latency_seconds
        )


__all__ = ["DataPlanePhaseTiming"]
