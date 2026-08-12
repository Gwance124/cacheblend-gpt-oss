"""Deterministic storage and query segmentation strategies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from cacheblend_gpt_oss.planner.models import (
    TokenRange,
    TokenSegment,
    normalize_token_ids,
)


@dataclass(frozen=True, slots=True, init=False)
class DelimiterSegmenter:
    """Split document regions at exact token delimiters, excluding delimiters."""

    delimiters: tuple[tuple[int, ...], ...]

    def __init__(self, delimiters: Iterable[Sequence[int]]) -> None:
        normalized = tuple(normalize_token_ids(delimiter) for delimiter in delimiters)
        if not normalized:
            raise ValueError("at least one delimiter is required")
        if any(not delimiter for delimiter in normalized):
            raise ValueError("delimiters must not be empty")
        # Longest-first resolves a shared-prefix delimiter deterministically.
        ordered = tuple(sorted(set(normalized), key=lambda item: (-len(item), item)))
        object.__setattr__(self, "delimiters", ordered)

    def segment(
        self, token_ids: Iterable[int], *, offset: int = 0
    ) -> tuple[TokenSegment, ...]:
        """Return non-empty regions in absolute request coordinates."""

        tokens = normalize_token_ids(token_ids)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        segments: list[TokenSegment] = []
        region_start = 0
        cursor = 0
        while cursor < len(tokens):
            delimiter = next(
                (
                    candidate
                    for candidate in self.delimiters
                    if tokens[cursor : cursor + len(candidate)] == candidate
                ),
                None,
            )
            if delimiter is None:
                cursor += 1
                continue
            if cursor > region_start:
                segments.append(
                    TokenSegment.at(offset + region_start, tokens[region_start:cursor])
                )
            cursor += len(delimiter)
            region_start = cursor

        if region_start < len(tokens):
            segments.append(
                TokenSegment.at(offset + region_start, tokens[region_start:])
            )
        return tuple(segments)


@dataclass(frozen=True, slots=True)
class FixedChunkStorageSegmenter:
    """Create chunks aligned to the beginning of each delimited document region."""

    chunk_size: int
    include_partial: bool = True
    minimum_partial_size: int = 1

    def __post_init__(self) -> None:
        for field_name in ("chunk_size", "minimum_partial_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.minimum_partial_size > self.chunk_size:
            raise ValueError("minimum_partial_size must not exceed chunk_size")

    def segment(self, regions: Iterable[TokenSegment]) -> tuple[TokenSegment, ...]:
        chunks: list[TokenSegment] = []
        for region in regions:
            for relative_start in range(0, len(region), self.chunk_size):
                token_slice = region.token_ids[
                    relative_start : relative_start + self.chunk_size
                ]
                is_partial = len(token_slice) < self.chunk_size
                if is_partial and (
                    not self.include_partial
                    or len(token_slice) < self.minimum_partial_size
                ):
                    continue
                chunks.append(
                    TokenSegment.at(
                        region.token_range.start + relative_start,
                        token_slice,
                    )
                )
        return tuple(chunks)


@dataclass(frozen=True, slots=True, init=False)
class RollingQuerySegmenter:
    """Generate candidate windows without assuming storage-chunk alignment."""

    window_sizes: tuple[int, ...]
    stride: int

    def __init__(self, window_sizes: Iterable[int], *, stride: int = 1) -> None:
        sizes = tuple(window_sizes)
        if not sizes:
            raise ValueError("at least one window size is required")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError("window sizes must be integers")
            if size < 1:
                raise ValueError("window sizes must be positive")
        if isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be an integer")
        if stride < 1:
            raise ValueError("stride must be positive")
        object.__setattr__(
            self, "window_sizes", tuple(sorted(set(sizes), reverse=True))
        )
        object.__setattr__(self, "stride", stride)

    def segment(self, regions: Iterable[TokenSegment]) -> tuple[TokenSegment, ...]:
        candidates: list[TokenSegment] = []
        for region in regions:
            for relative_start in range(0, len(region), self.stride):
                for window_size in self.window_sizes:
                    relative_end = relative_start + window_size
                    if relative_end > len(region):
                        continue
                    candidates.append(
                        TokenSegment(
                            TokenRange(
                                region.token_range.start + relative_start,
                                region.token_range.start + relative_end,
                            ),
                            region.token_ids[relative_start:relative_end],
                        )
                    )
        return tuple(candidates)
