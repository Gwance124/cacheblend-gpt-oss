import pytest

from cacheblend_gpt_oss.planner import (
    DelimiterSegmenter,
    FixedChunkStorageSegmenter,
    RollingQuerySegmenter,
    TokenSegment,
)


def test_delimiters_are_excluded_and_absolute_positions_are_preserved() -> None:
    segmenter = DelimiterSegmenter([[99], [99, 100]])

    segments = segmenter.segment(
        [99, 100, 1, 2, 99, 3, 4, 99, 100],
        offset=7,
    )

    assert [(segment.token_range.start, segment.token_ids) for segment in segments] == [
        (9, (1, 2)),
        (12, (3, 4)),
    ]


def test_fixed_storage_chunks_never_cross_delimited_regions() -> None:
    regions = DelimiterSegmenter([[0]]).segment([1, 2, 3, 4, 5, 0, 6, 7, 8])

    chunks = FixedChunkStorageSegmenter(
        chunk_size=2,
        include_partial=True,
    ).segment(regions)

    assert [(chunk.token_range.start, chunk.token_ids) for chunk in chunks] == [
        (0, (1, 2)),
        (2, (3, 4)),
        (4, (5,)),
        (6, (6, 7)),
        (8, (8,)),
    ]


def test_rolling_queries_find_chunks_at_unaligned_positions() -> None:
    region = TokenSegment.at(20, [8, 9, 1, 2, 3, 4, 7])

    candidates = RollingQuerySegmenter([4]).segment([region])

    assert [(item.token_range.start, item.token_ids) for item in candidates] == [
        (20, (8, 9, 1, 2)),
        (21, (9, 1, 2, 3)),
        (22, (1, 2, 3, 4)),
        (23, (2, 3, 4, 7)),
    ]


def test_invalid_segmentation_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        DelimiterSegmenter([])
    with pytest.raises(ValueError):
        FixedChunkStorageSegmenter(0)
    with pytest.raises(ValueError):
        RollingQuerySegmenter([])

