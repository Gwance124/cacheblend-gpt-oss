from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.gpt_oss import AttentionKind
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import (
    VllmAdapterError,
    VllmAdapterErrorCode,
    adapt_kv_cache_blocks,
    adapt_kv_cache_config,
    copy_request_prompt_token_ids,
)


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


class FullAttentionSpec:
    def __init__(
        self,
        *,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_size: int = 64,
        head_size_v: int = 64,
        sliding_window: int | None = None,
        attention_chunk_size: int | None = None,
    ) -> None:
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.sliding_window = sliding_window
        self.attention_chunk_size = attention_chunk_size


class SlidingWindowSpec:
    def __init__(
        self,
        *,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_size: int = 64,
        sliding_window: int = 128,
    ) -> None:
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.sliding_window = sliding_window


def _kv_cache_config(*, num_blocks: int = 100) -> SimpleNamespace:
    # Reverse layer order and put the full group first to prove that the
    # adapter canonicalizes membership without changing outer group identity.
    full = SimpleNamespace(
        layer_names=[_layer_name(index) for index in range(23, 0, -2)],
        kv_cache_spec=FullAttentionSpec(),
    )
    sliding = SimpleNamespace(
        layer_names=[_layer_name(index) for index in range(22, -1, -2)],
        kv_cache_spec=SlidingWindowSpec(),
    )
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[full, sliding],
    )


def _assert_error(
    expected: VllmAdapterErrorCode,
    operation: Callable[[], object],
) -> VllmAdapterError:
    with pytest.raises(VllmAdapterError) as caught:
        operation()
    assert caught.value.code is expected
    assert str(caught.value).endswith(expected.value)
    return caught.value


def test_finalized_config_translates_to_both_immutable_layouts() -> None:
    adapted = adapt_kv_cache_config(_kv_cache_config())

    assert adapted.num_blocks == 100
    assert [group.group_id for group in adapted.gpt_oss_layout.groups] == [0, 1]
    assert [
        group.attention_kind for group in adapted.gpt_oss_layout.groups
    ] == [AttentionKind.FULL, AttentionKind.SLIDING]
    assert adapted.gpt_oss_layout.group(0).layer_names == tuple(
        _layer_name(index) for index in range(1, 24, 2)
    )
    assert adapted.gpt_oss_layout.group(1).layer_names == tuple(
        _layer_name(index) for index in range(0, 24, 2)
    )
    assert adapted.gpt_oss_layout.group(0).block_size == 16
    assert adapted.gpt_oss_layout.group(0).sliding_window is None
    assert adapted.gpt_oss_layout.group(1).sliding_window == 128
    assert adapted.control_plane_layout.layer_names_by_group == (
        adapted.gpt_oss_layout.group(0).layer_names,
        adapted.gpt_oss_layout.group(1).layer_names,
    )
    assert [layer.layer_index for layer in adapted.gpt_oss_layout.layers] == list(
        range(24)
    )
    with pytest.raises(FrozenInstanceError):
        adapted.num_blocks = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec, "block_size", 32
            ),
            VllmAdapterErrorCode.BLOCK_SIZE_MISMATCH,
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec, "num_kv_heads", 4
            ),
            VllmAdapterErrorCode.KV_HEAD_COUNT_MISMATCH,
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec, "head_size", 128
            ),
            VllmAdapterErrorCode.HEAD_SIZE_MISMATCH,
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec, "head_size_v", 32
            ),
            VllmAdapterErrorCode.VALUE_HEAD_SIZE_MISMATCH,
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[1].kv_cache_spec,
                "sliding_window",
                256,
            ),
            VllmAdapterErrorCode.SLIDING_WINDOW_MISMATCH,
        ),
        (
            lambda config: setattr(
                config.kv_cache_groups[0].kv_cache_spec,
                "attention_chunk_size",
                128,
            ),
            VllmAdapterErrorCode.ATTENTION_CHUNKING_UNSUPPORTED,
        ),
    ],
)
def test_config_rejects_wrong_block_and_head_shapes(
    mutate: Callable[[SimpleNamespace], None],
    expected: VllmAdapterErrorCode,
) -> None:
    config = _kv_cache_config()
    mutate(config)
    _assert_error(expected, lambda: adapt_kv_cache_config(config))


def test_config_rejects_wrong_spec_type_group_count_and_num_blocks() -> None:
    config = _kv_cache_config()
    config.kv_cache_groups[0].kv_cache_spec = SimpleNamespace(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
    )
    _assert_error(
        VllmAdapterErrorCode.SPEC_TYPE_MISMATCH,
        lambda: adapt_kv_cache_config(config),
    )

    # 1 group is now accepted (unified mode), but only if all 24 layers are
    # present.  Popping a group leaves 12 layers → INVALID_LAYER_NAMES.
    one_group = _kv_cache_config()
    one_group.kv_cache_groups.pop()
    _assert_error(
        VllmAdapterErrorCode.INVALID_LAYER_NAMES,
        lambda: adapt_kv_cache_config(one_group),
    )
    _assert_error(
        VllmAdapterErrorCode.INVALID_NUM_BLOCKS,
        lambda: adapt_kv_cache_config(_kv_cache_config(num_blocks=0)),
    )
    _assert_error(
        VllmAdapterErrorCode.INVALID_NUM_BLOCKS,
        lambda: adapt_kv_cache_config(_kv_cache_config(num_blocks=True)),
    )


def test_config_rejects_missing_noncanonical_duplicate_and_misgrouped_layers() -> (
    None
):
    missing = _kv_cache_config()
    missing.kv_cache_groups[1].layer_names.pop()
    _assert_error(
        VllmAdapterErrorCode.INVALID_LAYER_NAMES,
        lambda: adapt_kv_cache_config(missing),
    )

    noncanonical = _kv_cache_config()
    noncanonical.kv_cache_groups[1].layer_names[0] = "model.layers.22.attn"
    _assert_error(
        VllmAdapterErrorCode.INVALID_LAYER_NAMES,
        lambda: adapt_kv_cache_config(noncanonical),
    )

    duplicate = _kv_cache_config()
    duplicate.kv_cache_groups[1].layer_names[0] = _layer_name(0)
    duplicate.kv_cache_groups[1].layer_names[1] = _layer_name(0)
    _assert_error(
        VllmAdapterErrorCode.INVALID_LAYER_NAMES,
        lambda: adapt_kv_cache_config(duplicate),
    )

    wrong_parity = _kv_cache_config()
    wrong_parity.kv_cache_groups[0].layer_names[0] = _layer_name(22)
    wrong_parity.kv_cache_groups[1].layer_names[0] = _layer_name(23)
    _assert_error(
        VllmAdapterErrorCode.INVALID_LAYER_NAMES,
        lambda: adapt_kv_cache_config(wrong_parity),
    )


def test_prompt_ids_are_copied_and_prompt_embeds_fail_closed() -> None:
    source_ids = [1, 2, 3, 4]
    request = SimpleNamespace(
        prompt_token_ids=source_ids,
        prompt_embeds=None,
        num_prompt_tokens=4,
    )

    copied = copy_request_prompt_token_ids(request)
    source_ids[0] = 99
    assert copied == (1, 2, 3, 4)

    request.prompt_embeds = []
    _assert_error(
        VllmAdapterErrorCode.PROMPT_EMBEDS_UNSUPPORTED,
        lambda: copy_request_prompt_token_ids(request),
    )


def test_prompt_ids_reject_missing_invalid_and_length_mismatch() -> None:
    _assert_error(
        VllmAdapterErrorCode.PROMPT_TOKEN_IDS_MISSING,
        lambda: copy_request_prompt_token_ids(
            SimpleNamespace(prompt_embeds=None, num_prompt_tokens=0)
        ),
    )
    _assert_error(
        VllmAdapterErrorCode.PROMPT_TOKEN_IDS_MISSING,
        lambda: copy_request_prompt_token_ids(
            SimpleNamespace(
                prompt_token_ids=None,
                prompt_embeds=None,
                num_prompt_tokens=0,
            )
        ),
    )
    _assert_error(
        VllmAdapterErrorCode.INVALID_PROMPT_TOKEN_IDS,
        lambda: copy_request_prompt_token_ids(
            SimpleNamespace(
                prompt_token_ids=(1, 2),
                prompt_embeds=None,
                num_prompt_tokens=2,
            )
        ),
    )
    _assert_error(
        VllmAdapterErrorCode.INVALID_PROMPT_TOKEN_IDS,
        lambda: copy_request_prompt_token_ids(
            SimpleNamespace(
                prompt_token_ids=[1, True],
                prompt_embeds=None,
                num_prompt_tokens=2,
            )
        ),
    )
    _assert_error(
        VllmAdapterErrorCode.PROMPT_TOKEN_COUNT_MISMATCH,
        lambda: copy_request_prompt_token_ids(
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                prompt_embeds=None,
                num_prompt_tokens=3,
            )
        ),
    )


@dataclass
class FakeBlock:
    block_id: int
    is_null: bool = False


class FakeKVCacheBlocks:
    def __init__(self, block_ids_by_group: tuple[tuple[int, ...], ...]) -> None:
        self.blocks = tuple(
            [FakeBlock(block_id) for block_id in group]
            for group in block_ids_by_group
        )
        self.get_block_ids_calls = 0

    def get_block_ids(self) -> tuple[list[int], ...]:
        self.get_block_ids_calls += 1
        return tuple(
            [block.block_id for block in group] for group in self.blocks
        )


def test_grouped_blocks_translate_to_control_and_scatter_tables() -> None:
    config = adapt_kv_cache_config(_kv_cache_config())
    source = FakeKVCacheBlocks(((7, 8), (11, 12, 13)))

    adapted = adapt_kv_cache_blocks(source, config)
    source.blocks[0][0].block_id = 99

    assert source.get_block_ids_calls == 1
    assert adapted.block_ids_by_group == ((7, 8), (11, 12, 13))
    assert adapted.grouped_allocation.group_layout == config.control_plane_layout
    assert [table.group_id for table in adapted.group_block_tables] == [0, 1]
    assert [table.block_size for table in adapted.group_block_tables] == [16, 16]
    assert [table.block_ids for table in adapted.group_block_tables] == [
        (7, 8),
        (11, 12, 13),
    ]
    with pytest.raises(FrozenInstanceError):
        adapted.group_block_tables = ()  # type: ignore[misc]


def test_null_block_is_rejected_before_it_becomes_a_writable_table() -> None:
    config = adapt_kv_cache_config(_kv_cache_config())
    blocks = FakeKVCacheBlocks(((7,), (11, 12)))
    blocks.blocks[1][0].is_null = True

    _assert_error(
        VllmAdapterErrorCode.NULL_BLOCK_UNSUPPORTED,
        lambda: adapt_kv_cache_blocks(blocks, config),
    )

    repeated_null = FakeKVCacheBlocks(((7,), (11, 11)))
    repeated_null.blocks[1][0].is_null = True
    _assert_error(
        VllmAdapterErrorCode.NULL_BLOCK_UNSUPPORTED,
        lambda: adapt_kv_cache_blocks(repeated_null, config),
    )


@pytest.mark.parametrize(
    ("block_ids", "expected"),
    [
        (((1,),), VllmAdapterErrorCode.BLOCK_GROUP_COUNT_MISMATCH),
        (((1,), (-1,)), VllmAdapterErrorCode.INVALID_BLOCK_IDS),
        (((1,), (True,)), VllmAdapterErrorCode.INVALID_BLOCK_IDS),
        (((1,), (2, 2)), VllmAdapterErrorCode.INVALID_BLOCK_IDS),
        (((1,), (100,)), VllmAdapterErrorCode.BLOCK_ID_OUT_OF_RANGE),
    ],
)
def test_invalid_grouped_block_ids_fail_closed(
    block_ids: tuple[tuple[int, ...], ...],
    expected: VllmAdapterErrorCode,
) -> None:
    config = adapt_kv_cache_config(_kv_cache_config())
    _assert_error(
        expected,
        lambda: adapt_kv_cache_blocks(FakeKVCacheBlocks(block_ids), config),
    )


def test_missing_or_failed_get_block_ids_fails_closed() -> None:
    config = adapt_kv_cache_config(_kv_cache_config())
    _assert_error(
        VllmAdapterErrorCode.BLOCK_ID_METHOD_MISSING,
        lambda: adapt_kv_cache_blocks(SimpleNamespace(), config),
    )

    class BrokenBlocks:
        def get_block_ids(self) -> tuple[list[int], ...]:
            raise RuntimeError("unbounded upstream detail")

    error = _assert_error(
        VllmAdapterErrorCode.BLOCK_ID_EXTRACTION_FAILED,
        lambda: adapt_kv_cache_blocks(BrokenBlocks(), config),
    )
    assert "unbounded upstream detail" not in str(error)


def test_raw_blocks_must_agree_with_get_block_ids() -> None:
    config = adapt_kv_cache_config(_kv_cache_config())
    blocks = FakeKVCacheBlocks(((7,), (11,)))

    def inconsistent_ids() -> tuple[list[int], ...]:
        return ([8], [11])

    blocks.get_block_ids = inconsistent_ids  # type: ignore[method-assign]
    _assert_error(
        VllmAdapterErrorCode.INVALID_BLOCK_IDS,
        lambda: adapt_kv_cache_blocks(blocks, config),
    )


def test_module_imports_neither_vllm_nor_torch() -> None:
    source = Path(
        "src/cacheblend_gpt_oss/vllm_compat/v0_19_1/adapters.py"
    ).read_text(encoding="utf-8")
    assert "\nimport vllm" not in source
    assert "\nfrom vllm" not in source
    assert "\nimport torch" not in source
    assert "\nfrom torch" not in source
