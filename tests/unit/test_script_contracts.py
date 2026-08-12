"""CPU-only tests for the manual solab-g3 script boundaries."""

from __future__ import annotations

import json
import runpy

import pytest

from cacheblend_gpt_oss.targets import PINNED_TARGET

_capture = runpy.run_path(
    "scripts/capture_moved_document.py",
    run_name="cacheblend_capture_script_test",
)
_responses = runpy.run_path(
    "scripts/check_responses_contract.py",
    run_name="cacheblend_responses_script_test",
)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:8000",
        "http://example.invalid:8000",
        "http://user@127.0.0.1:8000",
        "http://127.0.0.1:8000/?remote=1",
        "http://127.0.0.1:8000/path",
    ],
)
def test_manual_clients_reject_non_local_or_ambiguous_urls(base_url: str) -> None:
    for namespace in (_capture, _responses):
        client_type = namespace["LocalVllmClient"]
        with pytest.raises(ValueError, match="local HTTP vLLM endpoint"):
            client_type(base_url, "EMPTY", 1.0)


def test_capture_payload_uses_exact_raw_token_completion_contract() -> None:
    payload_builder = _capture["_completion_payload"]
    payload = payload_builder((11, 12, 13), full=True)

    assert payload["model"] == PINNED_TARGET.model_id
    assert payload["prompt"] == [11, 12, 13]
    assert payload["add_special_tokens"] is False
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 0
    assert payload["logprobs"] == 201_088
    assert payload["return_token_ids"] is True
    assert payload["stream"] is False

    narrow = payload_builder((11,), full=False)
    assert narrow["logprobs"] == 1


def test_responses_payload_is_cacheblend_transparent() -> None:
    payload_builder = _responses["_request_payload"]
    payload = payload_builder([{"role": "user", "content": "hello"}])

    assert payload["model"] == PINNED_TARGET.model_id
    assert payload["input"] == [{"role": "user", "content": "hello"}]
    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["truncation"] == "disabled"
    assert payload["store"] is False
    assert not any(
        key.casefold().startswith(("cache", "segment", "kv_"))
        for key in payload
    )


@pytest.mark.parametrize(
    "namespace",
    [_capture, _responses],
)
def test_manual_scripts_require_the_pinned_served_model(namespace: dict) -> None:
    require_model = namespace["_require_served_model"]
    require_model({"data": [{"id": PINNED_TARGET.model_id}]})

    with pytest.raises(ValueError, match="pinned GPT-OSS model"):
        require_model({"data": [{"id": "other-model"}]})
    with pytest.raises(ValueError, match="invalid /v1/models"):
        require_model({"data": "not-a-list"})


def test_capture_payload_is_json_serializable_without_nonfinite_values() -> None:
    payload = _capture["_completion_payload"]((1, 2), full=False)
    encoded = json.dumps(payload, allow_nan=False)
    assert '"prompt": [1, 2]' in encoded
