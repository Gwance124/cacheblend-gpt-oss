"""CPU-only tests for the manual solab-g3 script boundaries."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.correctness import CorrectnessCase
from cacheblend_gpt_oss.targets import PINNED_TARGET
from cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_registry import (
    SelectiveGateEvidence,
)

_capture = runpy.run_path(
    "scripts/capture_moved_document.py",
    run_name="cacheblend_capture_script_test",
)
_responses = runpy.run_path(
    "scripts/check_responses_contract.py",
    run_name="cacheblend_responses_script_test",
)
_gate_hash = runpy.run_path(
    "scripts/hash_selective_gate_artifacts.py",
    run_name="cacheblend_gate_hash_script_test",
)
_ensemble_freeze = runpy.run_path(
    "scripts/freeze_correctness_ensemble.py",
    run_name="cacheblend_ensemble_freeze_script_test",
)
_ensemble_evaluate = runpy.run_path(
    "scripts/evaluate_cacheblend_ensemble.py",
    run_name="cacheblend_ensemble_evaluate_script_test",
)
_gate_verify = runpy.run_path(
    "scripts/verify_selective_gate_artifacts.py",
    run_name="cacheblend_gate_verify_script_test",
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


def test_capture_parser_exposes_explicit_source_warmup_protocol() -> None:
    parser = _capture["_parser"]()
    args = parser.parse_args(
        [
            "--mode",
            "full_prefill",
            "--model-revision",
            "model-revision",
            "--tokenizer-revision",
            "tokenizer-revision",
            "--plugin-commit",
            "a" * 40,
            "--model-config-digest",
            "b" * 64,
            "--kv-cache-config-digest",
            "c" * 64,
            "--output",
            "artifact.json",
            "--warm-source-before-target",
            "--connector-attached-control",
        ]
    )

    assert args.warm_source_before_target is True
    assert args.connector_attached_control is True


def test_gate_hash_script_exposes_a_callable_main() -> None:
    assert callable(_gate_hash["main"])


def test_ensemble_scripts_expose_callable_mains() -> None:
    assert callable(_ensemble_freeze["main"])
    assert callable(_ensemble_evaluate["main"])


def test_ensemble_evaluator_requires_transfer_or_explicit_cache_miss() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_cacheblend_ensemble.py",
            "--manifest",
            "manifest.json",
            "--baseline",
            "one.json",
            "--baseline",
            "two.json",
            "--baseline",
            "three.json",
            "--baseline",
            "four.json",
            "--baseline",
            "five.json",
            "--cacheblend",
            "candidate.json",
            "--output",
            "verdict.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--transfer-evidence" in result.stderr


def test_ensemble_no_transfer_exception_is_cache_miss_only() -> None:
    validate = _ensemble_evaluate["_validate_no_transfer_cache_miss"]
    cache_miss = SimpleNamespace(
        prompt=SimpleNamespace(case=CorrectnessCase.CACHE_MISS),
        connector=SimpleNamespace(
            kv_tokens_found=0,
            kv_tokens_loaded=0,
            kv_tokens_rejected=0,
        ),
    )
    validate(cache_miss)

    cache_miss.prompt.case = CorrectnessCase.MOVED_DOCUMENT
    with pytest.raises(ValueError, match="CACHE_MISS"):
        validate(cache_miss)


def test_gate_verify_script_exposes_a_callable_main() -> None:
    assert callable(_gate_verify["main"])


def test_gate_verify_script_checks_current_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = []
    for index in range(5):
        path = tmp_path / f"artifact-{index}.txt"
        path.write_bytes(f"artifact-{index}".encode("ascii"))
        paths.append(path)
    evidence = SelectiveGateEvidence.from_artifact_paths(
        runtime=paths[0],
        full_prefill=paths[1],
        transfer=paths[2],
        yarn=paths[3],
        hybrid_sink=paths[4],
    )
    evidence_path = tmp_path / "selective-gate-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence.to_dict()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_selective_gate_artifacts.py",
            "--evidence",
            str(evidence_path),
            "--runtime",
            str(paths[0]),
            "--full-prefill",
            str(paths[1]),
            "--transfer",
            str(paths[2]),
            "--yarn",
            str(paths[3]),
            "--hybrid-sink",
            str(paths[4]),
        ],
    )

    assert _gate_verify["main"]() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is True
    assert report["evidence"] == evidence.to_dict()
