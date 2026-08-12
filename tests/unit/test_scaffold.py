from cacheblend_gpt_oss import PINNED_TARGET, __version__


def test_package_exposes_only_the_pinned_runtime_target() -> None:
    assert __version__ == "0.0.0"
    assert PINNED_TARGET.model_id == "openai/gpt-oss-20b"
    assert PINNED_TARGET.vllm_version == "0.19.1"
    assert PINNED_TARGET.lmcache_version == "0.4.3"
    assert PINNED_TARGET.torch_version == "2.10.0+cu128"
    assert PINNED_TARGET.cuda_runtime == "12.8"
    assert PINNED_TARGET.gpu_name == "NVIDIA A100-SXM4-80GB"

