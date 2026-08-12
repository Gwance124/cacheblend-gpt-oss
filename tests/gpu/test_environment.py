from importlib.metadata import version

import pytest

from cacheblend_gpt_oss import PINNED_TARGET


@pytest.mark.gpu
@pytest.mark.integration
def test_pinned_solab_g3_runtime() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("lmcache")

    if not torch.cuda.is_available():
        pytest.skip("manual solab-g3 test: CUDA is not available")

    assert version("vllm") == PINNED_TARGET.vllm_version
    assert version("lmcache") == PINNED_TARGET.lmcache_version
    assert torch.__version__ == PINNED_TARGET.torch_version
    assert torch.version.cuda == PINNED_TARGET.cuda_runtime
    assert torch.cuda.get_device_name(0) == PINNED_TARGET.gpu_name
    assert torch.cuda.get_device_capability(0) == (8, 0)
