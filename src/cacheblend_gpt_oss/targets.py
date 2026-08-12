"""The sole runtime envelope supported by this prototype."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeTarget:
    """Pinned software and hardware target; this is not capability detection."""

    model_id: str
    vllm_version: str
    lmcache_version: str
    torch_version: str
    cuda_runtime: str
    gpu_name: str


PINNED_TARGET = RuntimeTarget(
    model_id="openai/gpt-oss-20b",
    vllm_version="0.19.1",
    lmcache_version="0.4.3",
    torch_version="2.10.0+cu128",
    cuda_runtime="12.8",
    gpu_name="NVIDIA A100-SXM4-80GB",
)

