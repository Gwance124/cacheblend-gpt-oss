"""Compatibility namespace pinned to vLLM 0.19.1.

Import the concrete connector from ``.connector`` only in a pinned vLLM
runtime. Keeping it out of this module preserves lightweight CPU-only imports.
"""

CONNECTOR_CLASS_NAME = "GptOssCacheBlendConnector"
CONNECTOR_MODULE_PATH = "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector"

__all__ = ["CONNECTOR_CLASS_NAME", "CONNECTOR_MODULE_PATH"]
