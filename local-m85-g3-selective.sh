#!/usr/bin/env bash

run() {
  cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss || return 0

  export CACHEBLEND_SELECTIVE_CHECK_LAYER="${CACHEBLEND_SELECTIVE_CHECK_LAYER:-1}"
  export CACHEBLEND_SELECTIVE_RECOMPUTE_RATIO="${CACHEBLEND_SELECTIVE_RECOMPUTE_RATIO:-0.98}"
  export CACHEBLEND_SELECTIVE_SUFFIX_TOKENS="${CACHEBLEND_SELECTIVE_SUFFIX_TOKENS:-32}"
  export CACHEBLEND_RUN_BASE_DIR="${CACHEBLEND_RUN_BASE_DIR:-/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-selective-browsecomp-append-only-$(date +%Y%m%d)}"

  source ./local-m7-selective-g3.sh
  if test "${CACHEBLEND_VLLM_READY:-no}" != yes; then
    echo "STOP_SELECTIVE_SERVER_NOT_READY"
    return 0
  fi

  export CACHEBLEND_G3_RUN_POINTER="${CACHEBLEND_G3_RUN_POINTER:-/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-selective-browsecomp-append-only-$(date +%Y%m%d).current}"
  printf '%s\n' "$CACHEBLEND_RUN_DIR" > "$CACHEBLEND_G3_RUN_POINTER"

  .venv/bin/python -c "import json,os,importlib.metadata as m,torch; json.dump({'model_id':'openai/gpt-oss-20b','model_revision':os.environ['CACHEBLEND_MODEL_REVISION'],'tokenizer_revision':os.environ['CACHEBLEND_TOKENIZER_REVISION'],'plugin_commit':os.environ['CACHEBLEND_PLUGIN_COMMIT'],'model_config_digest':os.environ['CACHEBLEND_MODEL_CONFIG_DIGEST'],'kv_cache_config_digest':os.environ['CACHEBLEND_KV_CONFIG_DIGEST'],'vllm_version':m.version('vllm'),'lmcache_version':m.version('lmcache'),'torch_version':torch.__version__,'cuda_runtime':str(torch.version.cuda),'gpu_name':torch.cuda.get_device_name(0),'dtype':'torch.bfloat16'},open(os.path.join(os.environ['CACHEBLEND_RUN_DIR'],'runtime-identity.json'),'w'),indent=2); print('RUNTIME_IDENTITY_OK')"
  echo "SELECTIVE_RATIO=$CACHEBLEND_SELECTIVE_RECOMPUTE_RATIO"
  echo "G3_RUN_POINTER=$CACHEBLEND_G3_RUN_POINTER"
  echo "SELECTIVE_RUN_READY=$CACHEBLEND_RUN_DIR"
}

run
