#!/usr/bin/env bash

cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

export TIKTOKEN_ENCODINGS_BASE=/mnt/nvme3n1/labuser/.cache/tiktoken/encodings
export TIKTOKEN_RS_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
export TIKTOKEN_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken

export CACHEBLEND_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-policy-v2-20260815-retry123459
export CACHEBLEND_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/transfer-config.json"
export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_CONTROLS_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-c9bc7ab-recovery-20260815
export CACHEBLEND_EXCLUDED_CANDIDATE=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-c9bc7ab-formal-20260814/cacheblend-100pct-v2.json

if [ ! -s "$CACHEBLEND_TRANSFER_CONFIG" ]; then
  echo "MISSING_TRANSFER_CONFIG=$CACHEBLEND_TRANSFER_CONFIG"
else
  export CACHEBLEND_TRANSFER_CONFIG_JSON="$(<"$CACHEBLEND_TRANSFER_CONFIG")"

  export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
  export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742

  export CACHEBLEND_PLUGIN_COMMIT="$(.venv/bin/python -c 'import json,os; print(json.load(open(os.environ["CACHEBLEND_TRANSFER_CONFIG"]))["kv_connector_extra_config"]["adapter_revision"])')"
  export CACHEBLEND_TRANSFER_EVIDENCE="$(.venv/bin/python -c 'import json,os; print(json.load(open(os.environ["CACHEBLEND_TRANSFER_CONFIG"]))["kv_connector_extra_config"]["transfer_evidence_path"])')"

  .venv/bin/python -c 'import json,os; json.load(open(os.environ["CACHEBLEND_TRANSFER_CONFIG"])); print("TRANSFER_CONFIG_JSON_OK")'
  .venv/bin/python -c 'from openai_harmony import load_harmony_encoding; load_harmony_encoding("HarmonyGptOss"); print("HARMONY_OK")'

  if ! .venv/bin/python -c 'import socket; s=socket.socket(); s.settimeout(1); s.connect(("127.0.0.1",5556)); s.close()'; then
    nohup .venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
      --host 127.0.0.1 \
      --port 5556 \
      --chunk-size 256 \
      --hash-algorithm blake3 \
      --l1-size-gb 16 \
      --l1-init-size-gb 16 \
      --eviction-policy LRU \
      --max-workers 1 \
      > "$CACHEBLEND_RUN_DIR/lmcache-direct.log" 2>&1 < /dev/null &
    sleep 5
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM 8000/tcp >/dev/null 2>&1 || true
  fi

  sleep 3

  export CACHEBLEND_ATTEMPT_SUFFIX="$(date +%H%M%S)"
  export CACHEBLEND_CANDIDATE="$CACHEBLEND_RUN_DIR/cacheblend-100pct-harmony-direct-$CACHEBLEND_ATTEMPT_SUFFIX.json"
  export CACHEBLEND_CAPTURE_LOG="$CACHEBLEND_RUN_DIR/capture-harmony-direct-$CACHEBLEND_ATTEMPT_SUFFIX.txt"
  export CACHEBLEND_VLLM_LOG="$CACHEBLEND_RUN_DIR/vllm-harmony-direct-$CACHEBLEND_ATTEMPT_SUFFIX.log"

  nohup .venv/bin/vllm serve \
    "$CACHEBLEND_MODEL_PATH" \
    --served-model-name openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.50 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --long-prefill-token-threshold 0 \
    --no-async-scheduling \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --attention-backend TRITON_ATTN \
    --no-disable-hybrid-kv-cache-manager \
    --generation-config vllm \
    --max-logprobs -1 \
    --port 8000 \
    --kv-transfer-config "$CACHEBLEND_TRANSFER_CONFIG_JSON" \
    > "$CACHEBLEND_VLLM_LOG" 2>&1 < /dev/null &

  export CACHEBLEND_VLLM_PID=$!
  echo "VLLM_PID=$CACHEBLEND_VLLM_PID"

  export CACHEBLEND_READY=0

  for n in $(seq 1 180); do
    export CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_RUN_DIR/models-harmony-direct.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
    if [ "$CACHEBLEND_HTTP_CODE" = "200" ]; then
      export CACHEBLEND_READY=1
      break
    fi
    if ! kill -0 "$CACHEBLEND_VLLM_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if [ "$CACHEBLEND_READY" = "1" ]; then
    echo "VLLM_READY=yes"

    .venv/bin/python scripts/capture_moved_document.py \
      --base-url http://127.0.0.1:8000 \
      --mode cacheblend_100pct \
      --model-revision "$CACHEBLEND_MODEL_REVISION" \
      --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
      --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
      --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
      --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
      --output "$CACHEBLEND_CANDIDATE" \
      | tee "$CACHEBLEND_CAPTURE_LOG"

    export CACHEBLEND_CAPTURE_STATUS=${PIPESTATUS[0]}
    echo "CAPTURE_STATUS=$CACHEBLEND_CAPTURE_STATUS"

    if [ "$CACHEBLEND_CAPTURE_STATUS" = "0" ] && [ -s "$CACHEBLEND_CANDIDATE" ] && [ -s "$CACHEBLEND_TRANSFER_EVIDENCE" ]; then
      export CACHEBLEND_VERDICT="$CACHEBLEND_RUN_DIR/probability-v2-verdict-harmony-direct-$CACHEBLEND_ATTEMPT_SUFFIX.json"

      .venv/bin/python scripts/evaluate_probability_ensemble.py \
        --manifest "$CACHEBLEND_RUN_DIR/probability-v2-manifest.json" \
        --baseline "$CACHEBLEND_CONTROLS_DIR/scatter-disabled-control-1.json" \
        --baseline "$CACHEBLEND_CONTROLS_DIR/scatter-disabled-control-2.json" \
        --baseline "$CACHEBLEND_CONTROLS_DIR/scatter-disabled-control-3.json" \
        --baseline "$CACHEBLEND_CONTROLS_DIR/scatter-disabled-control-4.json" \
        --baseline "$CACHEBLEND_CONTROLS_DIR/scatter-disabled-control-5.json" \
        --excluded-candidate "$CACHEBLEND_EXCLUDED_CANDIDATE" \
        --cacheblend "$CACHEBLEND_CANDIDATE" \
        --transfer-evidence "$CACHEBLEND_TRANSFER_EVIDENCE" \
        --output "$CACHEBLEND_VERDICT" \
        | tee "$CACHEBLEND_RUN_DIR/probability-v2-verdict-harmony-direct-$CACHEBLEND_ATTEMPT_SUFFIX.txt"

      export CACHEBLEND_EVAL_STATUS=${PIPESTATUS[0]}
      echo "EVAL_STATUS=$CACHEBLEND_EVAL_STATUS"

      if [ -s "$CACHEBLEND_VERDICT" ]; then
        .venv/bin/python -c 'import json,os; d=json.load(open(os.environ["CACHEBLEND_VERDICT"])); print({"status":d.get("status"),"passed":d.get("passed"),"failure_reasons":d.get("failure_reasons"),"candidate_artifact_digest":d.get("candidate_artifact_digest"),"manifest_digest":d.get("manifest_digest")})'
      fi
    else
      echo "CANDIDATE_NOT_READY_OR_TRANSFER_EVIDENCE_MISSING"
      echo "TRANSFER_EVIDENCE=$CACHEBLEND_TRANSFER_EVIDENCE"
      tail -n 160 "$CACHEBLEND_VLLM_LOG"
    fi
  else
    echo "VLLM_READY=no"
    tail -n 160 "$CACHEBLEND_VLLM_LOG"
  fi
fi
