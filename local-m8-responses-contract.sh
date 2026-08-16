#!/usr/bin/env bash

cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

export TIKTOKEN_ENCODINGS_BASE=/mnt/nvme3n1/labuser/.cache/tiktoken/encodings
export TIKTOKEN_RS_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
export TIKTOKEN_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
export TRITON_CACHE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/.triton-cache
mkdir -p "$TRITON_CACHE_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0

export CACHEBLEND_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-policy-v2-20260815-retry123459
export CACHEBLEND_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/transfer-config.json"
export CACHEBLEND_M8_SUFFIX="$(date +%H%M%S)"
export CACHEBLEND_M8_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/responses-transfer-config-$CACHEBLEND_M8_SUFFIX.json"
export CACHEBLEND_M8_TRANSFER_EVIDENCE="$CACHEBLEND_RUN_DIR/responses-transfer-evidence-$CACHEBLEND_M8_SUFFIX.json"
export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742

if [ ! -s "$CACHEBLEND_TRANSFER_CONFIG" ]; then
  echo "MISSING_TRANSFER_CONFIG=$CACHEBLEND_TRANSFER_CONFIG"
else
  export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"

  .venv/bin/python -c 'from openai_harmony import load_harmony_encoding; load_harmony_encoding("HarmonyGptOss"); print("HARMONY_OK")'

  if ! .venv/bin/python -c 'import socket; s=socket.socket(); s.settimeout(1); s.connect(("127.0.0.1",5556)); s.close()'; then
    echo "LM_CACHE_NOT_LISTENING_ON_5556"
  else
    .venv/bin/python scripts/render_transfer_config.py \
      --lmcache-server-url tcp://127.0.0.1:5556 \
      --sidecar-path "$CACHEBLEND_RUN_DIR/sidecar.sqlite3" \
      --model-revision "$CACHEBLEND_MODEL_REVISION" \
      --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
      --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
      --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
      --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
      --transfer-evidence-path "$CACHEBLEND_M8_TRANSFER_EVIDENCE" \
      --staging-token-capacity 1024 \
      --request-timeout-seconds 180 \
      > "$CACHEBLEND_M8_TRANSFER_CONFIG"

    export CACHEBLEND_TRANSFER_CONFIG_JSON="$(<"$CACHEBLEND_M8_TRANSFER_CONFIG")"
    echo "M8_TRANSFER_CONFIG=$CACHEBLEND_M8_TRANSFER_CONFIG"
    echo "M8_TRANSFER_EVIDENCE=$CACHEBLEND_M8_TRANSFER_EVIDENCE"

    export CACHEBLEND_VLLM_PID="$(ss -ltnp 2>/dev/null | awk '$4 ~ /:8000$/ { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }')"
    if [ -n "$CACHEBLEND_VLLM_PID" ]; then
      echo "STOPPING_VLLM_PID=$CACHEBLEND_VLLM_PID"
      kill -TERM "$CACHEBLEND_VLLM_PID" 2>/dev/null || true
      sleep 5
    fi

    export CACHEBLEND_VLLM_LOG="$CACHEBLEND_RUN_DIR/responses-server-$CACHEBLEND_M8_SUFFIX.log"
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
      --enable-auto-tool-choice \
      --tool-call-parser openai \
      --generation-config vllm \
      --max-logprobs -1 \
      --port 8000 \
      --kv-transfer-config "$CACHEBLEND_TRANSFER_CONFIG_JSON" \
      > "$CACHEBLEND_VLLM_LOG" 2>&1 < /dev/null &

    export CACHEBLEND_VLLM_PID=$!
    echo "VLLM_PID=$CACHEBLEND_VLLM_PID"
    export CACHEBLEND_READY=0

    for n in $(seq 1 180); do
      export CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_RUN_DIR/responses-models.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
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

      export CACHEBLEND_RESPONSES_ARTIFACT="$CACHEBLEND_RUN_DIR/responses-contract-$CACHEBLEND_M8_SUFFIX.json"
      export CACHEBLEND_RESPONSES_REPORT="$CACHEBLEND_RUN_DIR/responses-contract-report-$CACHEBLEND_M8_SUFFIX.json"
      export CACHEBLEND_RESPONSES_LOG="$CACHEBLEND_RUN_DIR/responses-contract-$CACHEBLEND_M8_SUFFIX.txt"
      export CACHEBLEND_RESPONSES_METRICS="$CACHEBLEND_RUN_DIR/responses-contract-metrics-$CACHEBLEND_M8_SUFFIX.txt"
      export CACHEBLEND_RESPONSES_VALIDATION_LOG="$CACHEBLEND_RUN_DIR/responses-contract-validation-$CACHEBLEND_M8_SUFFIX.txt"

      .venv/bin/python scripts/check_responses_contract.py \
        --base-url http://127.0.0.1:8000 \
        --model-revision "$CACHEBLEND_MODEL_REVISION" \
        --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
        --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
        --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
        --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
        --output "$CACHEBLEND_RESPONSES_ARTIFACT" \
        | tee "$CACHEBLEND_RESPONSES_LOG"

      export CACHEBLEND_CONTRACT_STATUS=${PIPESTATUS[0]}
      echo "CONTRACT_CAPTURE_STATUS=$CACHEBLEND_CONTRACT_STATUS"

      curl -sS http://127.0.0.1:8000/metrics \
        | grep 'vllm:cacheblend_' \
        > "$CACHEBLEND_RESPONSES_METRICS"

      if [ -s "$CACHEBLEND_RESPONSES_ARTIFACT" ]; then
        .venv/bin/python scripts/validate_responses_contract.py \
          --input "$CACHEBLEND_RESPONSES_ARTIFACT" \
          --output "$CACHEBLEND_RESPONSES_REPORT" \
          | tee "$CACHEBLEND_RESPONSES_VALIDATION_LOG"

        export CACHEBLEND_VALIDATION_STATUS=${PIPESTATUS[0]}
        echo "CONTRACT_VALIDATION_STATUS=$CACHEBLEND_VALIDATION_STATUS"

        .venv/bin/python -c 'import json,os; d=json.load(open(os.environ["CACHEBLEND_RESPONSES_REPORT"])); print({"status":d.get("status"),"passed":d.get("passed"),"failure_reasons":d.get("failure_reasons"),"evidence_digest":d.get("evidence_digest")})'
      fi
    else
      echo "VLLM_READY=no"
      tail -n 160 "$CACHEBLEND_VLLM_LOG"
    fi
  fi
fi
