#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the prospective M3 probability-v2 gate on solab-g3.
#
# This runner deliberately reuses the five already-captured scatter-disabled
# controls.  It creates one new artifact directory, freezes the v2 policy,
# starts clean pinned LMCache/vLLM services in tmux, captures one real
# CacheBlend candidate, and evaluates it.  The serving identity remains the
# commit used by those controls; the current checkout supplies the gate code.

set -Eeuo pipefail

ROOT="/mnt/nvme3n1/mlee/cacheblend-gpt-oss"
PYTHON="$ROOT/.venv/bin/python"
VLLM="$ROOT/.venv/bin/vllm"
BRANCH="cacheblend-scatter-diagnostic-and-checklayer"
GATE_COMMIT="1ae52ce"
SERVING_COMMIT="c9bc7ab884a4fae1d4d8528807d7de5b215b8bd2"
MODEL_PATH="/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"
TOKENIZER_REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_CONFIG_DIGEST="1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0"
KV_CONFIG_DIGEST="131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742"
CONTROLS_DIR="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-c9bc7ab-recovery-20260815"
EXCLUDED_CANDIDATE="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-c9bc7ab-formal-20260814/cacheblend-100pct-v2.json"
RUN_DIR_BASE="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-1ae52ce-policy-v2-20260815"
SESSION="cacheblend-m3-probability-v2"
LM_CACHE_PORT=5556
VLLM_PORT=8000

stop_port_listener() {
    local port="$1"
    local pids
    pids="$(lsof -t -n -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        echo "Stopping listeners on TCP $port: $pids"
        for pid in $pids; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 3
    fi
}

port_ready() {
    local port="$1"
    "$PYTHON" -c 'import socket,sys; s=socket.socket(); s.settimeout(0.5); ok=s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0; s.close(); sys.exit(0 if ok else 1)' "$port"
}

wait_for_port() {
    local port="$1"
    local seconds="$2"
    local attempt
    for ((attempt = 1; attempt <= seconds * 2; attempt++)); do
        if port_ready "$port"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

http_ready() {
    "$PYTHON" -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8000/v1/models", timeout=1).read(1)' >/dev/null 2>&1
}

wait_for_http() {
    local seconds="$1"
    local attempt
    for ((attempt = 1; attempt <= seconds * 2; attempt++)); do
        if http_ready; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

show_pane_logs() {
    tmux capture-pane -t "$SESSION:0.0" -p -S -80 2>/dev/null || true
    tmux capture-pane -t "$SESSION:0.1" -p -S -80 2>/dev/null || true
}

main() {
    cd "$ROOT"

    echo "Updating $BRANCH..."
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
    if [[ "$(git rev-parse --short HEAD)" != "$GATE_COMMIT"* ]]; then
        echo "Expected gate commit $GATE_COMMIT, observed $(git rev-parse HEAD)."
        echo "Push the gate commit and run this script again."
        return 0
    fi

    local run_dir="$RUN_DIR_BASE"
    if [[ -e "$run_dir" ]]; then
        run_dir="${RUN_DIR_BASE}-retry$(date +%H%M%S)"
    fi
    mkdir -p "$run_dir"

    echo "Using run directory: $run_dir"
    echo "Reusing controls from: $CONTROLS_DIR"

    local freeze_report="$run_dir/probability-v2-freeze-report.json"
    local manifest="$run_dir/probability-v2-manifest.json"
    local freeze_status=0
    "$PYTHON" "$ROOT/scripts/freeze_probability_ensemble.py" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-1.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-2.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-3.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-4.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-5.json" \
        --excluded-candidate "$EXCLUDED_CANDIDATE" \
        --output "$manifest" \
        | tee "$freeze_report" || freeze_status=$?

    if ((freeze_status != 0)); then
        echo "Probability-v2 freeze did not pass. No GPU candidate was started."
        return 0
    fi

    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({"stable":d["stable"],"status":d["status"],"policy_version":d["manifest"]["policy_version"],"manifest_digest":d["manifest_digest"]})' "$freeze_report"

    local sidecar="$run_dir/sidecar.sqlite3"
    local transfer_evidence="$run_dir/transfer-evidence.json"
    local transfer_config="$run_dir/transfer-config.json"
    "$PYTHON" -c 'import sys; from cacheblend_gpt_oss.storage.sidecar import SidecarMode,open_sidecar_index; i=open_sidecar_index(sys.argv[1],SidecarMode.WORKER_READ_WRITE); i.close()' "$sidecar"
    "$PYTHON" "$ROOT/scripts/render_transfer_config.py" \
        --lmcache-server-url "tcp://127.0.0.1:$LM_CACHE_PORT" \
        --sidecar-path "$sidecar" \
        --model-revision "$MODEL_REVISION" \
        --tokenizer-revision "$TOKENIZER_REVISION" \
        --model-config-digest "$MODEL_CONFIG_DIGEST" \
        --kv-cache-config-digest "$KV_CONFIG_DIGEST" \
        --adapter-revision "$SERVING_COMMIT" \
        --transfer-evidence-path "$transfer_evidence" \
        --staging-token-capacity 1024 \
        --request-timeout-seconds 120 \
        > "$transfer_config"

    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); x=d["kv_connector_extra_config"]; print({"connector":d["kv_connector"],"mode":x["mode"],"sidecar_path":x["sidecar_path"],"lmcache_server_url":x["lmcache_server_url"]})' "$transfer_config"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi
    stop_port_listener "$LM_CACHE_PORT"
    stop_port_listener "$VLLM_PORT"

    tmux new-session -d -s "$SESSION" -n services
    tmux send-keys -t "$SESSION:0.0" \
        "cd '$ROOT' && '$PYTHON' -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 --host 127.0.0.1 --port $LM_CACHE_PORT --chunk-size 256 --hash-algorithm blake3 --l1-size-gb 16 --l1-init-size-gb 16 --eviction-policy LRU --max-workers 1 2>&1 | tee '$run_dir/lmcache-server.log'" C-m

    echo "Waiting for LMCache on TCP $LM_CACHE_PORT..."
    if ! wait_for_port "$LM_CACHE_PORT" 60; then
        echo "LMCache did not become ready. Recent service logs:"
        show_pane_logs
        return 0
    fi

    tmux split-window -h -t "$SESSION:0"
    tmux send-keys -t "$SESSION:0.1" \
        "cd '$ROOT' && export VLLM_USE_V2_MODEL_RUNNER=0 && '$VLLM' serve '$MODEL_PATH' --served-model-name openai/gpt-oss-20b --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 131072 --gpu-memory-utilization 0.50 --max-num-seqs 1 --max-num-batched-tokens 1024 --long-prefill-token-threshold 0 --no-async-scheduling --enforce-eager --no-enable-prefix-caching --kv-cache-dtype auto --attention-backend TRITON_ATTN --no-disable-hybrid-kv-cache-manager --generation-config vllm --max-logprobs -1 --port $VLLM_PORT --kv-transfer-config \"\$(cat '$transfer_config')\" 2>&1 | tee '$run_dir/normal-cacheblend-server.log'" C-m

    echo "Waiting for vLLM on TCP $VLLM_PORT..."
    if ! wait_for_http 180; then
        echo "vLLM did not become ready. Recent service logs:"
        show_pane_logs
        return 0
    fi

    local candidate="$run_dir/cacheblend-100pct-v2.json"
    local capture_log="$run_dir/cacheblend-capture-v2.txt"
    local capture_status=0
    "$PYTHON" "$ROOT/scripts/capture_moved_document.py" \
        --base-url "http://127.0.0.1:$VLLM_PORT" \
        --mode cacheblend_100pct \
        --model-revision "$MODEL_REVISION" \
        --tokenizer-revision "$TOKENIZER_REVISION" \
        --plugin-commit "$SERVING_COMMIT" \
        --model-config-digest "$MODEL_CONFIG_DIGEST" \
        --kv-cache-config-digest "$KV_CONFIG_DIGEST" \
        --output "$candidate" \
        | tee "$capture_log" || capture_status=$?

    if ((capture_status != 0)); then
        echo "Candidate capture failed. Recent service logs:"
        show_pane_logs
        return 0
    fi

    local verdict="$run_dir/probability-v2-verdict.json"
    local verdict_log="$run_dir/probability-v2-verdict-output.txt"
    local evaluate_status=0
    "$PYTHON" "$ROOT/scripts/evaluate_probability_ensemble.py" \
        --manifest "$manifest" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-1.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-2.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-3.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-4.json" \
        --baseline "$CONTROLS_DIR/scatter-disabled-control-5.json" \
        --excluded-candidate "$EXCLUDED_CANDIDATE" \
        --cacheblend "$candidate" \
        --transfer-evidence "$transfer_evidence" \
        --output "$verdict" \
        | tee "$verdict_log" || evaluate_status=$?

    echo
    echo "M3 probability-v2 run complete"
    echo "RUN_DIR=$run_dir"
    echo "TMUX_SESSION=$SESSION"
    echo "EVALUATOR_EXIT=$evaluate_status"
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({"status":d["status"],"passed":d["passed"],"failure_reasons":d["failure_reasons"],"transfer_evidence_bound":d["transfer_evidence_bound"],"manifest_digest":d["manifest_digest"]})' "$verdict"
    echo "For logs: tmux attach -t $SESSION"
}

main "$@"
