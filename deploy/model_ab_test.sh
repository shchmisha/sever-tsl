#!/usr/bin/env bash
# Model A/B on the recognition server: read the same videos with ONE git ref and
# a CANDIDATE model, then restore the production model.
#
# Sequence: verify the API is idle -> stop sever-tsl -> point the model service at
# the candidate GGUF/mmproj -> restart sever-model and wait for its health ->
# read each video with the staged release's CLI -> restore the original model env,
# restart sever-model, start sever-tsl -> compare against baseline runs made with
# the same ref and the production model (tools/compare_runs.py).
#
# The original env files are backed up and restored on any exit, including errors.
# Run as root, detached:
#   systemd-run --unit sever-model-ab --property=TimeoutStartSec=infinity \
#     env RELEASE=/opt/sever-tsl/releases/20261018-v5-ab-b \
#         CANDIDATE_GGUF=/opt/sever-tsl/models/qwen36-27b-q6/Qwen3.6-27B-Q6_K.gguf \
#         CANDIDATE_MMPROJ=/opt/sever-tsl/models/qwen36-27b-q6/mmproj-F16.gguf \
#         CANDIDATE_ALIAS=qwen36-27b-q6 \
#         BASELINE_RUNS="/srv/sever/validation/ab-speed-20261018-v5/phase1-speed /srv/sever/validation/ab-speed-20261018-v4/phase1-speed" \
#         bash /root/sever-ab/model_ab_test.sh
set -euo pipefail

RELEASE=${RELEASE:?staged release directory with .venv and tools/compare_runs.py}
CANDIDATE_GGUF=${CANDIDATE_GGUF:?}
CANDIDATE_MMPROJ=${CANDIDATE_MMPROJ:?}
CANDIDATE_ALIAS=${CANDIDATE_ALIAS:?}
VIDEOS=${VIDEOS:-"/srv/sever/input/20260818-5_132107.mp4 /srv/sever/input/20260818-4_132351.mp4"}
BASELINE_RUNS=${BASELINE_RUNS:-}      # one baseline run dir per video, same order as VIDEOS
MAX_REQUESTS=${MAX_REQUESTS:-4000}
MAX_TOKENS=${MAX_TOKENS:-40000000}
STAMP=${STAMP:-$(date -u +%Y%m%d-%H%M%S)}
OUT=${OUT:-/srv/sever/validation/model-ab-$STAMP}
MODEL_ENV=${MODEL_ENV:-/etc/sever-tsl/local-model.env}
PROVIDER_ENV=${PROVIDER_ENV:-/etc/sever-tsl/local-provider.env}
RUNS_DIR=${RUNS_DIR:-/srv/sever/runs}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:8090}
API_URL=${API_URL:-http://127.0.0.1:8080}
HEALTH_WAIT_SECONDS=${HEALTH_WAIT_SECONDS:-900}

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
fail() { say "STOP: $*"; exit 1; }

[ "$(id -u)" = 0 ] || fail "run as root"
[ -x "$RELEASE/.venv/bin/sever" ] || fail "no CLI in $RELEASE"
[ -f "$RELEASE/tools/compare_runs.py" ] || fail "no compare_runs.py in $RELEASE"
for f in "$CANDIDATE_GGUF" "$CANDIDATE_MMPROJ" "$MODEL_ENV" "$PROVIDER_ENV" "$RUNS_DIR/.inference-evaluation.lock"; do
  [ -e "$f" ] || fail "missing $f"
done
for v in $VIDEOS; do [ -r "$v" ] || fail "missing video $v"; done
runuser -u sever -- test -r "$CANDIDATE_GGUF" || fail "the sever user cannot read $CANDIDATE_GGUF"

active_jobs() {
  python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get("status") in {"queued", "processing"}
          for p in Path(sys.argv[1]).glob("*/job.json") if p.is_file()))
PY
}
[ "$(active_jobs)" -eq 0 ] || fail "API jobs are queued or processing; try later"

install -d -o sever -g sever -m 0750 "$OUT"
cp -a "$MODEL_ENV" "$OUT/local-model.env.original"
cp -a "$PROVIDER_ENV" "$OUT/local-provider.env.original"
ORIGINAL_ALIAS=$(sed -n 's/^MODEL_ALIAS=//p' "$MODEL_ENV" | tr -d '"' | tail -1)
[ -n "$ORIGINAL_ALIAS" ] || fail "MODEL_ALIAS missing from $MODEL_ENV"

wait_model() {  # wait_model ALIAS
  local waited=0
  until curl -sf "$MODEL_URL/health" >/dev/null && curl -sf "$MODEL_URL/v1/models" | grep -q "\"$1\""; do
    [ "$waited" -lt "$HEALTH_WAIT_SECONDS" ] || return 1
    sleep 10; waited=$((waited + 10))
  done
  say "model service healthy with alias $1 after ${waited}s"
}

restore() {
  set +e
  say "restoring the production model ($ORIGINAL_ALIAS)"
  cp -a "$OUT/local-model.env.original" "$MODEL_ENV"
  cp -a "$OUT/local-provider.env.original" "$PROVIDER_ENV"
  systemctl restart sever-model
  if wait_model "$ORIGINAL_ALIAS"; then
    systemctl start sever-tsl
    sleep 5
    curl -sf "$API_URL/health" >/dev/null && say "API back up" || say "WARNING: API health check failed after restore"
  else
    say "WARNING: production model did not come back healthy; sever-tsl left stopped"
  fi
  chown -R sever:sever "$OUT" 2>/dev/null
}
trap restore EXIT

say "stopping the API and switching the model service to $CANDIDATE_ALIAS"
systemctl stop sever-tsl
sed -i -e "s|^MODEL_PATH=.*|MODEL_PATH=$CANDIDATE_GGUF|" -e "s|^MMPROJ_PATH=.*|MMPROJ_PATH=$CANDIDATE_MMPROJ|" \
       -e "s|^MODEL_ALIAS=.*|MODEL_ALIAS=$CANDIDATE_ALIAS|" "$MODEL_ENV"
grep -E '^(MODEL_PATH|MMPROJ_PATH|MODEL_ALIAS)=' "$MODEL_ENV" > "$OUT/candidate-model.env"
systemctl restart sever-model
wait_model "$CANDIDATE_ALIAS" || fail "candidate model did not become healthy"
journalctl -u sever-model --since "-15min" --no-pager -o cat | grep -i -E "arch|n_ctx|image_min|image_max|nextn|unused tensor|error|fail" | head -40 > "$OUT/candidate-startup.log" || true

# Same settings as the API service, except the model alias.
mapfile -t RUN_ENV < <(sed 's/^export //' "$PROVIDER_ENV" | grep -E '^(HARNESS|SEVER)_[A-Z0-9_]+=' \
  | sed -E -e 's/^([A-Z0-9_]+)="(.*)"$/\1=\2/' -e "s/^([A-Z0-9_]+)='(.*)'\$/\1=\2/" \
  | grep -v -E '^HARNESS_(MODEL_NAME|VISION_MODEL)=')
RUN_ENV+=("HARNESS_MODEL_NAME=$CANDIDATE_ALIAS" "HARNESS_VISION_MODEL=$CANDIDATE_ALIAS")

index=0
for video in $VIDEOS; do
  name=$(basename "$video" .mp4)
  say "reading $video with $CANDIDATE_ALIAS"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 \
    > "$OUT/gpu-$name.csv" 2>/dev/null & gpu_pid=$!
  before=$(date -u +%s); status=0
  ( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
      SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
      PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 PYTHONDONTWRITEBYTECODE=1 \
      "$RELEASE/.venv/bin/sever" process "$video" --out "$OUT/$CANDIDATE_ALIAS/$name" \
      --model "$CANDIDATE_ALIAS" --max-requests "$MAX_REQUESTS" --max-tokens "$MAX_TOKENS" ) \
      > "$OUT/run-$name.log" 2>&1 || status=$?
  kill "$gpu_pid" 2>/dev/null || true
  printf '{"model":"%s","video":"%s","exit_status":%d,"wall_seconds":%d}\n' "$CANDIDATE_ALIAS" "$name" "$status" \
    "$(( $(date -u +%s) - before ))" > "$OUT/run-$name.json"
  [ "$status" -eq 0 ] || say "WARNING: $name exited with status $status; see $OUT/run-$name.log"
  baseline=$(echo $BASELINE_RUNS | awk -v i=$((index + 1)) '{print $i}')
  if [ -n "$baseline" ] && [ -f "$baseline/results.json" ] && [ -f "$OUT/$CANDIDATE_ALIAS/$name/results.json" ]; then
    "$RELEASE/.venv/bin/python" "$RELEASE/tools/compare_runs.py" \
      --a "$baseline" --label-a "$ORIGINAL_ALIAS" --b "$OUT/$CANDIDATE_ALIAS/$name" --label-b "$CANDIDATE_ALIAS" \
      --truth "$RELEASE/analysis/ground-truth-expanded-provisional.json" --matching bracket \
      --out "$OUT/comparison-$name.json" --markdown "$OUT/comparison-$name.md" >/dev/null || say "WARNING: comparison failed for $name"
  fi
  index=$((index + 1))
done
say "candidate runs finished: $OUT"
