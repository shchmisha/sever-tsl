#!/usr/bin/env bash
# Side-by-side speed and accuracy test of two git refs on the recognition server.
#
# Installs each ref as its own release under /opt/sever-tsl/releases, runs its test
# suite, waits until no API job is queued or processing, then reads the same video
# with each release one after the other against the shared local model server, and
# finally compares calls, seconds, tokens and labelled accuracy.
#
# It never changes /opt/sever-tsl/current, the API service, the model service or
# existing runs. Run as root, ideally detached:
#   systemd-run --unit sever-ab-speed --property=TimeoutStartSec=infinity \
#     bash /root/sever-ab/ab_speed_test.sh
#   journalctl -fu sever-ab-speed
set -euo pipefail

BUNDLE=${BUNDLE:-/root/sever-ab/sever.bundle}
VIDEO=${VIDEO:-/srv/sever/input/20260818-5_132107.mp4}
REF_A=${REF_A:-main}
REF_B=${REF_B:-phase1-speed}
LEAF_CONCURRENCY_B=${LEAF_CONCURRENCY_B:-1}
MAX_REQUESTS=${MAX_REQUESTS:-4000}
MAX_TOKENS=${MAX_TOKENS:-40000000}
STAMP=${STAMP:-$(date -u +%Y%m%d-%H%M%S)}
RELEASES=${RELEASES:-/opt/sever-tsl/releases}
OUT=${OUT:-/srv/sever/validation/ab-speed-$STAMP}
ENV_FILE=${ENV_FILE:-/etc/sever-tsl/local-provider.env}
RUNS_DIR=${RUNS_DIR:-/srv/sever/runs}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:8090}
IDLE_POLL_SECONDS=${IDLE_POLL_SECONDS:-60}
IDLE_MAX_WAIT_SECONDS=${IDLE_MAX_WAIT_SECONDS:-43200}
# The production release's interpreter is readable by the sever user; uv's own
# managed Python would live under /root, which the recognition user cannot run.
PYTHON=${PYTHON:-$(readlink -f /opt/sever-tsl/current/.venv/bin/python3 2>/dev/null || true)}
UV=${UV:-$(command -v uv || echo /opt/sever-tsl/bootstrap/bin/uv)}

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
fail() { say "STOP: $*"; exit 1; }

[ "$(id -u)" = 0 ] || fail "run as root"
command -v git >/dev/null || fail "git is not installed"
[ -x "$UV" ] || fail "uv not found; set UV=/path/to/uv"
[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || fail "no interpreter from /opt/sever-tsl/current; set PYTHON=/path/to/python3.13"
[ -f "$BUNDLE" ] || fail "missing git bundle $BUNDLE"
[ -r "$VIDEO" ] || fail "missing video $VIDEO"
[ -r "$ENV_FILE" ] || fail "missing $ENV_FILE"
[ -f "$RUNS_DIR/.inference-evaluation.lock" ] || fail "missing shared lock $RUNS_DIR/.inference-evaluation.lock"
curl -sf "$MODEL_URL/health" >/dev/null || fail "model server $MODEL_URL is not healthy"
if [ "$LEAF_CONCURRENCY_B" -gt 1 ]; then
  slots=$(curl -sf "$MODEL_URL/props" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("total_slots", 1))')
  [ "$slots" -ge "$LEAF_CONCURRENCY_B" ] || fail "model server has $slots slot(s); LEAF_CONCURRENCY_B=$LEAF_CONCURRENCY_B needs llama-server --parallel $LEAF_CONCURRENCY_B"
fi

install -d -o sever -g sever -m 0750 "$OUT"
# Only HARNESS_* and SEVER_* settings reach the recognition process; secrets are not logged.
mapfile -t RUN_ENV < <(sed 's/^export //' "$ENV_FILE" | grep -E '^(HARNESS|SEVER)_[A-Z0-9_]+=' \
  | sed -E -e 's/^([A-Z0-9_]+)="(.*)"$/\1=\2/' -e "s/^([A-Z0-9_]+)='(.*)'\$/\1=\2/")
MODEL_NAME=$(printf '%s\n' "${RUN_ENV[@]}" | sed -n 's/^HARNESS_MODEL_NAME=//p' | tail -1)
[ -n "$MODEL_NAME" ] || fail "HARNESS_MODEL_NAME is not set in $ENV_FILE"

active_jobs() {
  python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
active = 0
for path in Path(sys.argv[1]).glob("*/job.json"):
    try:
        active += json.loads(path.read_text()).get("status") in {"queued", "processing"}
    except (OSError, ValueError):
        pass
print(active)
PY
}

stage() {  # stage LABEL REF -> prints release directory
  local label=$1 ref=$2 dir="$RELEASES/$STAMP-ab-$1"
  say "staging $ref as $dir" >&2
  git clone --quiet --branch "$ref" "$BUNDLE" "$dir"
  (cd "$dir" && "$UV" sync --frozen --extra dev --python "$PYTHON" --no-editable --quiet) >&2
  runuser -u sever -- "$dir/.venv/bin/python" -c "import sever" \
    || fail "the sever user cannot run $dir/.venv; set PYTHON to an interpreter it can read"
  say "testing $label" >&2
  if ! (cd "$dir" && PYDANTIC_DISABLE_PLUGINS=__all__ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
          .venv/bin/python -B -m pytest -q -p no:cacheprovider) > "$OUT/tests-$label.txt" 2>&1; then
    tail -20 "$OUT/tests-$label.txt" >&2
    fail "tests failed for $ref; see $OUT/tests-$label.txt"
  fi
  tail -1 "$OUT/tests-$label.txt" >&2
  git -C "$dir" rev-parse HEAD > "$OUT/commit-$label.txt"
  echo "$dir"
}

wait_idle() {
  local waited=0
  while [ "$(active_jobs)" -gt 0 ]; do
    [ "$waited" -lt "$IDLE_MAX_WAIT_SECONDS" ] || fail "API jobs stayed active for ${waited}s"
    say "waiting for active API jobs to finish (${waited}s)"
    sleep "$IDLE_POLL_SECONDS"; waited=$((waited + IDLE_POLL_SECONDS))
  done
}

run_ref() {  # run_ref LABEL RELEASE_DIR LEAF_CONCURRENCY
  local label=$1 dir=$2 concurrency=$3 gpu_pid status=0
  wait_idle
  say "reading $VIDEO with $label (leaf concurrency $concurrency)"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 \
    > "$OUT/gpu-$label.csv" 2>/dev/null & gpu_pid=$!
  local before; before=$(date -u +%s)
  # Same working directory and shared inference lock as the API service; the
  # env file's own values come later and win.
  cd /srv/sever
  runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
      SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
      HARNESS_LEAF_CONCURRENCY="$concurrency" PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 \
      PYTHONDONTWRITEBYTECODE=1 "$dir/.venv/bin/sever" process "$VIDEO" --out "$OUT/$label" \
      --model "$MODEL_NAME" --max-requests "$MAX_REQUESTS" --max-tokens "$MAX_TOKENS" \
      > "$OUT/run-$label.log" 2>&1 || status=$?
  kill "$gpu_pid" 2>/dev/null || true
  local overlapping; overlapping=$(active_jobs)
  printf '{"label":"%s","exit_status":%d,"wall_seconds":%d,"api_jobs_active_at_end":%d}\n' \
    "$label" "$status" "$(( $(date -u +%s) - before ))" "$overlapping" > "$OUT/run-$label.json"
  [ "$overlapping" -eq 0 ] || say "WARNING: $overlapping API job(s) were active when $label finished; its timing is contended"
  [ "$status" -eq 0 ] || say "WARNING: $label exited with status $status; see $OUT/run-$label.log"
}

DIR_A=$(stage a "$REF_A")
DIR_B=$(stage b "$REF_B")
printf 'video=%s\nref_a=%s\nref_b=%s\nleaf_concurrency_b=%s\nstarted=%s\n' \
  "$VIDEO" "$REF_A" "$REF_B" "$LEAF_CONCURRENCY_B" "$STAMP" > "$OUT/settings.txt"
run_ref "$REF_A" "$DIR_A" 1
run_ref "$REF_B" "$DIR_B" "$LEAF_CONCURRENCY_B"

say "comparing (ground truth is read only now, after both runs)"
"$DIR_B/.venv/bin/python" "$DIR_B/tools/compare_runs.py" \
  --a "$OUT/$REF_A" --label-a "$REF_A" --b "$OUT/$REF_B" --label-b "$REF_B" \
  --truth "$DIR_B/analysis/ground-truth-expanded-provisional.json" \
  --out "$OUT/comparison.json" --markdown "$OUT/comparison.md"
chown -R sever:sever "$OUT"
say "done: $OUT/comparison.md"
