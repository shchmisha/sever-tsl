#!/usr/bin/env bash
# GPU queue for 2026-10-18, run after the Qwen3.6 model A/B (sever-model-ab) finishes:
#   1. verify the API came back on the new release;
#   2. live confirmation of the fusion fix: accuracy-ab-1 (371cd3e) vs phase1-speed HEAD on videos 5 and 4;
#   3. MTP speculative decoding test: add --spec-type draft-mtp --spec-draft-n-max 2 to the model
#      service through a temporary systemd drop-in, read video 5 with /opt/sever-tsl/current,
#      remove the drop-in, restore the model and API, and compare against step 2's MTP-off run.
# Every model-service change is undone on exit, including on errors.
set -euo pipefail

OUT=${OUT:-/srv/sever/validation/gpu-queue-20261018}
RUNS_DIR=/srv/sever/runs
MODEL_URL=http://127.0.0.1:8090
API_URL=http://127.0.0.1:8080
DROPIN_DIR=/etc/systemd/system/sever-model.service.d
DROPIN=$DROPIN_DIR/zz-mtp-test.conf
MTP_N=${MTP_N:-2}
V5=/srv/sever/input/20260818-5_132107.mp4
V4=/srv/sever/input/20260818-4_132351.mp4

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
active_jobs() {
  python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get("status") in {"queued", "processing"} for p in Path(sys.argv[1]).glob("*/job.json") if p.is_file()))
PY
}
wait_model() {
  local waited=0
  until curl -sf "$MODEL_URL/health" >/dev/null && curl -sf "$MODEL_URL/v1/models" | grep -q '"qwen38-27b-q6"'; do
    [ "$waited" -lt 900 ] || return 1; sleep 10; waited=$((waited + 10))
  done
}

install -d -o sever -g sever -m 0750 "$OUT"

say "waiting for sever-model-ab to finish"
while systemctl is-active -q sever-model-ab; do sleep 60; done
journalctl -u sever-model-ab --no-pager -o cat | grep -v pam_unix | tail -5 > "$OUT/model-ab-tail.log" || true

say "checking the API on the new release"
for i in $(seq 1 30); do curl -sf "$API_URL/health" > "$OUT/api-health-after-deploy.json" && break; sleep 10; done
readlink -f /opt/sever-tsl/current > "$OUT/current-release.txt"
python3 -c "import json; d=json.load(open('$OUT/api-health-after-deploy.json')); print(d.get('status'), d.get('model_name'), d.get('document_reader'), d.get('native_implementation',{}).get('fingerprint_sha256'))" \
  > "$OUT/api-health-summary.txt" 2>&1 || echo "API health unavailable" > "$OUT/api-health-summary.txt"
say "API: $(cat "$OUT/api-health-summary.txt")"

say "fusion-fix confirmation on video 5 and video 4"
STAMP=20261018-fusion-v5 VIDEO=$V5 REF_A=accuracy-ab-1 REF_B=phase1-speed bash /root/sever-ab/ab_speed_test.sh || say "WARNING: fusion v5 A/B failed"
STAMP=20261018-fusion-v4 VIDEO=$V4 REF_A=accuracy-ab-1 REF_B=phase1-speed bash /root/sever-ab/ab_speed_test.sh || say "WARNING: fusion v4 A/B failed"

say "MTP test: waiting for an idle API"
while [ "$(active_jobs)" -gt 0 ]; do sleep 60; done

restore() {
  set +e
  say "restoring the model service without MTP"
  rm -f "$DROPIN"
  systemctl daemon-reload
  systemctl restart sever-model
  if wait_model; then systemctl start sever-tsl; sleep 5; curl -sf "$API_URL/health" >/dev/null && say "API back up" || say "WARNING: API health failed"
  else say "WARNING: model did not come back; sever-tsl left stopped"; fi
  chown -R sever:sever "$OUT" 2>/dev/null
}
trap restore EXIT

systemctl stop sever-tsl
ORIGINAL=$(systemctl show sever-model -p ExecStart --value | sed -n 's/.*argv\[\]=\(.*\) ; ignore_errors.*/\1/p')
[ -n "$ORIGINAL" ] || { say "STOP: cannot read ExecStart"; exit 1; }
install -d -m 0755 "$DROPIN_DIR"
# Rebuild the exact unit line (systemd expands ${VAR} at start) and append the MTP flags.
LINE=$(grep '^ExecStart=' /etc/systemd/system/sever-model.service | tail -1 | sed 's/^ExecStart=//')
printf '[Service]\nExecStart=\nExecStart=%s --spec-type draft-mtp --spec-draft-n-max %s\n' "$LINE" "$MTP_N" > "$DROPIN"
cp "$DROPIN" "$OUT/mtp-dropin.conf"
systemctl daemon-reload
systemctl restart sever-model
if ! wait_model; then say "STOP: model with MTP did not become healthy"; journalctl -u sever-model --since "-10min" --no-pager -o cat | tail -40 > "$OUT/mtp-startup-failure.log"; exit 1; fi
journalctl -u sever-model --since "-10min" --no-pager -o cat | grep -i -E "spec|draft|mtp|nextn|unused tensor" | head -40 > "$OUT/mtp-startup.log" || true
say "MTP startup log: $(wc -l < "$OUT/mtp-startup.log") relevant lines"

mapfile -t RUN_ENV < <(sed 's/^export //' /etc/sever-tsl/local-provider.env | grep -E '^(HARNESS|SEVER)_[A-Z0-9_]+=' \
  | sed -E -e 's/^([A-Z0-9_]+)="(.*)"$/\1=\2/' -e "s/^([A-Z0-9_]+)='(.*)'\$/\1=\2/")
REL=$(readlink -f /opt/sever-tsl/current)
say "reading video 5 with MTP on ($REL)"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$OUT/gpu-mtp-v5.csv" 2>/dev/null & gpu=$!
start=$(date -u +%s); status=0
( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
    SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
    PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 PYTHONDONTWRITEBYTECODE=1 \
    "$REL/.venv/bin/sever" process "$V5" --out "$OUT/mtp-on-v5" --model qwen38-27b-q6 \
    --max-requests 4000 --max-tokens 40000000 ) > "$OUT/run-mtp-v5.log" 2>&1 || status=$?
kill "$gpu" 2>/dev/null || true
printf '{"run":"mtp-on-v5","exit_status":%d,"wall_seconds":%d,"spec_draft_n_max":%s}\n' "$status" "$(( $(date -u +%s) - start ))" "$MTP_N" > "$OUT/run-mtp-v5.json"
BASE=/srv/sever/validation/ab-speed-20261018-fusion-v5/phase1-speed
if [ -f "$BASE/results.json" ] && [ -f "$OUT/mtp-on-v5/results.json" ]; then
  "$REL/.venv/bin/python" "$REL/tools/compare_runs.py" --a "$BASE" --label-a mtp-off --b "$OUT/mtp-on-v5" --label-b mtp-on \
    --truth "$REL/analysis/ground-truth-expanded-provisional.json" --matching bracket \
    --out "$OUT/comparison-mtp-v5.json" --markdown "$OUT/comparison-mtp-v5.md" >/dev/null || say "WARNING: MTP comparison failed"
fi
say "queue finished: $OUT"
