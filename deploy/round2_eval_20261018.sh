#!/usr/bin/env bash
# After the 2026-10-18 GPU queue: read the five round-2 labelled videos with the
# production release and model (no service changes), for accuracy on new labels.
set -euo pipefail
OUT=${OUT:-/srv/sever/validation/round2-eval-20261018}
RUNS_DIR=/srv/sever/runs
VIDEOS=${VIDEOS:-"20260821_144838 20260821_145110 20260818-6_131851 20260821_145342 20260818_131702"}
say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
active_jobs() {
  python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get("status") in {"queued", "processing"} for p in Path(sys.argv[1]).glob("*/job.json") if p.is_file()))
PY
}
install -d -o sever -g sever -m 0750 "$OUT"
say "waiting for sever-gpu-queue to finish"
while systemctl is-active -q sever-gpu-queue; do sleep 60; done
REL=$(readlink -f /opt/sever-tsl/current)
echo "$REL $(git -C "$REL" rev-parse HEAD 2>/dev/null)" > "$OUT/release.txt"
mapfile -t RUN_ENV < <(sed 's/^export //' /etc/sever-tsl/local-provider.env | grep -E '^(HARNESS|SEVER)_[A-Z0-9_]+=' \
  | sed -E -e 's/^([A-Z0-9_]+)="(.*)"$/\1=\2/' -e "s/^([A-Z0-9_]+)='(.*)'\$/\1=\2/")
for v in $VIDEOS; do
  while [ "$(active_jobs)" -gt 0 ]; do sleep 60; done
  say "reading $v with $REL"
  start=$(date -u +%s); status=0
  ( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
      SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
      PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 PYTHONDONTWRITEBYTECODE=1 \
      "$REL/.venv/bin/sever" process "/srv/sever/input/$v.mp4" --out "$OUT/$v" --model qwen38-27b-q6 \
      --max-requests 4000 --max-tokens 40000000 ) > "$OUT/run-$v.log" 2>&1 || status=$?
  printf '{"video":"%s","exit_status":%d,"wall_seconds":%d,"api_jobs_active_at_end":%d}\n' "$v" "$status" \
    "$(( $(date -u +%s) - start ))" "$(active_jobs)" > "$OUT/run-$v.json"
done
chown -R sever:sever "$OUT"
say "round-2 runs finished: $OUT"
