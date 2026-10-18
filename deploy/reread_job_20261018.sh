#!/usr/bin/env bash
# Reread one finished customer job in place with the current production release and the
# portal's own job settings (sample interval 0.5 s, up to 300 sheets, fresh cache).
# Previous artifacts are archived by the pipeline under <job>/history/; the source upload is unchanged.
# The internal round-2 evaluation is paused meanwhile so the customer job gets the GPU.
set -euo pipefail
JOB_ID=${JOB_ID:?set JOB_ID}
RUNS_DIR=/srv/sever/runs
JOB="$RUNS_DIR/$JOB_ID"
REL=$(readlink -f /opt/sever-tsl/current)
LOG=/srv/sever/validation/reread-$JOB_ID.log
SRC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['source_path'])" "$JOB/job.json")
[ -f "$SRC" ] || { echo "missing source $SRC" >&2; exit 1; }
mapfile -t RUN_ENV < <(sed 's/^export //' /etc/sever-tsl/local-provider.env | grep -E '^(HARNESS|SEVER)_[A-Z0-9_]+=' \
  | sed -E -e 's/^([A-Z0-9_]+)="(.*)"$/\1=\2/' -e "s/^([A-Z0-9_]+)='(.*)'\$/\1=\2/")
resume() { systemctl kill --signal=SIGCONT sever-round2-eval-mtp 2>/dev/null || true; }
trap resume EXIT
systemctl kill --signal=SIGSTOP sever-round2-eval-mtp 2>/dev/null || true
echo "$(date -u +%H:%M:%S) rereading $JOB_ID from $SRC with $REL" | tee -a "$LOG"
start=$(date -u +%s)
( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
    SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
    PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 PYTHONDONTWRITEBYTECODE=1 \
    "$REL/.venv/bin/sever" process "$SRC" --out "$JOB" --model qwen38-27b-q6 \
    --interval 0.5 --max-pages 300 --refresh-cache \
    --max-requests "${MAX_REQUESTS:-4000}" --max-tokens 40000000 ) >> "$LOG" 2>&1
echo "$(date -u +%H:%M:%S) done in $(( $(date -u +%s) - start )) s" | tee -a "$LOG"
