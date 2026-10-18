#!/usr/bin/env bash
# Reread one customer job in place with the current production release and the
# portal's own job settings (sample interval 0.5 s, up to 300 sheets, fresh cache).
# Previous artifacts go to <job>/history/ (the pipeline archives results and reports;
# an incompatible frame analysis is moved there too). The source upload is unchanged.
# On success the job is marked finished in the API job state and in the portal.
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
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
resume() { systemctl kill --signal=SIGCONT sever-round2-eval-mtp 2>/dev/null || true; }
trap resume EXIT
systemctl kill --signal=SIGSTOP sever-round2-eval-mtp 2>/dev/null || true
# A frame analysis made by an older algorithm or with other settings cannot be reused.
if [ -f "$JOB/video/video-analysis.json" ] && ! python3 - "$JOB/video/video-analysis.json" <<'PY'
import json, sys
signature = json.load(open(sys.argv[1])).get("signature", {})
ok = signature.get("algorithm_version") == 4 and signature.get("sample_interval") == 0.5 and signature.get("max_pages") == 300
sys.exit(0 if ok else 1)
PY
then
  HIST="$JOB/history/reread-video-$(date -u +%s)"
  mkdir -p "$HIST"
  for d in video additional-frames image-readings image-reading-evidence; do [ -e "$JOB/$d" ] && mv "$JOB/$d" "$HIST/"; done
  chown -R sever:sever "$JOB/history"
  say "moved the old frame analysis to $HIST"
fi
say "rereading $JOB_ID from $SRC with $REL"
start=$(date -u +%s)
( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
    SEVER_RUNS_DIR="$RUNS_DIR" SEVER_INFERENCE_GATE_PATH="$RUNS_DIR/.inference-evaluation.lock" "${RUN_ENV[@]}" \
    PYDANTIC_DISABLE_PLUGINS=__all__ PYDANTIC_AI_NO_BANNER=1 PYTHONDONTWRITEBYTECODE=1 \
    "$REL/.venv/bin/sever" process "$SRC" --out "$JOB" --model qwen38-27b-q6 \
    --interval 0.5 --max-pages 300 --refresh-cache \
    --max-requests "${MAX_REQUESTS:-4000}" --max-tokens 40000000 ) >> "$LOG" 2>&1
say "read in $(( $(date -u +%s) - start )) s"
( cd /srv/sever && runuser -u sever -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/srv/sever \
    SEVER_RUNS_DIR="$RUNS_DIR" PORTAL_DATA_DIR=/srv/sever/customer PYDANTIC_DISABLE_PLUGINS=__all__ \
    "$REL/.venv/bin/python" - "$JOB_ID" "$REL" <<'PY'
import json, sys
from datetime import datetime, timezone
job_id, release = sys.argv[1], sys.argv[2]
from sever import api, portal
result = json.loads((api.ROOT / job_id / "results.json").read_text(encoding="utf-8"))
failed = sum(bool(page["result"]["errors"]) for page in result["pages"])
status = "completed_with_errors" if failed else "completed"
api.state(job_id, {"status": status, "page_count": len(result["pages"]), "pages_with_errors": failed,
                   "usage": result["usage"], "error_type": None, "interrupted_status": None,
                   "results_url": f"/v1/jobs/{job_id}/results", "csv_url": f"/v1/jobs/{job_id}/fields.csv",
                   "documents_csv_url": f"/v1/jobs/{job_id}/documents.csv",
                   "reread": {"release": release, "finished_at": datetime.now(timezone.utc).isoformat()}})
with portal.database() as db:
    rows = [row["id"] for row in db.execute("SELECT id FROM jobs WHERE job_id=?", (job_id,))]
for row in rows:
    portal.update(row, status=status, progress="Обработка завершена", error=None)
print(json.dumps({"job_id": job_id, "status": status, "portal_rows": rows, "sheets": len(result["pages"])}))
PY
) >> "$LOG" 2>&1
say "job marked finished"
