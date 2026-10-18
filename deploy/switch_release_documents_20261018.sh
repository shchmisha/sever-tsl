#!/usr/bin/env bash
# 2026-10-18: switch production to the per-document export release once the API is idle.
# Rolls back to the previous release if the API or portal does not come back healthy.
# Then rebuilds documents.csv/json and report.html for finished customer jobs (old report kept
# under history/), and resumes the paused internal round-2 evaluation.
set -euo pipefail
NEW=${NEW:-/opt/sever-tsl/releases/20261018-documents}
RUNS_DIR=/srv/sever/runs
PORTAL_DB=/srv/sever/customer/portal.sqlite3
LOG=${LOG:-/srv/sever/validation/switch-documents-20261018.log}
say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >&2; }
active_jobs() {
  python3 - "$RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get("status") in {"queued", "processing"} for p in Path(sys.argv[1]).glob("*/job.json") if p.is_file()))
PY
}
healthy() {
  curl -sf --max-time 10 http://127.0.0.1:8080/health >/dev/null && \
  curl -s --max-time 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/ | grep -qE '^(200|302|303|401|403)$'
}
[ -x "$NEW/.venv/bin/sever" ] || { say "missing $NEW"; exit 1; }
OLD=$(readlink -f /opt/sever-tsl/current)
say "waiting for an idle API (current release $OLD)"
while [ "$(active_jobs)" -gt 0 ]; do sleep 30; done
# A job submitted in the last seconds must not be cut: check twice.
sleep 20; while [ "$(active_jobs)" -gt 0 ]; do sleep 30; done
printf 'deployed_at_utc=%s\nprevious_release=%s\ncommit=%s\nnote=per-document export (documents.csv/json, report document table); model and reading code unchanged from 61c7274 except output\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OLD" "$(git -C "$NEW" rev-parse HEAD)" > "$NEW/DEPLOYED.txt"
ln -sfn "$NEW" /opt/sever-tsl/current.next && mv -T /opt/sever-tsl/current.next /opt/sever-tsl/current
systemctl restart sever-tsl sever-portal
ok=0; for i in $(seq 1 12); do sleep 5; if healthy; then ok=1; break; fi; done
if [ "$ok" -ne 1 ]; then
  say "health check failed on $NEW; rolling back to $OLD"
  ln -sfn "$OLD" /opt/sever-tsl/current.next && mv -T /opt/sever-tsl/current.next /opt/sever-tsl/current
  systemctl restart sever-tsl sever-portal
  sleep 10; healthy && say "rolled back, healthy" || say "WARNING: rollback health check failed"
  systemctl kill --signal=SIGCONT sever-round2-eval-mtp 2>/dev/null || true
  exit 1
fi
say "switched to $NEW; API and portal healthy"
curl -s http://127.0.0.1:8080/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("fingerprint", h["native_implementation"]["fingerprint_sha256"])' | tee -a "$LOG"
# Finished customer jobs get the per-document table; results.json is not modified.
python3 - "$PORTAL_DB" <<'PY' > /tmp/portal-jobs.txt
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for (job_id,) in db.execute("SELECT job_id FROM jobs WHERE job_id IS NOT NULL"):
    print(job_id)
PY
rebuilt=0; failed=0
while read -r job; do
  dir="$RUNS_DIR/$job"
  [ -f "$dir/results.json" ] || continue
  if runuser -u sever -- "$NEW/.venv/bin/sever" documents "$dir" >> "$LOG" 2>&1; then rebuilt=$((rebuilt+1)); else failed=$((failed+1)); say "rebuild failed for $job (previous report kept)"; fi
done < /tmp/portal-jobs.txt
say "customer jobs rebuilt: $rebuilt, failed: $failed"
systemctl kill --signal=SIGCONT sever-round2-eval-mtp 2>/dev/null && say "round-2 evaluation resumed" || say "round-2 evaluation not running"
say "done"
