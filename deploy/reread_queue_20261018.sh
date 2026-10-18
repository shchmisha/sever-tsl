#!/usr/bin/env bash
# Reread customer jobs one after another, after any reread already running.
set -uo pipefail
while systemctl is-active -q sever-reread-tv2; do sleep 20; done
for job in "$@"; do
  JOB_ID="$job" /root/reread_job_v2_20261018.sh || echo "$(date -u +%H:%M:%S) reread failed for $job" >&2
done
