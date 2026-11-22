#!/bin/sh
# Ablation of Table 5 on the 1-per-second features. Safe to rerun: finished runs are skipped and a stopped run resumes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

while pgrep -f "m2sm_train.py --name text_only" >/dev/null; do sleep 60; done

run() {
  name=$1; shift
  [ -f "outputs/m2sm/runs/$name/results.json" ] && { echo "$name already finished"; return; }
  echo "=== $name $(date)"
  $PY src/m2sm_train.py --name "$name" --threads 6 "$@" || echo "$name FAILED"
}

run video_early --no-transcript --no-bistream --fusion early
run video_transcript_early --no-bistream --fusion early
run m2sm_full
echo "Queue finished $(date)"
