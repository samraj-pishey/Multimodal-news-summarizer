#!/bin/sh
# Higher Adagrad learning rates, after the paper-setting queue. Compare the runs on validation only.
# Safe to rerun: finished runs are skipped and a stopped run resumes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

while pgrep -f "m2sm_queue.sh" >/dev/null; do sleep 60; done

run() {
  name=$1; shift
  [ -f "outputs/m2sm/runs/$name/results.json" ] && { echo "$name already finished"; return; }
  echo "=== $name $(date)"
  $PY src/m2sm_train.py --name "$name" --threads 6 "$@" || echo "$name FAILED"
}

for lr in 0.01 0.001; do
  run "text_only_lr$lr" --no-video --lr "$lr"
  run "m2sm_full_lr$lr" --lr "$lr"
done
for lr in 0.01 0.001; do
  run "video_early_lr$lr" --no-transcript --no-bistream --fusion early --lr "$lr"
  run "video_transcript_early_lr$lr" --no-bistream --fusion early --lr "$lr"
done
echo "LR queue finished $(date)"
