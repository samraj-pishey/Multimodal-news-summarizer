#!/bin/sh
# Video-only mode: Whisper sentences replace the article. Chosen on validation only, so no run here touches test.
# Safe to rerun: finished runs are skipped and a stopped run resumes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

run() {
  name=$1; shift
  [ -f "outputs/m2sm/runs/$name/results.json" ] && { echo "$name already finished"; return; }
  echo "=== $name $(date)"
  $PY src/m2sm_train.py --name "$name" --threads 6 --no-test --speech "$@" || echo "$name FAILED"
}

run speech_text --no-video --lr 0.01
run speech_text_tuned --no-video --lr 0.001 --init-from text_only_lr0.01
run speech_video --no-transcript --no-bistream --fusion early --lr 0.01
run speech_video_tuned --no-transcript --no-bistream --fusion early --lr 0.001 --init-from video_early_lr0.01
echo "Speech queue finished $(date)"
