#!/bin/sh
# The full hybrid, then one run with each part switched off. All chosen on validation; each scores test once.
# Safe to rerun: finished runs are skipped and a stopped run resumes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

run() {
  name=$1; shift
  [ -f "outputs/hybrid/runs/$name/results.json" ] && { echo "$name already finished"; return; }
  echo "=== $name $(date)"
  $PY src/hybrid_train.py --name "$name" --threads 6 "$@" || echo "$name FAILED"
}

run hybrid
run hybrid_no_video --no-video
run hybrid_no_speech --no-speech
run hybrid_no_align --no-align
run hybrid_no_contrastive --no-contrastive
run hybrid_no_sentence_lstm --no-sentence-lstm
run hybrid_no_lateplus --fusion none
echo "Hybrid queue finished $(date)"
