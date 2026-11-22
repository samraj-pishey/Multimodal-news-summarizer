#!/bin/sh
# Text-only M2SM with frozen RoBERTa sentence vectors in place of GloVe + word BiLSTM.
# Compared with text_only_lr0.01 on validation only, so no run here touches test.
# Safe to rerun: finished runs are skipped and a stopped run resumes.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

run() {
  name=$1; shift
  [ -f "outputs/m2sm/runs/$name/results.json" ] && { echo "$name already finished"; return; }
  echo "=== $name $(date)"
  $PY src/m2sm_train.py --name "$name" --threads 6 --no-test --sentence-vectors "$@" || echo "$name FAILED"
}

run text_roberta_lr0.01 --no-video --lr 0.01
run text_roberta_lr0.001 --no-video --lr 0.001
echo "Vectors queue finished $(date)"
