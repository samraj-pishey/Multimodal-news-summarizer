from pathlib import Path
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "outputs" / "full_run_v3"
TRANSCRIPTS = PROJECT_ROOT / "outputs" / "audio" / "whisper_transcripts.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "kaggle_bundle" / "audio_inputs.jsonl"
META_PATH = PROJECT_ROOT / "outputs" / "audio" / "audio_inputs_meta.json"

# All fixed before any audio experiment was run.
SEED = 42
MAX_SECONDS = 90            # matches transcribe_audio.py; older full-length records are cut here
MAX_WORDS = 100             # keeps the transcript to ~130 BART tokens
MIN_WORDS = 5               # shorter transcripts are music, applause or stray words
NO_SPEECH_MAX = 0.6         # Whisper's own estimate that a segment is not speech
LOGPROB_MIN = -1.0          # very low-confidence segments are where Whisper hallucinates
NULL_PERCENTILE = 99        # relevant = more similar to its own article than 99% of mismatched pairs


def clean_transcript(record):
    keep = [s["text"].strip() for s in record.get("segments", [])
            if s["start"] < MAX_SECONDS and s["no_speech_prob"] < NO_SPEECH_MAX and s["avg_logprob"] > LOGPROB_MIN]
    words = " ".join(keep).split()
    return " ".join(words[:MAX_WORDS]) if len(words) >= MIN_WORDS else ""


def main():
    articles, split_of = {}, {}
    for split in ("train", "val", "test"):
        for line in open(CLEAN_DIR / f"clean_{split}.jsonl"):
            row = json.loads(line)
            if row["source"] == "dailymail":
                articles[row["id"]] = row["title"] + " " + row["article"]
                split_of[row["id"]] = split
    transcripts = {r["id"]: clean_transcript(r) for r in map(json.loads, open(TRANSCRIPTS))}
    ids = [i for i in transcripts if transcripts[i]]

    # The vocabulary and the threshold both come from training stories only.
    train_ids = [i for i in ids if split_of[i] == "train"]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(
        [articles[i] for i in split_of if split_of[i] == "train"])
    relevance = dict(zip(ids, np.asarray(vec.transform([transcripts[i] for i in ids])
                                         .multiply(vec.transform([articles[i] for i in ids])).sum(1)).ravel()))
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(train_ids)
    null = [float(vec.transform([transcripts[a]]).multiply(vec.transform([articles[b]])).sum())
            for a, b in zip(train_ids, shuffled) if a != b]
    threshold = float(np.percentile(null, NULL_PERCENTILE))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in transcripts:
            rel = float(relevance.get(i, 0.0))
            f.write(json.dumps({"id": i, "transcript_all": transcripts[i],
                                "transcript_relevant": transcripts[i] if rel > threshold else "",
                                "relevance": round(rel, 4)}, ensure_ascii=False) + "\n")

    by_split = {s: {"with_audio": sum(split_of[i] == s for i in transcripts),
                    "with_speech": sum(split_of[i] == s for i in ids),
                    "relevant": int(sum(split_of[i] == s and relevance[i] > threshold for i in ids))}
                for s in ("train", "val", "test")}
    meta = {"threshold": round(threshold, 4), "null_pairs": len(null), "counts": by_split,
            "settings": {"max_seconds": MAX_SECONDS, "max_words": MAX_WORDS, "min_words": MIN_WORDS,
                         "no_speech_max": NO_SPEECH_MAX, "logprob_min": LOGPROB_MIN,
                         "null_percentile": NULL_PERCENTILE}}
    META_PATH.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
