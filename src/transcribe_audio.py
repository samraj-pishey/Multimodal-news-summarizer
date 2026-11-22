from pathlib import Path
import csv
import json
import time

import mlx_whisper
import numpy as np
import soundfile as sf
import soxr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
SPLITS_PATH = PROJECT_ROOT / "outputs" / "splits.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "audio" / "whisper_transcripts.jsonl"

MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
# Only the first 90 s of each video are transcribed.
MAX_SECONDS = 90


def load_audio(path):
    # soundfile decodes mp3 directly, so ffmpeg is not needed.
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = soxr.resample(audio.mean(1), sr, SAMPLE_RATE).astype(np.float32)
    return audio[:MAX_SECONDS * SAMPLE_RATE]


def pending_stories(done):
    # Test first, so partial runs are already usable for evaluation.
    rows = list(csv.DictReader(open(SPLITS_PATH)))
    for split in ("test", "val", "train"):
        for row in rows:
            if row["split"] != split or row["source"] != "dailymail" or row["id"] in done:
                continue
            path = DATA_ROOT / "daily mail" / row["id"].split("/")[1] / "audio.mp3"
            if path.exists():
                yield split, row["id"], path


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(OUT_PATH)} if OUT_PATH.exists() else set()
    todo = list(pending_stories(done))
    print(f"{len(done)} already transcribed, {len(todo)} to go")

    t0, audio_seconds = time.time(), 0.0
    with open(OUT_PATH, "a") as out:
        for n, (split, story_id, path) in enumerate(todo, 1):
            record = {"id": story_id, "split": split}
            try:
                audio = load_audio(path)
                # Without conditioning, a hallucinated segment can't propagate into the next one.
                result = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL_NAME, language="en",
                                                condition_on_previous_text=False)
                record.update(seconds=round(len(audio) / SAMPLE_RATE, 1), text=result["text"].strip(),
                              segments=[{k: round(s[k], 3) if isinstance(s[k], float) else s[k]
                                         for k in ("start", "end", "text", "avg_logprob", "no_speech_prob")}
                                        for s in result["segments"]])
                audio_seconds += record["seconds"]
            except Exception as e:      # a few files are corrupt; record and move on
                record["error"] = repr(e)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                mins = (time.time() - t0) / 60
                print(f"  {n}/{len(todo)} | {mins:.0f} min | {audio_seconds / 60 / mins:.1f}x realtime", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.0f} min -> {OUT_PATH}")


if __name__ == "__main__":
    main()
