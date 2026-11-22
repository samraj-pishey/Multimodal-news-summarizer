"""Inputs for the hybrid model (M2SM + A2Summ): frame labels and speech sentence times.

Frame labels follow A2Summ's Daily Mail annotation (one key frame per reference image): for each reference image,
the frame with the highest ResNet cosine is marked. Labels are used for training only; validation and test are
still scored with the Cos metric against the images themselves.

Speech sentence times come from the Whisper segments behind each sentence, so speech can be aligned with frames.
The sentences are rebuilt with the same rule as speech_stories() and checked against that cache.

Usage: python src/hybrid_data.py
"""
import json
import re

import numpy as np
import torch

from m2sm_data import (CACHE_PATH, LOGPROB_MIN, MIN_SPEECH_SENTENCE_WORDS, NO_SPEECH_MAX, PROJECT_ROOT,
                       SPEECH_CACHE_PATH, WHISPER_PATH, feature_path)

HYBRID_PATH = PROJECT_ROOT / "data" / "hybrid_inputs.pt"


def frame_labels(frames, images):
    """One key frame per reference image: the frame most similar to it."""
    if len(images) == 0:
        return None
    frames = frames / np.linalg.norm(frames, axis=1, keepdims=True).clip(min=1e-8)
    images = images / np.linalg.norm(images, axis=1, keepdims=True).clip(min=1e-8)
    labels = np.zeros(len(frames), dtype=np.float32)
    labels[(images @ frames.T).argmax(1)] = 1
    return labels


def sentence_times(segments):
    """Sentences of the confident segments, each with (start, end) seconds of the segments it overlaps."""
    kept = [s for s in segments if s["no_speech_prob"] <= NO_SPEECH_MAX and s["avg_logprob"] >= LOGPROB_MIN]
    pieces, spans, offset = [], [], 0
    for segment in kept:
        text = segment["text"].strip()
        spans.append((offset, offset + len(text), segment["start"], segment["end"]))
        pieces.append(text)
        offset += len(text) + 1
    joined = " ".join(pieces)

    sentences, times, cursor = [], [], 0
    for part in re.split(r"(?<=[.!?])\s+", joined):
        start = joined.find(part, cursor)
        cursor = start + len(part)
        if len(part.strip().split()) < MIN_SPEECH_SENTENCE_WORDS:
            continue
        touched = [(a, b) for lo, hi, a, b in spans if lo < cursor and hi > start]
        sentences.append(part.strip())
        times.append((min(a for a, _ in touched), max(b for _, b in touched)))
    return sentences, times


def main():
    stories = torch.load(CACHE_PATH, weights_only=False)["stories"]
    speech = {s["id"]: s for s in torch.load(SPEECH_CACHE_PATH, weights_only=False)}
    segments = {}
    with open(WHISPER_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            segments[row["id"]] = row.get("segments", [])

    inputs, mismatched = {}, 0
    for story in stories:
        data = np.load(feature_path(story["id"]))
        entry = {"frame_times": data["times"].astype(np.float32),
                 "frame_labels": frame_labels(data["frames"].astype(np.float32), data["images"].astype(np.float32))}
        if story["id"] in speech:
            sentences, times = sentence_times(segments[story["id"]])
            cached = speech[story["id"]]["sentences"]
            if sentences[:len(cached)] != cached:
                mismatched += 1
                continue
            entry["speech_times"] = np.array(times[:len(cached)], dtype=np.float32)
        inputs[story["id"]] = entry

    torch.save(inputs, HYBRID_PATH)
    labelled = sum(e["frame_labels"] is not None for e in inputs.values())
    with_speech = sum("speech_times" in e for e in inputs.values())
    keys = [int(e["frame_labels"].sum()) for e in inputs.values() if e["frame_labels"] is not None]
    print(f"{len(inputs)} stories, {labelled} with frame labels (mean {np.mean(keys):.2f} key frames), "
          f"{with_speech} with timed speech, {mismatched} speech stories whose sentences didn't match")


if __name__ == "__main__":
    main()
