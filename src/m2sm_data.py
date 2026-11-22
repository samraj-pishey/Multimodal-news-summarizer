"""Stories, vocabulary, labels and splits for the M2SM replication (Fu et al. 2020).

Two train/val regimes share the authors' official Daily Mail test ids:
  paper: all remaining stories, 7:1 train/val. Near-duplicate copies of test stories can be in train.
  clean: outputs/splits.csv, which groups near-duplicates so no story crosses a split.
CNN is test-only, following the paper ("we mainly use them during testing").
"""
from collections import Counter
from pathlib import Path
import csv
import json
import random
import re

import numpy as np
import torch
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
FEATURE_DIR = PROJECT_ROOT / "data" / "m2sm_features"
GLOVE_PATH = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"
CACHE_PATH = PROJECT_ROOT / "data" / "m2sm_cache.pt"
WHISPER_PATH = PROJECT_ROOT / "outputs" / "audio" / "whisper_transcripts.jsonl"
SPEECH_CACHE_PATH = PROJECT_ROOT / "data" / "m2sm_speech.pt"
VECTOR_PATHS = {"article": PROJECT_ROOT / "data" / "m2sm_vectors_article.pt",       # m2sm_sentence_vectors.py
                "speech": PROJECT_ROOT / "data" / "m2sm_vectors_speech.pt"}

MAX_SENTENCES, MAX_SENTENCE_WORDS, MAX_TRANSCRIPT_WORDS = 100, 60, 300
MIN_WORD_COUNT = 2
MIN_SPEECH_SENTENCE_WORDS, MIN_SPEECH_WORDS = 3, 5
NO_SPEECH_MAX, LOGPROB_MIN = 0.6, -1.0
PAD, UNK = 0, 1
SPLIT_SEED = 42

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", text.lower())


def read(path):
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def read_transcript(path):
    """transcript.txt alternates timestamps with text in Daily Mail; CNN holds plain text."""
    lines = [line.strip() for line in read(path).splitlines() if line.strip()]
    return " ".join(line for line in lines if not re.fullmatch(r"[\d.:,]+", line))


def score_summary(reference, sentences):
    """ROUGE F1. Summary-level R-L matches the paper's Perl ROUGE (lead3 30.73 vs its 30.90)."""
    scores = scorer.score(reference, "\n".join(sentences))
    return {"r1": scores["rouge1"].fmeasure, "r2": scores["rouge2"].fmeasure, "rl": scores["rougeLsum"].fmeasure}


def greedy_labels(sentences, reference, limit=10):
    """Greedy ROUGE oracle (Nallapati et al. 2017). Used for CNN, where the authors ship no labels."""
    chosen, best = [], 0.0
    for _ in range(limit):
        options = [(scorer.score(reference, " ".join(sentences[j] for j in sorted(chosen + [i])))["rouge2"].fmeasure, i)
                   for i in range(len(sentences)) if i not in chosen]
        if not options or max(options)[0] <= best:
            break
        best, pick = max(options)
        chosen.append(pick)
    return [float(i in chosen) for i in range(len(sentences))]


def speech_sentences(text):
    """Whisper's punctuation gives the sentences; very short fragments are dropped. Same rule as the web app."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= MIN_SPEECH_SENTENCE_WORDS]


def read_whisper():
    """Text of each story's confident Whisper segments, with the thresholds used in make_audio_inputs.py."""
    texts = {}
    with open(WHISPER_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            kept = [s["text"] for s in row.get("segments", [])
                    if s["no_speech_prob"] <= NO_SPEECH_MAX and s["avg_logprob"] >= LOGPROB_MIN]
            texts[row["id"]] = " ".join(t.strip() for t in kept)
    return texts


def speech_stories(stories, vocabulary):
    """Copies of the stories with usable speech, whose Whisper sentences replace the article (video-only mode)."""
    if SPEECH_CACHE_PATH.exists():
        return torch.load(SPEECH_CACHE_PATH, weights_only=False)
    whisper = read_whisper()
    rows = []
    for story in stories:
        sentences = speech_sentences(whisper.get(story["id"], ""))[:MAX_SENTENCES]
        if sum(len(s.split()) for s in sentences) < MIN_SPEECH_WORDS:
            continue
        rows.append({**story, "sentences": sentences, "labels": greedy_labels(sentences, story["reference"]),
                     "sentence_ids": [[vocabulary.get(w, UNK) for w in tokenize(s)][:MAX_SENTENCE_WORDS] or [UNK]
                                      for s in sentences], "transcript": " ".join(sentences), "transcript_ids": []})
    torch.save(rows, SPEECH_CACHE_PATH)
    return rows


def load_authors_labels():
    folder = DATA_ROOT / "daily mail"
    ids = sorted((p.name for p in folder.iterdir() if p.is_dir() and p.name.isdigit()), key=int)
    arrays = re.findall(r"\[([^\]]*)\]", read(folder / "labels.txt"))
    return {i: [float(x) for x in a.split()] for i, a in zip(ids, arrays)}


def load_raw_stories():
    labels = load_authors_labels()
    official = set(read(DATA_ROOT / "test_id_dm.txt").strip().split(","))
    stories, dropped = [], Counter()

    for folder in sorted((DATA_ROOT / "daily mail").iterdir()):
        if not (folder.is_dir() and folder.name.isdigit()):
            continue
        lines = read(folder / "article_segment.txt").splitlines()
        sentences = [re.sub(r"^<s>", "", line).strip() for line in lines if line.strip()]
        reference = read(folder / "highlight.txt").strip()
        if not sentences or not reference:
            dropped["dailymail: no text"] += 1
            continue
        if len(labels.get(folder.name, [])) != len(sentences):
            dropped["dailymail: label length differs"] += 1
            if folder.name not in official:
                continue
            label = [0.0] * len(sentences)        # test labels are never used for scoring
        else:
            label = labels[folder.name]
        stories.append({"id": f"dailymail/{folder.name}", "source": "dailymail", "sentences": sentences,
                        "reference": reference, "labels": label, "official_test": folder.name in official,
                        "transcript": read_transcript(folder / "transcript.txt")})

    for folder in sorted((DATA_ROOT / "cnn_data").iterdir()):
        if not (folder.is_dir() and folder.name.isdigit()):
            continue
        # One line of artitle_section.txt is one unit: its mean of 30.5 matches the paper's 29.84.
        sentences = [line.strip() for line in read(folder / "artitle_section.txt").splitlines() if line.strip()]
        reference = read(folder / "highlight.txt").strip()
        if not sentences or not reference:
            dropped["cnn: no text"] += 1
            continue
        stories.append({"id": f"cnn/{folder.name}", "source": "cnn", "sentences": sentences, "reference": reference,
                        "labels": [0.0] * len(sentences), "official_test": False,
                        "transcript": read_transcript(folder / "transcript.txt")})
    return stories, dropped


def feature_path(story_id):
    source, number = story_id.split("/")
    return FEATURE_DIR / f"{source}_{number}.npz"


def assign_splits(stories):
    """Adds story['split_paper'] and story['split_clean'], each train, val, test or None."""
    rows = {r["id"]: r for r in csv.DictReader(open(PROJECT_ROOT / "outputs" / "splits.csv"))}
    remaining = sorted((s["id"] for s in stories if s["source"] == "dailymail" and not s["official_test"]),
                       key=lambda i: int(i.split("/")[1]))
    random.Random(SPLIT_SEED).shuffle(remaining)
    val_ids = set(remaining[:len(remaining) // 8])

    for story in stories:
        if story["source"] == "cnn":
            story["split_paper"] = story["split_clean"] = "test"
            continue
        row = rows.get(story["id"])
        representative = bool(row) and row["is_group_rep"] == "True"
        if story["official_test"]:
            story["split_paper"] = "test"
            story["split_clean"] = "test" if representative else None
        else:
            story["split_paper"] = "val" if story["id"] in val_ids else "train"
            story["split_clean"] = row["split"] if representative and row["split"] != "test" else None


def build_vocabulary(train_stories):
    counts = Counter()
    for story in train_stories:
        for sentence in story["sentences"]:
            counts.update(tokenize(sentence))
        counts.update(tokenize(story["transcript"]))
    glove = {}
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            word, _, values = line.partition(" ")
            if counts.get(word, 0) >= MIN_WORD_COUNT:
                glove[word] = np.array(values.split(), dtype=np.float32)
    words = sorted(glove, key=lambda w: (-counts[w], w))
    matrix = np.zeros((len(words) + 2, 100), np.float32)
    matrix[UNK] = np.random.default_rng(SPLIT_SEED).normal(0, 0.1, 100)
    for i, word in enumerate(words, 2):
        matrix[i] = glove[word]
    return {w: i for i, w in enumerate(words, 2)}, torch.from_numpy(matrix)


def build_cache():
    stories, dropped = load_raw_stories()
    with_video = []
    for story in stories:
        path = feature_path(story["id"])
        if path.exists() and np.load(path)["frames"].shape[0] > 0:
            with_video.append(story)
        else:
            dropped[f"{story['source']}: no video features"] += 1
    stories = with_video
    assign_splits(stories)

    for story in stories:
        if story["source"] == "cnn":
            story["labels"] = greedy_labels(story["sentences"], story["reference"])

    # The vocabulary comes from the paper-regime train split, the larger of the two.
    vocabulary, embeddings = build_vocabulary([s for s in stories if s["split_paper"] == "train"])
    encode = lambda text, limit: [vocabulary.get(w, UNK) for w in tokenize(text)][:limit] or [UNK]
    for story in stories:
        story["sentences"] = story["sentences"][:MAX_SENTENCES]
        story["labels"] = story["labels"][:MAX_SENTENCES]
        story["sentence_ids"] = [encode(s, MAX_SENTENCE_WORDS) for s in story["sentences"]]
        story["transcript_ids"] = encode(story["transcript"], MAX_TRANSCRIPT_WORDS) if story["transcript"] else []

    torch.save({"stories": stories, "embeddings": embeddings, "vocabulary": vocabulary}, CACHE_PATH)
    summary = {"stories": len(stories), "vocabulary": len(vocabulary), "dropped": dict(dropped),
               "with_transcript": sum(bool(s["transcript_ids"]) for s in stories)}
    for regime in ("split_paper", "split_clean"):
        summary[regime] = dict(Counter(f"{s['source']}:{s[regime]}" for s in stories))
    print(json.dumps(summary, indent=1))
    (PROJECT_ROOT / "outputs" / "m2sm").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "m2sm" / "data_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    build_cache()
