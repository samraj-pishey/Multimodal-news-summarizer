"""Scores the video-only runs on test, beside first-k sentences of the speech, the oracle and article-based models.

first-k uses the k that does best on validation. Article-based models are scored on the same stories, to show what is
lost without the article. Writes outputs/m2sm/speech_results.json.
"""
import json
import statistics

import numpy as np
import torch

from m2sm_data import CACHE_PATH, PROJECT_ROOT, SPEECH_CACHE_PATH, score_summary

RUNS = ("speech_text", "speech_text_tuned", "speech_video", "speech_video_tuned")
ARTICLE_RUNS = ("text_only_lr0.01", "m2sm_full_lr0.01")
CHOSEN = "speech_video_tuned"      # fixed from validation before any of these test scores were seen
METRICS = ("r1", "r2", "rl")
SIZES = (2, 3, 4, 5)
OUT_DIR = PROJECT_ROOT / "outputs" / "m2sm"


def rows_for(stories, pick):
    rows = []
    for story in stories:
        result = score_summary(story["reference"], pick(story))
        rows.append([100 * result[m] for m in METRICS])
    return np.array(rows)


def means(rows):
    return dict(zip(METRICS, rows.mean(0).round(2).tolist()))


def top_sentences(story, scores, size):
    return [story["sentences"][i] for i in sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:size])]


def model_rows(run, stories):
    result = json.loads((OUT_DIR / "runs" / run / "results.json").read_text())
    scores = json.loads((OUT_DIR / "runs" / run / "test_scores.json").read_text())
    size = result["summary_size"]
    rows = rows_for(stories, lambda s: top_sentences(s, scores[s["id"]]["scores"], size))
    copied = statistics.mean(
        sorted(sorted(range(len(scores[s["id"]]["scores"])), key=lambda i: -scores[s["id"]]["scores"][i])[:size])
        == list(range(min(size, len(s["sentences"])))) for s in stories)
    return rows, size, 100 * copied


def bootstrap(a, b, seed=42):
    difference = a - b
    picks = np.random.default_rng(seed).integers(0, len(difference), (10000, len(difference)))
    low, high = np.percentile(difference[picks].mean(1), [2.5, 97.5], axis=0)
    return {m: {"difference": round(float(d), 2), "low": round(float(l), 2), "high": round(float(h), 2)}
            for m, d, l, h in zip(METRICS, difference.mean(0), low, high)}


def main():
    speech = torch.load(SPEECH_CACHE_PATH, weights_only=False)
    articles = {s["id"]: s for s in torch.load(CACHE_PATH, weights_only=False)["stories"]}
    split = lambda name: [s for s in speech if s["split_paper"] == name]
    val, test = split("val"), split("test")

    first_val = {k: rows_for(val, lambda s, k=k: s["sentences"][:k]).mean() for k in SIZES}
    first_k = max(first_val, key=first_val.get)
    first = rows_for(test, lambda s: s["sentences"][:first_k])
    oracle = rows_for(test, lambda s: [t for t, y in zip(s["sentences"], s["labels"]) if y] or s["sentences"][:1])
    out = {"n_test": len(test), "first_k": first_k, "first": means(first), "oracle": means(oracle), "runs": {}}

    rows = {}
    for run in RUNS:
        rows[run], size, copied = model_rows(run, test)
        out["runs"][run] = {"size": size, "copied_opening_percent": round(copied, 1), **means(rows[run]),
                            "minus_first": bootstrap(rows[run], first)}
    out["chosen"] = CHOSEN
    out["chosen_minus_text"] = bootstrap(rows[CHOSEN], rows["speech_text"])

    same_articles = [articles[s["id"]] for s in test]
    out["article_models_same_stories"] = {}
    for run in ARTICLE_RUNS:
        article_rows, _, _ = model_rows(run, same_articles)
        out["article_models_same_stories"][run] = {**means(article_rows),
                                                   "chosen_minus_this": bootstrap(rows[CHOSEN], article_rows)}
    (OUT_DIR / "speech_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
