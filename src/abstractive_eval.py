"""Abstractive summaries from BART-large-CNN (pretrained on CNN/Daily Mail highlights, used without retraining).

The decoding setting (greedy or beam search) is chosen on validation stories only; test is then scored once.
Also records how abstractive each summary is: the share of its word bigrams that never appear in the article.
Resumable: summaries are cached per split and setting in outputs/abstractive/.

Usage: python src/abstractive_eval.py --split val --beams 1 4 --limit 60
       python src/abstractive_eval.py --split test --beams 1
"""
import argparse
import json
import re
import statistics
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from m2sm_data import CACHE_PATH, PROJECT_ROOT, score_summary

MODEL_NAME = "facebook/bart-large-cnn"
OUT_DIR = PROJECT_ROOT / "outputs" / "abstractive"


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def sentences(text):
    """Summary-level ROUGE-L (rougeLsum) compares sentence by sentence, so split the generated text first."""
    return [s for s in re.split(r"(?<=[.!?'\"’])\s+(?=[A-Z0-9'\"‘“])", text) if s.strip()]


def scores(reference, text):
    return {k: round(100 * v, 2) for k, v in score_summary(reference, sentences(text)).items()}


def novel_bigrams(summary, article):
    """Share of the summary's bigrams that are not in the article (0 = copied, higher = more rewritten)."""
    pairs = lambda w: set(zip(w, w[1:]))
    summary_pairs = pairs(words(summary))
    return len(summary_pairs - pairs(words(article))) / max(1, len(summary_pairs))


def summarize(model, tokenizer, article, beams):
    inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        output = model.generate(**inputs, num_beams=beams, max_length=142, min_length=56, no_repeat_ngram_size=3)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--beams", type=int, nargs="+", default=[1])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True).eval()
    stories = sorted([s for s in torch.load(CACHE_PATH, weights_only=False)["stories"]
                      if s["split_paper"] == args.split and s["source"] == "dailymail"], key=lambda s: s["id"])
    stories = stories[:args.limit or None]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {}
    for beams in args.beams:
        path = OUT_DIR / f"{args.split}_beams{beams}.json"
        done = json.loads(path.read_text()) if path.exists() else {}
        began = time.time()
        for n, story in enumerate(stories, 1):
            article = " ".join(story["sentences"])
            if story["id"] in done:        # cached text; scores are recomputed so scoring fixes apply
                done[story["id"]].update(scores(story["reference"], done[story["id"]]["summary"]))
                continue
            text = summarize(model, tokenizer, article, beams)
            done[story["id"]] = {"summary": text, "novel_bigrams": round(novel_bigrams(text, article), 3),
                                 **scores(story["reference"], text)}
            if n % 10 == 0 or n == len(stories):
                path.write_text(json.dumps(done, ensure_ascii=False))
                print(f"  beams {beams}: {n}/{len(stories)} ({(time.time() - began) / n:.1f}s each)", flush=True)
        path.write_text(json.dumps(done, ensure_ascii=False))
        rows = [done[s["id"]] for s in stories]
        report[f"beams{beams}"] = {"n": len(rows), **{k: round(statistics.mean(r[k] for r in rows), 2)
                                                      for k in ("r1", "r2", "rl", "novel_bigrams")}}
    print(json.dumps(report, indent=1))
    (OUT_DIR / f"{args.split}_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
