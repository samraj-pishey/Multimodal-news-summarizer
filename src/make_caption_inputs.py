from pathlib import Path
import json
import random
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIP_SCORES = PROJECT_ROOT / "outputs" / "clip" / "clip_scores.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "kaggle_bundle" / "caption_inputs.jsonl"

SEED = 42
TOP_K = 2      # fixed before any caption experiment was run


def rank_by_clip(record):
    """Caption indices, most relevant first.

    A caption aligned to an image inherits that image's CLIP similarity to the
    article; captions with no image follow, ordered by their own text similarity.
    """
    image_sim = {i: img["sim_article"] for i, img in enumerate(record["images"])}

    def key(k):
        cap = record["captions"][k]
        if cap["image"] is not None:
            return (1, image_sim[cap["image"]])
        return (0, cap["sim_article"])

    return sorted(range(len(record["captions"])), key=key, reverse=True)


def main():
    out, n_caps = [], []
    for line in open(CLIP_SCORES):
        r = json.loads(line)
        texts = [c["text"] for c in r["captions"]]
        rng = random.Random(f"{SEED}-{r['id']}")   # per-story seed: reproducible regardless of order
        out.append({
            "id": r["id"],
            "all_captions": texts,
            "clip_top2": [texts[k] for k in rank_by_clip(r)[:TOP_K]],
            "random2": rng.sample(texts, min(TOP_K, len(texts))),
        })
        n_caps.append(len(texts))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    differs = sum(set(r["clip_top2"]) != set(r["random2"]) for r in out)
    print(f"{len(out)} stories | captions per story {statistics.mean(n_caps):.2f} | "
          f"stories where CLIP and random selections differ: {differs}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
