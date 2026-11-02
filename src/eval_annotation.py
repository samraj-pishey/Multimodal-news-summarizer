from pathlib import Path
import csv
import json

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = PROJECT_ROOT / "outputs" / "annotation" / "labels"
SAMPLE = PROJECT_ROOT / "outputs" / "annotation" / "annotation_sample.json"
CLIP_SCORES = PROJECT_ROOT / "outputs" / "clip" / "clip_scores.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "annotation" / "cover_image_results.json"

SEED = 42
N_BOOT = 10000


def load_labels():
    """(story, person, shard, choice) rows. Shards labelled by the same person are one annotator."""
    rows = []
    for path in sorted(LABEL_DIR.glob("cover_labels_annotator_*.csv")):
        for r in csv.DictReader(open(path)):
            if r["choice"]:
                rows.append((r["story_id"], r["annotator"].strip().lower(), r["shard"], r["choice"]))
    return rows


def pick(images, key):
    return max(images, key=lambda im: im[key])["file"]


def main():
    clip = {r["id"]: r for r in map(json.loads, open(CLIP_SCORES))}
    labels = load_labels()
    methods = {
        "clip_vs_finetuned_summary": lambda imgs: pick(imgs, "sim_finetuned"),
        "clip_vs_article": lambda imgs: pick(imgs, "sim_article"),
        "first_image": lambda imgs: imgs[0]["file"],
    }

    # Precision@1 over every (story, label) pair where the annotator picked an image.
    usable = [(s, c) for s, _, _, c in labels if c != "none"]
    hits = {m: np.array([f(clip[s]["images"]) == c for s, c in usable], dtype=float) for m, f in methods.items()}
    hits["random"] = np.array([1 / len(clip[s]["images"]) for s, _ in usable])
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(usable), len(usable)) for _ in range(N_BOOT)]
    precision = {m: {"p_at_1": round(float(h.mean()), 4),
                     "ci95": [round(float(np.percentile([h[i].mean() for i in idx], q)), 4) for q in (2.5, 97.5)]}
                 for m, h in hits.items()}
    d = hits["clip_vs_finetuned_summary"] - hits["first_image"]
    p_clip_beats_first = float(np.mean([d[i].mean() > 0 for i in idx]))

    # Agreement only between two *different* people on the same story.
    by_story = {}
    for s, person, shard, c in labels:
        by_story.setdefault(s, []).append((person, c))
    pairs = [(v[0][1], v[1][1], len(clip[s]["images"])) for s, v in by_story.items()
             if len(v) == 2 and v[0][0] != v[1][0]]
    same_person = sum(1 for v in by_story.values() if len(v) == 2 and v[0][0] == v[1][0])
    agree = float(np.mean([a == b for a, b, _ in pairs])) if pairs else None
    chance = float(np.mean([1 / (n + 1) for _, _, n in pairs])) if pairs else None    # n images + "none"

    result = {
        "labels": len(labels), "stories": len(by_story), "people": sorted({p for _, p, _, _ in labels}),
        "none_choices": sum(c == "none" for *_, c in labels), "labels_used_for_precision": len(usable),
        "precision_at_1": precision, "p_clip_summary_beats_first_image": round(p_clip_beats_first, 4),
        "agreement": {"pairs_different_people": len(pairs), "pairs_same_person_excluded": same_person,
                      "raw_agreement": agree, "chance_agreement": chance,
                      "kappa_like": round((agree - chance) / (1 - chance), 3) if pairs else None},
    }
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
