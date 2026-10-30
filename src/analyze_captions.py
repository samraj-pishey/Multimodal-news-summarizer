from pathlib import Path
import csv
import json
import re

import numpy as np
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "outputs" / "clip_captions_v4"
CLEAN_TEST = PROJECT_ROOT / "outputs" / "full_run_v3" / "clean_test.jsonl"
CAPTIONS = PROJECT_ROOT / "outputs" / "kaggle_bundle" / "caption_inputs.jsonl"

VARIANTS = ["article", "all_captions", "clip_top2", "random2"]
METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
SEED = 42
N_BOOT = 10000

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being
has have had do does did this that these those it its he she they them his her their we you i our your
who whom which what when where why how not no so than then there here into over after before about
up down out off said says say will would could should can may might also just more most very""".split())
stem = PorterStemmer().stem


def content_words(text):
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP}


def bigrams(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return set(zip(w, w[1:]))


def load_predictions(variant):
    return {r["id"]: r for r in csv.DictReader(open(RUN_DIR / f"test_predictions_{variant}.csv"))}


def paired(a, b, ids, rng):
    """Mean difference b - a with a 95% bootstrap CI, and the share of resamples where b wins."""
    out = {}
    for m in METRICS:
        d = np.array([float(b[i][m]) - float(a[i][m]) for i in ids])
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
        out[m] = {"diff": round(float(d.mean()), 4),
                  "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                  "p_b_better": round(float((boots > 0).mean()), 4)}
    return out


def significance(preds, captions):
    test_ids = list(preds["article"])
    dm = [i for i in test_ids if preds["article"][i]["source"] == "dailymail"]
    differs = [i for i in dm if i in captions and set(captions[i]["clip_top2"]) != set(captions[i]["random2"])]
    groups = {"all": test_ids, "dailymail": dm, "dm_clip_differs_from_random": differs}
    pairs = [("article", "all_captions"), ("article", "clip_top2"), ("article", "random2"),
             ("random2", "clip_top2"), ("clip_top2", "all_captions")]
    out = {}
    for g, ids in groups.items():
        rng = np.random.default_rng(SEED)
        out[g] = {"n": len(ids), **{f"{b}_vs_{a}": paired(preds[a], preds[b], ids, rng) for a, b in pairs}}
    return out


def overlap(preds, captions, articles):
    """Where do reference and predicted words come from: the article, or only the captions?"""
    ref_cap_only, ref_bi_cap_only, ref_cap_any = [], [], []
    from_captions = {v: [] for v in VARIANTS[1:]}
    from_captions_in_ref = {v: [] for v in VARIANTS[1:]}
    for i, row in preds["article"].items():
        if row["source"] != "dailymail" or i not in captions or not captions[i]["all_captions"]:
            continue
        art = content_words(articles[i])
        caps = content_words(" ".join(captions[i]["all_captions"]))
        ref = content_words(row["reference"])
        ref_cap_any.append(len(ref & caps) / max(len(ref), 1))
        ref_cap_only.append(len(ref & (caps - art)) / max(len(ref), 1))
        ref_bi = bigrams(row["reference"])
        ref_bi_cap_only.append(len(ref_bi & (bigrams(" ".join(captions[i]["all_captions"])) - bigrams(articles[i])))
                               / max(len(ref_bi), 1))
        for v in from_captions:
            shown = content_words(" ".join(captions[i][v]))
            pred = content_words(preds[v][i]["prediction"])
            new = pred & (shown - art)          # words the model could only have taken from the captions
            from_captions[v].append(len(new) / max(len(pred), 1))
            if new:
                from_captions_in_ref[v].append(len(new & ref) / len(new))
    pct = lambda xs: round(100 * float(np.mean(xs)), 2)
    return {
        "stories": len(ref_cap_only),
        "reference_words_in_captions_pct": pct(ref_cap_any),
        "reference_words_only_in_captions_pct": pct(ref_cap_only),
        "reference_bigrams_only_in_captions_pct": pct(ref_bi_cap_only),
        "prediction_words_only_from_captions_pct": {v: pct(x) for v, x in from_captions.items()},
        "of_those_also_in_reference_pct": {v: pct(x) for v, x in from_captions_in_ref.items()},
    }


def examples(preds, captions, n=3):
    dm = [i for i, r in preds["article"].items() if r["source"] == "dailymail" and i in captions]
    gain = sorted(dm, key=lambda i: float(preds["all_captions"][i]["rouge2"]) - float(preds["article"][i]["rouge2"]))
    pick = lambda ids: [{"id": i, "captions": captions[i]["all_captions"], "reference": preds["article"][i]["reference"],
                         "article_only": preds["article"][i]["prediction"],
                         "all_captions": preds["all_captions"][i]["prediction"],
                         "rouge2_gain": round(float(preds["all_captions"][i]["rouge2"]) - float(preds["article"][i]["rouge2"]), 4)}
                        for i in ids]
    return {"largest_gains": pick(gain[::-1][:n]), "largest_losses": pick(gain[:n])}


def main():
    preds = {v: load_predictions(v) for v in VARIANTS}
    captions = {r["id"]: r for r in map(json.loads, open(CAPTIONS))}
    articles = {r["id"]: r["article"] for r in map(json.loads, open(CLEAN_TEST))}
    result = {"significance": significance(preds, captions), "overlap": overlap(preds, captions, articles),
              "examples": examples(preds, captions)}
    (RUN_DIR / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))

    for g, res in result["significance"].items():
        print(f"\n== {g} (n={res['n']})")
        for name, r in res.items():
            if name != "n":
                print(f"  {name:28s} " + "  ".join(
                    f"{m.replace('rouge', 'R')} {100 * r[m]['diff']:+.2f} [{100 * r[m]['ci95'][0]:+.2f},{100 * r[m]['ci95'][1]:+.2f}] p={r[m]['p_b_better']:.3f}"
                    for m in METRICS))
    print("\n== overlap", json.dumps(result["overlap"], indent=1))
    print(f"\nWrote {RUN_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
