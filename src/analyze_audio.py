from pathlib import Path
import csv
import json

import numpy as np

from analyze_captions import content_words, paired, METRICS, SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "outputs" / "audio_transcripts_v5"
CAPTION_RUN = PROJECT_ROOT / "outputs" / "clip_captions_v4"
CLEAN_TEST = PROJECT_ROOT / "outputs" / "full_run_v3" / "clean_test.jsonl"
AUDIO = PROJECT_ROOT / "outputs" / "kaggle_bundle" / "audio_inputs.jsonl"

VARIANTS = ["article", "transcript_all", "transcript_relevant", "clip_top2_transcript_relevant"]


def load(run_dir, variant):
    return {r["id"]: r for r in csv.DictReader(open(run_dir / f"test_predictions_{variant}.csv"))}


def main():
    preds = {v: load(RUN_DIR, v) for v in VARIANTS}
    preds["clip_top2_run4"] = load(CAPTION_RUN, "clip_top2")
    audio = {r["id"]: r for r in map(json.loads, open(AUDIO))}
    articles = {r["id"]: r["article"] for r in map(json.loads, open(CLEAN_TEST))}

    ids = list(preds["article"])
    with_speech = [i for i in ids if audio.get(i, {}).get("transcript_all")]
    relevant = [i for i in with_speech if audio[i]["transcript_relevant"]]
    irrelevant = [i for i in with_speech if not audio[i]["transcript_relevant"]]
    groups = {"all": ids, "with_speech": with_speech, "relevant": relevant, "irrelevant": irrelevant}
    pairs = [("article", "transcript_all"), ("article", "transcript_relevant"),
             ("transcript_all", "transcript_relevant"), ("article", "clip_top2_transcript_relevant"),
             ("clip_top2_run4", "clip_top2_transcript_relevant")]

    significance = {}
    for g, members in groups.items():
        rng = np.random.default_rng(SEED)
        significance[g] = {"n": len(members), **{f"{b}_vs_{a}": paired(preds[a], preds[b], members, rng)
                                                  for a, b in pairs}}

    # How much of each summary could only have come from the transcript, and is it right?
    copied = {}
    for v, field in (("transcript_all", "transcript_all"), ("transcript_relevant", "transcript_relevant")):
        share, correct = [], []
        for i in with_speech:
            shown = audio[i][field]
            if not shown:
                continue
            new = content_words(preds[v][i]["prediction"]) & (content_words(shown) - content_words(articles[i]))
            share.append(len(new) / max(len(content_words(preds[v][i]["prediction"])), 1))
            if new:
                correct.append(len(new & content_words(preds[v][i]["reference"])) / len(new))
        copied[v] = {"stories": len(share), "pred_words_only_from_transcript_pct": round(100 * float(np.mean(share)), 2),
                     "of_those_in_reference_pct": round(100 * float(np.mean(correct)), 2) if correct else None}

    result = {"significance": significance, "transcript_word_use": copied}
    (RUN_DIR / "analysis.json").write_text(json.dumps(result, indent=1))
    for g, res in significance.items():
        print(f"\n== {g} (n={res['n']})")
        for name, r in res.items():
            if name != "n":
                print(f"  {name:52s} " + "  ".join(
                    f"{m.replace('rouge', 'R')} {100 * r[m]['diff']:+.2f} p={r[m]['p_b_better']:.3f}" for m in METRICS))
    print("\n", json.dumps(copied, indent=1))


if __name__ == "__main__":
    main()
