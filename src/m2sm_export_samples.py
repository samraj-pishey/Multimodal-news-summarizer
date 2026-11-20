"""Write the sample stories for webapp/ from one finished M2SM run.

Stories come from the Daily Mail test split at random with a fixed seed, before any output is seen. The browser
runs the model on them live, using the stored frame features; ROUGE and thumbnails are saved here.

Usage: python src/m2sm_export_samples.py --run m2sm_full_lr0.01 [--skip 5099]
"""
import argparse
import json
import re
import shutil

import torch

import m2sm_export_demo as demo
from m2sm_data import CACHE_PATH, DATA_ROOT, PROJECT_ROOT
from m2sm_train import OUT_DIR, feature_path, load_frames, np

OUT = PROJECT_ROOT / "webapp" / "samples"
SAMPLE_COUNT = 5
GLUED_HIGHLIGHT = re.compile(r"(?<!\bMc)(?<!\bMac)(?<=[a-z0-9%)'’\".!?])(?=[A-Z][a-z])")


def human_points(reference):
    """Highlights are glued together in the raw files: "last FridayRaiders seen". For display only."""
    return [part.strip() for part in GLUED_HIGHLIGHT.sub("\n", reference.replace("\xa0", " ")).split("\n") if part.strip()]


def title_of(story):
    path = DATA_ROOT / "daily mail" / story["id"].split("/")[1] / "title.txt"
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def lead_matches(run, size):
    """How many test stories get exactly their first sentences as a summary."""
    scores = json.loads((OUT_DIR / run / "test_scores.json").read_text())
    rows = [v["scores"] for k, v in scores.items() if k.startswith("dailymail")]
    same = sum(demo.top_sentences(r, size) == list(range(min(size, len(r)))) for r in rows)
    return same, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--skip", nargs="*", default=[], help="story numbers to leave out, e.g. for graphic content")
    args = parser.parse_args()
    torch.set_num_threads(2)
    demo.FRAMES_PER_STORY = 6        # a few spare thumbnails, in case of ties near the top four

    cache = torch.load(CACHE_PATH, weights_only=False)
    model, config, size = demo.load_model(args.run, cache["embeddings"])
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    candidates = demo.pick_stories(cache["stories"], 12)
    for story in candidates:
        print(story["id"], "|", title_of(story)[:100], flush=True)
    chosen = [s for s in candidates if s["id"].split("/")[1] not in args.skip][:SAMPLE_COUNT]

    samples = []
    for story in chosen:
        row = demo.export_story(story, model, config, OUT)
        number = story["id"].split("/")[1]
        load_frames(story["id"]).numpy().astype(np.float32).tofile(OUT / f"{number}.bin")
        times = np.load(feature_path(story["id"]))["times"]
        samples.append({"id": story["id"], "number": number, "title": title_of(story),
                        "sentences": story["sentences"], "transcript": story["transcript"],
                        "human": human_points(story["reference"]), "rouge": row["rouge"], "frames": row["frames"],
                        "times": [round(float(t), 1) for t in times]})
    same, total = lead_matches(args.run, size)
    (OUT / "samples.json").write_text(json.dumps(
        {"samples": samples, "lead_matches": same, "lead_total": total}, ensure_ascii=False))
    print(f"Wrote {len(samples)} samples to {OUT}: {sum(p.stat().st_size for p in OUT.iterdir()) / 2 ** 20:.1f} MB")
    for sample in samples:
        print(sample["id"], sample["human"][:2], len(sample["frames"]))


if __name__ == "__main__":
    main()
