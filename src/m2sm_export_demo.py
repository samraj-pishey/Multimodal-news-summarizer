"""Write the saved examples for the static demo site in demo/ from one finished M2SM run.

Stories are drawn from the Daily Mail test split at random with a fixed seed, before any output is seen,
so the demo isn't limited to stories where the model looks good.

Usage: python src/m2sm_export_demo.py --run m2sm_full [--stories 6] [--out demo/data]
"""
from pathlib import Path
import argparse
import json
import random
import shutil

import av
import torch

from m2sm_data import CACHE_PATH, DATA_ROOT, PROJECT_ROOT, score_summary
from m2sm_model import M2SM
from m2sm_train import OUT_DIR, RUN_DIR, load_frames, np, feature_path

MAX_SENTENCES = 8
FRAMES_PER_STORY = 4
THUMBNAIL_WIDTH = 480


def pick_stories(stories, count):
    usable = [s for s in stories if s["split_paper"] == "test" and s["source"] == "dailymail"
              and s["transcript_ids"] and len(s["sentences"]) >= MAX_SENTENCES]
    usable.sort(key=lambda s: int(s["id"].split("/")[1]))
    return random.Random(42).sample(usable, count)


def load_model(run, embeddings):
    config = json.loads((OUT_DIR / run / "results.json").read_text())
    args = config["config"]
    model = M2SM(embeddings, video=not args["no_video"], transcript=not args["no_transcript"],
                 bistream=not (args["no_bistream"] or args["no_video"]), attention=args["attention"],
                 fusion=args["fusion"], beta=args["beta"])
    best = torch.load(RUN_DIR / run / "best.pt", weights_only=False)
    model.load_state_dict(best["model"])
    return model.eval(), config, best["size"]


def top_sentences(scores, count):
    return sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:count])


def scores_by_length(story, picks_for):
    """ROUGE F1 (percent) for each summary length, so the page can update without running anything."""
    rows = {}
    for count in range(1, MAX_SENTENCES + 1):
        result = score_summary(story["reference"], [story["sentences"][i] for i in picks_for(count)])
        rows[count] = {k: round(100 * v, 1) for k, v in result.items()}
    return rows


def save_frames(story_id, times, out_dir):
    """Decodes the source video and saves a thumbnail at each selected time."""
    number = story_id.split("/")[1]
    path = DATA_ROOT / "daily mail" / number / "video.mp4"
    names, targets = {}, sorted(times)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            start = frame.time if start is None else start
            while targets and frame.time - start >= targets[0] - 1e-3:
                target = targets.pop(0)
                image = frame.to_image()
                image.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH))
                name = f"{number}_{len(names)}.jpg"
                image.convert("RGB").save(out_dir / name, quality=80)
                names[target] = name
            if not targets:
                break
    return names


@torch.no_grad()
def export_story(story, model, config, frames_dir):
    args = config["config"]
    sentence_ids = [torch.tensor(ids) for ids in story["sentence_ids"]]
    if args["no_video"]:
        logits, frame_logits = model(sentence_ids)
    else:
        words = None if args["no_transcript"] else torch.tensor(story["transcript_ids"], dtype=torch.long)
        logits, frame_logits = model(sentence_ids, load_frames(story["id"]), words)
    scores = logits.sigmoid().tolist()

    frames = []
    if frame_logits is not None:
        times = np.load(feature_path(story["id"]))["times"]
        chosen = frame_logits.topk(min(FRAMES_PER_STORY, len(frame_logits)))
        names = save_frames(story["id"], [float(times[i]) for i in chosen.indices], frames_dir)
        frames = sorted(({"file": names[float(times[i])], "time": round(float(times[i]), 1),
                          "score": round(s, 3)} for i, s in zip(chosen.indices.tolist(), chosen.values.sigmoid().tolist())
                         if float(times[i]) in names), key=lambda f: f["time"])

    return {"id": story["id"], "sentences": story["sentences"], "scores": [round(s, 3) for s in scores],
            "reference": story["reference"], "transcript": story["transcript"], "frames": frames,
            "rouge": {"model": scores_by_length(story, lambda n: top_sentences(scores, n)),
                      "lead": scores_by_length(story, lambda n: list(range(n)))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--stories", type=int, default=6)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo" / "data")
    args = parser.parse_args()
    torch.set_num_threads(2)

    cache = torch.load(CACHE_PATH, weights_only=False)
    model, config, size = load_model(args.run, cache["embeddings"])
    if args.out.exists():
        shutil.rmtree(args.out)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True)

    examples = [export_story(s, model, config, frames_dir) for s in pick_stories(cache["stories"], args.stories)]
    payload = {"run": args.run, "learning_rate": config["config"]["lr"], "default_length": size,
               "max_length": MAX_SENTENCES, "test": config.get("test_dailymail"), "test_cos": config.get("test_cos"),
               "uses_video": not config["config"]["no_video"], "selects_frames": bool(examples[0]["frames"]),
               "examples": examples}
    (args.out / "examples.json").write_text(json.dumps(payload, ensure_ascii=False))
    print(f"Wrote {len(examples)} stories from {args.run} to {args.out}")


if __name__ == "__main__":
    main()
