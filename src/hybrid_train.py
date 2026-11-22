"""Train and evaluate the hybrid model (hybrid_model.py). Resumable, like m2sm_train.py.

Checkpoints are chosen on validation only: the text one by mean ROUGE F1 with the summary size picked on validation,
the frame one by Cos. Test is scored once, after training. A2Summ picked its checkpoints on test and set each story's
summary size to its number of gold labels; with --oracle-sizes that sizing is also reported, marked as such.

Usage: python src/hybrid_train.py --name hybrid [--no-video] [--no-speech] [--no-align] [--no-contrastive]
       [--no-sentence-lstm] [--fusion lateplus|none] [--seed 42] [--epochs 100] [--patience 10]
"""
import argparse
import json
import random
import statistics
import time

import numpy as np
import torch

from hybrid_data import HYBRID_PATH
from hybrid_model import Hybrid, focal_loss
from m2sm_baselines import cos_for_frames
from m2sm_data import CACHE_PATH, PROJECT_ROOT, SPEECH_CACHE_PATH, VECTOR_PATHS, feature_path, score_summary

RUN_DIR = PROJECT_ROOT / "data" / "hybrid_runs"
OUT_DIR = PROJECT_ROOT / "outputs" / "hybrid" / "runs"
SUMMARY_SIZES = (2, 3, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    for flag in ("--no-video", "--no-speech", "--no-align", "--no-contrastive", "--no-sentence-lstm", "--no-test"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--fusion", choices=["lateplus", "none"], default="lateplus")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inter-weight", type=float, default=1e-3)
    parser.add_argument("--intra-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit-train", type=int, default=0, help="pilot only")
    parser.add_argument("--limit-val", type=int, default=0, help="pilot only")
    return parser.parse_args()


def build_samples(args):
    """One dict of tensors per Daily Mail and CNN story, loaded once."""
    stories = torch.load(CACHE_PATH, weights_only=False)["stories"]
    article_vectors = torch.load(VECTOR_PATHS["article"])
    speech_vectors = torch.load(VECTOR_PATHS["speech"])
    speech_ids = {s["id"] for s in torch.load(SPEECH_CACHE_PATH, weights_only=False)}
    extra = torch.load(HYBRID_PATH, weights_only=False)
    samples = []
    for story in stories:
        data = np.load(feature_path(story["id"]))
        info = extra[story["id"]]
        sample = {"id": story["id"], "source": story["source"], "split": story["split_paper"], "story": story,
                  "article": article_vectors[story["id"]].float(),
                  "sentence_labels": torch.tensor(story["labels"], dtype=torch.float32),
                  "frames": torch.from_numpy(data["frames"].astype(np.float32)),
                  "frame_times": torch.from_numpy(info["frame_times"]),
                  "images": torch.from_numpy(data["images"].astype(np.float32))}
        if info["frame_labels"] is not None:
            sample["frame_labels"] = torch.from_numpy(info["frame_labels"])
        if story["id"] in speech_ids and "speech_times" in info:
            sample["speech"] = speech_vectors[story["id"]].float()
            sample["speech_times"] = torch.from_numpy(info["speech_times"])
        samples.append(sample)
    return samples


def make_model(args):
    return Hybrid(use_video=not args.no_video, use_speech=not args.no_speech, sentence_lstm=not args.no_sentence_lstm,
                  fusion=args.fusion, align=not args.no_align)


@torch.no_grad()
def predict(model, samples, batch_size=8):
    model.eval()
    scores = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        for sample, out in zip(batch, model(batch)):
            scores[sample["id"]] = {"sentences": out["sentence_logits"].sigmoid().tolist(),
                                    "frames": out["frame_logits"].tolist() if "frame_logits" in out else None}
    return scores


def top(scores, count):
    return sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:count])


def rouge(samples, scores, size_of):
    rows = [score_summary(s["story"]["reference"], [s["story"]["sentences"][i]
                                                     for i in top(scores[s["id"]]["sentences"], size_of(s))])
            for s in samples]
    return {k: 100 * statistics.mean(r[k] for r in rows) for k in ("r1", "r2", "rl")}


def cos(samples, scores):
    """Cos as in m2sm_baselines.py: k = number of reference images, top-k frames by score."""
    values = []
    for s in samples:
        if len(s["images"]) == 0 or scores[s["id"]]["frames"] is None:
            continue
        frames = torch.nn.functional.normalize(s["frames"], dim=1)
        images = torch.nn.functional.normalize(s["images"], dim=1)
        chosen = top(scores[s["id"]]["frames"], min(len(images), len(frames)))
        values.append(cos_for_frames(frames, images, chosen))
    return (100 * statistics.mean(values), len(values)) if values else (None, 0)


def mean_rouge(metrics):
    return (metrics["r1"] + metrics["r2"] + metrics["rl"]) / 3


def train_step(model, batch, args):
    outputs = model(batch)
    sentence_loss = torch.stack([focal_loss(o["sentence_logits"], s["sentence_labels"])
                                 for o, s in zip(outputs, batch)]).mean()
    loss = sentence_loss
    frame_losses = [focal_loss(o["frame_logits"], s["frame_labels"])
                    for o, s in zip(outputs, batch) if "frame_logits" in o and "frame_labels" in s]
    if frame_losses:
        loss = loss + torch.stack(frame_losses).mean()
    if not args.no_contrastive and not args.no_video:
        inter = model.inter_loss(outputs)
        intra = [l for l in (model.intra_loss(o, s) for o, s in zip(outputs, batch)) if l is not None]
        if inter is not None:
            loss = loss + args.inter_weight * inter
        if intra:
            loss = loss + args.intra_weight * torch.stack(intra).mean()
    return loss, sentence_loss.item()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_samples(args)
    split = lambda name, source="dailymail": [s for s in samples if s["split"] == name and s["source"] == source]
    train, val = split("train"), split("val")
    train = train[:args.limit_train or None]
    val = val[:args.limit_val or None]

    model = make_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir, out_dir = RUN_DIR / args.name, OUT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {"history": [], "best_text": float("-inf"), "best_text_epoch": 0, "best_cos": float("-inf"),
             "best_cos_epoch": 0, "epoch": 0}
    if (run_dir / "last.pt").exists():
        saved = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        print(f"Resuming {args.name} after epoch {state['epoch']}", flush=True)
    print(f"{args.name}: train {len(train)}, val {len(val)}, {sum(p.numel() for p in model.parameters()):,} parameters",
          flush=True)

    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        last_gain = max(state["best_text_epoch"], state["best_cos_epoch"])
        if last_gain and epoch - last_gain > args.patience:
            break
        began = time.time()
        model.train()
        random.shuffle(train)
        losses = []
        for start in range(0, len(train), args.batch_size):
            loss, sentence_loss = train_step(model, train[start:start + args.batch_size], args)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(sentence_loss)

        scores = predict(model, val)
        by_size = {k: rouge(val, scores, lambda s, k=k: k) for k in SUMMARY_SIZES}
        size = max(by_size, key=lambda k: mean_rouge(by_size[k]))
        value, (val_cos, _) = mean_rouge(by_size[size]), cos(val, scores)
        marks = ""
        if value > state["best_text"]:
            state.update(best_text=value, best_text_epoch=epoch, size=size)
            torch.save(model.state_dict(), run_dir / "best_text.pt")
            marks += " *text"
        if val_cos is not None and val_cos > state["best_cos"]:
            state.update(best_cos=val_cos, best_cos_epoch=epoch)
            torch.save(model.state_dict(), run_dir / "best_frames.pt")
            marks += " *frames"
        state["history"].append({"epoch": epoch, "loss": statistics.mean(losses), "val": by_size[size], "k": size,
                                 "val_cos": val_cos, "seconds": time.time() - began})
        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                    "python_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, run_dir / "last.pt")
        v = by_size[size]
        cos_text = f" cos {val_cos:.2f}" if val_cos is not None else ""
        print(f"  epoch {epoch}: loss {statistics.mean(losses):.4f} val {v['r1']:.2f}/{v['r2']:.2f}/{v['rl']:.2f} "
              f"k={size}{cos_text}{marks} ({time.time() - began:.0f}s)", flush=True)

    result = {"config": vars(args), "state": {k: v for k, v in state.items() if k != "history"},
              "history": state["history"], "train_stories": len(train), "val_stories": len(val)}
    if not args.no_test:
        predictions = {}
        model.load_state_dict(torch.load(run_dir / "best_text.pt"))
        for source in ("dailymail", "cnn"):
            test = split("test", source)
            scores = predict(model, test)
            result[f"test_{source}"] = {"n": len(test), **rouge(test, scores, lambda s: state["size"])}
            result[f"test_{source}_gold_sizes"] = {
                "n": len(test), "note": "summary size = number of gold labels, as in A2Summ's code",
                **rouge(test, scores, lambda s: max(1, int(s["sentence_labels"].sum())))}
            for s in test:
                predictions[s["id"]] = {"sentences": [round(x, 4) for x in scores[s["id"]]["sentences"]]}
        if not args.no_video and (run_dir / "best_frames.pt").exists():
            model.load_state_dict(torch.load(run_dir / "best_frames.pt"))
            test = split("test")
            scores = predict(model, test)
            value, n = cos(test, scores)
            result["test_cos"] = {"n": n, "cos": value}
            for s in test:
                predictions[s["id"]]["frames"] = [round(x, 4) for x in scores[s["id"]]["frames"]]
        (out_dir / "test_scores.json").write_text(json.dumps(predictions))
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k.startswith("test")}), flush=True)


if __name__ == "__main__":
    main()
