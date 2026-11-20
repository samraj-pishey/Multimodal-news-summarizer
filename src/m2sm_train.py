"""Train and evaluate one M2SM configuration. Resumable: a stopped run continues from its last epoch.

The epoch and the number of extracted sentences are chosen on validation; the test split is scored once,
after training ends, with those choices fixed.
"""
from pathlib import Path
import argparse
import json
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from m2sm_data import CACHE_PATH, PROJECT_ROOT, VECTOR_PATHS, feature_path, score_summary, speech_stories
from m2sm_model import M2SM, normalized, reinforce_loss

RUN_DIR = PROJECT_ROOT / "data" / "m2sm_runs"
OUT_DIR = PROJECT_ROOT / "outputs" / "m2sm" / "runs"
SUMMARY_SIZES = (2, 3, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--regime", choices=["paper", "clean"], default="paper")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-transcript", action="store_true")
    parser.add_argument("--no-bistream", action="store_true")
    parser.add_argument("--attention", choices=["bilinear", "concat", "none"], default="bilinear")
    parser.add_argument("--fusion", choices=["early", "tensor", "late", "lateplus"], default="lateplus")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--alpha-ratio", type=float, default=3.33)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adagrad", "adam"], default="adagrad")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speech", action="store_true", help="video-only mode: Whisper sentences replace the article")
    parser.add_argument("--init-from", help="start from this finished run's best weights")
    parser.add_argument("--sentence-vectors", action="store_true",
                        help="use the RoBERTa vectors from m2sm_sentence_vectors.py instead of the GloVe word BiLSTM")
    parser.add_argument("--limit-train", type=int, default=0, help="pilot only")
    parser.add_argument("--limit-val", type=int, default=0, help="pilot only")
    parser.add_argument("--no-test", action="store_true", help="pilot only: stop before the test split")
    return parser.parse_args()


def load_frames(story_id):
    return torch.from_numpy(np.load(feature_path(story_id))["frames"].astype(np.float32))


SENTENCE_VECTORS = {}


def forward(model, story, args):
    if args.sentence_vectors:
        sentence_ids = SENTENCE_VECTORS[story["id"]].float()
    else:
        sentence_ids = [torch.tensor(ids) for ids in story["sentence_ids"]]
    if args.no_video:
        return model(sentence_ids)
    frames = load_frames(story["id"])
    transcript = None if args.no_transcript else torch.tensor(story["transcript_ids"], dtype=torch.long)
    return (*model(sentence_ids, frames, transcript), frames)


@torch.no_grad()
def sentence_scores(model, stories, args):
    model.eval()
    return {s["id"]: forward(model, s, args)[0].sigmoid().tolist() for s in stories}


def summarize(story, scores, size):
    """Top sentences by score, restored to article order."""
    chosen = sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:size])
    return [story["sentences"][i] for i in chosen]


def evaluate(stories, scores, size):
    rows = [score_summary(s["reference"], summarize(s, scores[s["id"]], size)) for s in stories]
    return {k: 100 * statistics.mean(r[k] for r in rows) for k in ("r1", "r2", "rl")}


def selection_value(metrics):
    return (metrics["r1"] + metrics["r2"] + metrics["rl"]) / 3


@torch.no_grad()
def cosine_score(model, stories, args):
    """Sec. 4.1: cosine between reference images and the top frames, one frame kept per image."""
    model.eval()
    values = []
    for story in stories:
        images = torch.from_numpy(np.load(feature_path(story["id"]))["images"].astype(np.float32))
        if len(images) == 0:
            continue
        _, frame_logits, frames = forward(model, story, args)
        top = frame_logits.topk(min(len(images), len(frames))).indices
        similarity = normalized(images) @ normalized(frames[top]).T
        values.append(similarity.max(1).values.mean().item())
    return 100 * statistics.mean(values), len(values)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = torch.load(CACHE_PATH, weights_only=False)
    key = f"split_{args.regime}"
    stories = cache["stories"]
    if args.speech:
        stories = speech_stories(stories, cache["vocabulary"])
    split = lambda name, source: [s for s in stories if s[key] == name and s["source"] == source]
    train, val = split("train", "dailymail"), split("val", "dailymail")
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_val:
        val = val[:args.limit_val]

    bistream = not (args.no_bistream or args.no_video)
    sentence_dim = None
    if args.sentence_vectors:
        SENTENCE_VECTORS.update(torch.load(VECTOR_PATHS["speech" if args.speech else "article"]))
        stories = [s for s in stories if s["id"] in SENTENCE_VECTORS]
        sentence_dim = next(iter(SENTENCE_VECTORS.values())).shape[1]
    model = M2SM(cache["embeddings"], video=not args.no_video, transcript=not args.no_transcript,
                 bistream=bistream, attention=args.attention, fusion=args.fusion, beta=args.beta,
                 sentence_dim=sentence_dim)
    make_optimizer = torch.optim.Adagrad if args.optimizer == "adagrad" else torch.optim.Adam
    optimizer = make_optimizer(model.parameters(), lr=args.lr)

    run_dir, out_dir = RUN_DIR / args.name, OUT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_value, best_epoch, start_epoch, reward_baseline = [], float("-inf"), 0, 1, 0.0
    if (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_value, best_epoch = state["history"], state["best_value"], state["best_epoch"]
        start_epoch, reward_baseline = state["epoch"] + 1, state["reward_baseline"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"Resuming {args.name} at epoch {start_epoch}", flush=True)
    elif args.init_from:
        model.load_state_dict(torch.load(RUN_DIR / args.init_from / "best.pt", weights_only=False)["model"])

    alpha_vs = 1 / args.alpha_ratio    # the text loss keeps weight 1; only the ratio is reported in the paper
    print(f"{args.name}: train {len(train)}, val {len(val)}, {sum(p.numel() for p in model.parameters()):,} parameters",
          flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - best_epoch > args.patience and best_epoch:
            break
        t0 = time.time()
        model.train()
        random.shuffle(train)
        text_losses, rewards = [], []
        optimizer.zero_grad()
        for n, story in enumerate(train, 1):
            output = forward(model, story, args)
            labels = torch.tensor(story["labels"])
            text_loss = F.binary_cross_entropy_with_logits(output[0], labels)
            loss = text_loss
            if bistream:
                features = normalized(output[2])
                video_loss, reward = reinforce_loss(output[1], features, reward_baseline)
                reward_baseline = 0.9 * reward_baseline + 0.1 * reward
                loss = loss + alpha_vs * video_loss
                rewards.append(reward)
            (loss / args.batch_size).backward()
            if n % args.batch_size == 0 or n == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad()
            text_losses.append(text_loss.item())

        scores = sentence_scores(model, val, args)
        by_size = {size: evaluate(val, scores, size) for size in SUMMARY_SIZES}
        size = max(by_size, key=lambda k: selection_value(by_size[k]))
        value = selection_value(by_size[size])
        history.append({"epoch": epoch, "train_loss": statistics.mean(text_losses), "val": by_size[size],
                        "val_size": size, "val_value": value, "seconds": time.time() - t0,
                        "reward": statistics.mean(rewards) if rewards else None})
        if value > best_value:
            best_value, best_epoch = value, epoch
            torch.save({"model": model.state_dict(), "size": size}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                    "best_value": best_value, "best_epoch": best_epoch, "epoch": epoch,
                    "reward_baseline": reward_baseline, "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state()}, run_dir / "last.pt")
        h = history[-1]
        print(f"  epoch {epoch}: loss {h['train_loss']:.4f} val R1/R2/RL {h['val']['r1']:.2f}/{h['val']['r2']:.2f}/"
              f"{h['val']['rl']:.2f} k={size}{' *' if epoch == best_epoch else ''} ({h['seconds']:.0f}s)", flush=True)

    result = {"config": vars(args), "best_epoch": best_epoch, "best_val": best_value, "history": history,
              "train_stories": len(train), "val_stories": len(val)}
    if not args.no_test:
        best = torch.load(run_dir / "best.pt", weights_only=False)
        model.load_state_dict(best["model"])
        size = best["size"]
        result["summary_size"] = size
        predictions = {}
        for source in ("dailymail", "cnn"):
            test = split("test", source)
            if not test:      # video-only mode has no Whisper text for CNN
                continue
            scores = sentence_scores(model, test, args)
            result[f"test_{source}"] = {"n": len(test), **evaluate(test, scores, size)}
            for story in test:
                predictions[story["id"]] = {"scores": [round(x, 4) for x in scores[story["id"]]]}
        if bistream:
            value, n = cosine_score(model, split("test", "dailymail"), args)
            result["test_cos"] = {"n": n, "cos": value}
        (out_dir / "test_scores.json").write_text(json.dumps(predictions))
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k.startswith("test") or k == "best_epoch"}), flush=True)


if __name__ == "__main__":
    main()
