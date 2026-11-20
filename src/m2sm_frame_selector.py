"""Supervised frame selector, our own addition beside the M2SM replication. Not part of Fu et al. 2020.

A VASNet-style scorer (Fajtl et al. 2018): one self-attention layer and a small head over saved ResNet-50 frame
features. The target for each frame is its best cosine with the story's reference images, scaled to 0-1 per video.
Reference images are read only as training labels on the train split and to score validation and test; the model
never sees them when it picks frames.

Learning rate, epoch and minimum spacing between picked frames are chosen on validation. Test is scored once, at
the end, with the same cosine as the baselines in m2sm_baselines.py.

With --article, each frame also attends over the article's sentences with the paper's bilinear attention (Eq. 4-7),
so the attention scores between every sentence and every frame feed into the frame's score.

Usage: python src/m2sm_frame_selector.py [--article] [--seed 42] [--threads 6] [--limit-train N] [--limit-val N]
       [--no-test]
"""
import argparse
import json
import random
import statistics
import time

import numpy as np
import torch

from m2sm_baselines import cos_for_frames, load_video, uniform_choice, vsumm_choice
from m2sm_data import CACHE_PATH, PROJECT_ROOT
from m2sm_model import BilinearAttention

OUT_DIR = PROJECT_ROOT / "outputs" / "m2sm" / "runs" / "frame_selector"
RUN_DIR = PROJECT_ROOT / "data" / "m2sm_runs" / "frame_selector"
LEARNING_RATES = (1e-3, 1e-4)
GAPS = (1, 3, 5, 10)
MAX_EPOCHS = 30
PATIENCE = 5
RANDOM_REPEATS = 20
BOOTSTRAPS = 10000


class FrameSelector(torch.nn.Module):
    def __init__(self, hidden=256, heads=4, dropout=0.5):
        super().__init__()
        self.project = torch.nn.Linear(2048, hidden)
        self.attention = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                        torch.nn.Linear(hidden, 1))

    def encode(self, frames):
        x = self.project(frames).unsqueeze(0)
        context = self.attention(x, x, x, need_weights=False)[0]
        return self.norm(x + self.dropout(context)).squeeze(0)

    def forward(self, frames):
        return self.head(self.encode(frames)).squeeze(-1)

    def score(self, story, frames):
        return self(frames)


class ArticleFrameSelector(FrameSelector):
    """Adds attention from every frame to every sentence. Sentences are averaged GloVe vectors, kept fixed."""

    def __init__(self, embeddings, hidden=256, dropout=0.5):
        super().__init__(hidden=hidden, dropout=dropout)
        self.register_buffer("glove", embeddings.clone().float())
        self.sentence_project = torch.nn.Linear(embeddings.shape[1], hidden)
        self.frame_to_sentence = BilinearAttention(hidden)
        self.combine = torch.nn.Linear(2 * hidden, hidden)

    def sentences(self, story):
        vectors = [self.glove[torch.tensor(ids)].mean(0) for ids in story["sentence_ids"] if len(ids)]
        return self.sentence_project(torch.stack(vectors))

    def attention_scores(self, story, frames):
        """Frames x sentences: how much each frame attends to each sentence. Each row sums to 1."""
        return self.frame_to_sentence.weights(self.encode(frames), self.sentences(story))

    def score(self, story, frames):
        states = self.encode(frames)
        sentences = self.sentences(story)
        context = self.frame_to_sentence(states, sentences, sentences)
        return self.head(torch.tanh(self.combine(torch.cat([states, self.dropout(context)], -1)))).squeeze(-1)


def frame_targets(frames, images):
    best = (frames @ images.T).max(1).values
    span = best.max() - best.min()
    return (best - best.min()) / span if span > 1e-6 else torch.zeros_like(best)


def usable(story):
    frames, images = load_video(story)
    return len(frames) >= 2 and len(images) > 0


def pick_frames(scores, k, gap):
    """Highest scores first, at least `gap` frames apart; the spacing is dropped if the video runs out of room."""
    order = scores.argsort(descending=True).tolist()
    chosen = []
    for index in order:
        if all(abs(index - other) >= gap for other in chosen):
            chosen.append(index)
        if len(chosen) == k:
            return chosen
    return chosen + [i for i in order if i not in chosen][:k - len(chosen)]


@torch.no_grad()
def validation_cos(model, stories):
    """Mean cosine for every spacing, so the spacing is picked with the epoch."""
    model.eval()
    totals = {gap: [] for gap in GAPS}
    for story in stories:
        frames, images = load_video(story)
        k = min(len(images), len(frames))
        scores = model.score(story, frames)
        for gap in GAPS:
            totals[gap].append(cos_for_frames(frames, images, pick_frames(scores, k, gap)))
    return {gap: 100 * statistics.mean(v) for gap, v in totals.items()}


def train_one(make_model, lr, train, val, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best = {"cos": -1.0}
    stale = 0
    stories = list(train)
    for epoch in range(1, MAX_EPOCHS + 1):
        began = time.time()
        random.shuffle(stories)
        model.train()
        losses = []
        for story in stories:
            frames, images = load_video(story)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model.score(story, frames),
                                                                        frame_targets(frames, images))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = validation_cos(model, val)
        gap = max(scores, key=scores.get)
        improved = scores[gap] > best["cos"]
        print(f"  lr {lr} epoch {epoch}: loss {statistics.mean(losses):.4f} val cos {scores[gap]:.2f} gap {gap}"
              f"{' *' if improved else ''} ({time.time() - began:.0f}s)", flush=True)
        if improved:
            best = {"cos": scores[gap], "gap": gap, "epoch": epoch, "lr": lr,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best


@torch.no_grad()
def per_story(model, gap, stories, seed):
    """Cosine per story for the selector and the unsupervised baselines, in story order."""
    model.eval()
    rng = random.Random(seed)
    rows = {"Selector": [], "VSUMM": [], "Uniform": [], "Random": []}
    for story in stories:
        frames, images = load_video(story)
        if len(images) == 0:
            continue
        k = min(len(images), len(frames))
        rows["Selector"].append(cos_for_frames(frames, images, pick_frames(model.score(story, frames), k, gap)))
        rows["VSUMM"].append(cos_for_frames(frames, images, vsumm_choice(frames, k, 42)))
        rows["Uniform"].append(cos_for_frames(frames, images, uniform_choice(len(frames), k)))
        rows["Random"].append(statistics.mean(cos_for_frames(frames, images, rng.sample(range(len(frames)), k))
                                              for _ in range(RANDOM_REPEATS)))
    return {name: 100 * np.array(v) for name, v in rows.items()}


def paired_bootstrap(a, b, seed):
    difference = a - b
    picks = np.random.default_rng(seed).integers(0, len(difference), (BOOTSTRAPS, len(difference)))
    low, high = np.percentile(difference[picks].mean(1), [2.5, 97.5])
    return {"difference": float(difference.mean()), "low": float(low), "high": float(high)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cache = torch.load(CACHE_PATH, weights_only=False)
    stories = cache["stories"]
    make_model = (lambda: ArticleFrameSelector(cache["embeddings"])) if args.article else FrameSelector
    tag = f"{'article_' if args.article else ''}seed{args.seed}"
    split = lambda name: [s for s in stories if s["split_paper"] == name and s["source"] == "dailymail"]
    train = [s for s in split("train") if usable(s)][:args.limit_train]
    val = [s for s in split("val") if usable(s)][:args.limit_val]
    print(f"train {len(train)}, val {len(val)}, seed {args.seed}", flush=True)

    runs = [train_one(make_model, lr, train, val, args.seed) for lr in LEARNING_RATES]
    best = max(runs, key=lambda r: r["cos"])
    print(f"Chosen on validation: lr {best['lr']}, epoch {best['epoch']}, gap {best['gap']}, cos {best['cos']:.2f}",
          flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best, RUN_DIR / f"best_{tag}.pt")
    result = {"seed": args.seed, "n_train": len(train), "n_val": len(val),
              "validation": {r["lr"]: {k: r[k] for k in ("cos", "gap", "epoch")} for r in runs}}

    model = make_model()
    model.load_state_dict(best["state"])
    val_rows = per_story(model, best["gap"], val, args.seed)
    result["validation_methods"] = {name: float(v.mean()) for name, v in val_rows.items()}
    print("Validation:", json.dumps(result["validation_methods"]), flush=True)

    if not args.no_test:
        rows = per_story(model, best["gap"], split("test"), args.seed)
        result["test"] = {name: {"n": len(v), "cos": float(v.mean())} for name, v in rows.items()}
        result["test_selector_minus"] = {name: paired_bootstrap(rows["Selector"], rows[name], args.seed)
                                         for name in ("VSUMM", "Uniform", "Random")}
        print("Test:", json.dumps(result["test"]), flush=True)
        print("Selector minus:", json.dumps(result["test_selector_minus"]), flush=True)
    path = OUT_DIR / f"results_{tag}{'_notest' if args.no_test else ''}.json"
    path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
