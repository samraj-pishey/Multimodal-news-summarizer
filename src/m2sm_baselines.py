"""Baselines that need no multi-modal training (Fu et al. 2020, Tables 3, 4 and 7).

Text: lead3 and the extraction oracle given by the labels. Video: Random, Uniform, VSUMM and DR-DSN,
scored with the same cosine as M2SM. Also the transcript overlap statistics of Table 7.
"""
import json
import random
import statistics

import numpy as np
import torch
from sklearn.cluster import KMeans

from m2sm_data import CACHE_PATH, PROJECT_ROOT, feature_path, score_summary, scorer
from m2sm_model import normalized, reinforce_loss

OUT_PATH = PROJECT_ROOT / "outputs" / "m2sm" / "baselines.json"
RANDOM_REPEATS = 20
DR_DSN_EPOCHS = 8


def mean_scores(rows):
    return {"n": len(rows), **{k: 100 * statistics.mean(r[k] for r in rows) for k in ("r1", "r2", "rl")}}


def text_baselines(stories):
    lead3 = [score_summary(s["reference"], s["sentences"][:3]) for s in stories]
    oracle = [score_summary(s["reference"], [t for t, y in zip(s["sentences"], s["labels"]) if y] or s["sentences"][:1])
              for s in stories]
    return {"lead3": mean_scores(lead3), "oracle": mean_scores(oracle)}


def transcript_overlap(stories):
    """Table 7: ROUGE F1 of each transcript against its article and against its reference."""
    rows = {"article": [], "reference": []}
    for story in stories:
        if not story["transcript"]:
            continue
        rows["article"].append(score_summary(" ".join(story["sentences"]), [story["transcript"]]))
        rows["reference"].append(score_summary(story["reference"], [story["transcript"]]))
    return {k: mean_scores(v) for k, v in rows.items()}


def load_video(story):
    data = np.load(feature_path(story["id"]))
    frames = data["frames"].astype(np.float32)
    return normalized(torch.from_numpy(frames)), normalized(torch.from_numpy(data["images"].astype(np.float32)))


def cos_for_frames(frames, images, chosen):
    return (images @ frames[chosen].T).max(1).values.mean().item()


def uniform_choice(n_frames, k):
    return np.linspace(0, n_frames - 1, k + 2)[1:-1].round().astype(int).tolist()


def vsumm_choice(frames, k, seed):
    """VSUMM clusters frames and keeps the frame nearest each centre. ResNet replaces its colour histograms."""
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(frames.numpy())
    return [int(np.linalg.norm(frames.numpy() - c, axis=1).argmin()) for c in km.cluster_centers_]


class DrDsn(torch.nn.Module):
    """Frame scorer of Zhou et al. (2018): a BiLSTM with the same rewards M2SM uses."""

    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(2048, hidden, batch_first=True, bidirectional=True)
        self.head = torch.nn.Linear(2 * hidden, 1)

    def forward(self, frames):
        return self.head(self.lstm(frames.unsqueeze(0))[0].squeeze(0)).squeeze(-1)


def train_dr_dsn(stories, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    model = DrDsn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    baseline = 0.0
    stories = list(stories)
    for epoch in range(DR_DSN_EPOCHS):
        random.shuffle(stories)
        rewards = []
        for story in stories:
            frames, _ = load_video(story)
            loss, reward = reinforce_loss(model(frames), frames, baseline)
            baseline = 0.9 * baseline + 0.1 * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rewards.append(reward)
        print(f"  DR-DSN epoch {epoch + 1}: reward {statistics.mean(rewards):.4f}", flush=True)
    return model.eval()


def video_baselines(train, test):
    model = train_dr_dsn(train)
    totals = {"Random": [], "Uniform": [], "VSUMM": [], "DR-DSN": []}
    rng = random.Random(42)
    for story in test:
        frames, images = load_video(story)
        if len(images) == 0:
            continue
        k = min(len(images), len(frames))
        totals["Random"].append(statistics.mean(cos_for_frames(frames, images, rng.sample(range(len(frames)), k))
                                                for _ in range(RANDOM_REPEATS)))
        totals["Uniform"].append(cos_for_frames(frames, images, uniform_choice(len(frames), k)))
        totals["VSUMM"].append(cos_for_frames(frames, images, vsumm_choice(frames, k, 42)))
        with torch.no_grad():
            top = model(frames).topk(k).indices.tolist()
        totals["DR-DSN"].append(cos_for_frames(frames, images, top))
    return {name: {"n": len(v), "cos": 100 * statistics.mean(v)} for name, v in totals.items()}


def main():
    torch.set_num_threads(3)
    cache = torch.load(CACHE_PATH, weights_only=False)
    stories = cache["stories"]
    test = lambda source: [s for s in stories if s["split_paper"] == "test" and s["source"] == source]
    results = {"text_dailymail": text_baselines(test("dailymail")), "text_cnn": text_baselines(test("cnn")),
               "transcript_overlap_dailymail": transcript_overlap(test("dailymail")),
               "transcript_overlap_cnn": transcript_overlap(test("cnn"))}
    print(json.dumps(results, indent=1), flush=True)
    results["video"] = video_baselines([s for s in stories if s["split_paper"] == "train"], test("dailymail"))
    print(json.dumps(results["video"], indent=1), flush=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
