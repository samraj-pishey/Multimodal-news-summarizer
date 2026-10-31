from pathlib import Path
from itertools import permutations
import csv
import json
import re
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
CLEAN_DIR = PROJECT_ROOT / "outputs" / "full_run_v3"      # the exact text the BART baseline saw
OUT_DIR = PROJECT_ROOT / "outputs" / "clip"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
# CLIP reads at most 77 tokens, so the article is represented by its title and lead.
QUERY_SENTENCES = 2


def read_lines(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\xa0", " ")
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def article_query(row):
    lead = re.split(r"(?<=[.!?])\s+", row["article"])[:QUERY_SENTENCES]
    return " ".join([row["title"]] + lead)


def load_stories():
    stories = []
    for split in ("train", "val", "test"):
        for line in open(CLEAN_DIR / f"clean_{split}.jsonl"):
            row = json.loads(line)
            if row["source"] != "dailymail":    # only Daily Mail stories have images
                continue
            folder = DATA_ROOT / "daily mail" / row["id"].split("/")[1]
            images = sorted(folder.glob("*.png"), key=lambda p: int(p.stem)) + sorted(folder.glob("*.jpg"))
            stories.append({"id": row["id"], "split": split, "query": article_query(row),
                            "images": images, "captions": read_lines(folder / "img_caption.txt")})
    return stories


class Clip:
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def text(self, texts):
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self.processor(text=texts[i:i + BATCH_SIZE], padding=True, truncation=True,
                                 max_length=77, return_tensors="pt").to(self.device)
            # transformers 5 returns an output object; pooler_output is the projected embedding.
            out.append(torch.nn.functional.normalize(self.model.get_text_features(**enc).pooler_output, dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, self.model.config.projection_dim)

    @torch.no_grad()
    def images(self, paths):
        out = []
        for i in range(0, len(paths), BATCH_SIZE):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE]]
            enc = self.processor(images=batch, return_tensors="pt").to(self.device)
            out.append(torch.nn.functional.normalize(self.model.get_image_features(**enc).pooler_output, dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, self.model.config.projection_dim)


def align_captions(n_images, n_captions, sim):
    """Map caption index -> image index.

    Captions follow image order when the counts match; otherwise pick the
    one-to-one assignment with the highest total image-caption similarity.
    """
    if n_images == 0 or n_captions == 0:
        return [None] * n_captions
    if n_images == n_captions:
        return list(range(n_captions))
    best, best_score = None, float("-inf")
    k = min(n_images, n_captions)
    for caps in permutations(range(n_captions), k):
        for imgs in permutations(range(n_images), k):
            score = sum(sim[i, c] for c, i in zip(caps, imgs))
            if score > best_score:
                best, best_score = list(zip(caps, imgs)), score
    mapping = [None] * n_captions
    for c, i in best:
        mapping[c] = i
    return mapping


def summaries_for_test():
    """Generated and reference summaries per test story, for cover-image selection."""
    rows = csv.DictReader(open(CLEAN_DIR / "test_predictions.csv"))
    return {r["id"]: {"finetuned": r["finetuned"], "pretrained": r["pretrained"], "reference": r["reference"]}
            for r in rows}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    stories = load_stories()
    test_summaries = summaries_for_test()
    print(f"{len(stories)} Daily Mail stories, {sum(len(s['images']) for s in stories)} images | device {device}")

    clip = Clip(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(OUT_DIR / "clip_scores.jsonl", "w") as out:
        for n, s in enumerate(stories, 1):
            img = clip.images(s["images"])
            query = clip.text([s["query"]])
            cap = clip.text(s["captions"])
            image_caption = img @ cap.T

            record = {
                "id": s["id"], "split": s["split"],
                "images": [{"file": p.name, "sim_article": round(float(v), 4)}
                           for p, v in zip(s["images"], (img @ query.T).squeeze(1).tolist())],
                "captions": [{"text": c, "image": i, "sim_article": round(float(v), 4)}
                             for c, i, v in zip(s["captions"],
                                                align_captions(len(s["images"]), len(s["captions"]), image_caption),
                                                (cap @ query.T).squeeze(1).tolist())],
            }
            if s["id"] in test_summaries and len(s["images"]):
                names = ("finetuned", "pretrained", "reference")
                summ = clip.text([test_summaries[s["id"]][k] for k in names])
                for image, sims in zip(record["images"], (img @ summ.T).tolist()):
                    image.update({f"sim_{k}": round(v, 4) for k, v in zip(names, sims)})
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if n % 200 == 0:
                print(f"  {n}/{len(stories)} stories, {(time.time() - t0) / 60:.1f} min")

    meta = {"model": MODEL_NAME, "query": f"title + first {QUERY_SENTENCES} sentences of the cleaned article",
            "stories": len(stories), "minutes": round((time.time() - t0) / 60, 1)}
    (OUT_DIR / "clip_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR / 'clip_scores.jsonl'} in {meta['minutes']} min")


if __name__ == "__main__":
    main()
