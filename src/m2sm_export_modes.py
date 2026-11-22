"""Export the article-only and video-only runs for webapp/, next to the article + video model.

Each run trained its own word vectors, so each gets its own embeddings and word encoder. Every export is checked
against PyTorch on validation stories before it is kept in config.json.

Usage: python src/m2sm_export_modes.py [--text text_only_lr0.01] [--speech speech_video_tuned]
"""
import argparse
import json

import numpy as np
import onnxruntime
import torch

from m2sm_data import CACHE_PATH, SPEECH_CACHE_PATH
from m2sm_export_demo import load_model
from m2sm_export_web import OUT_DIR, Summarizer, WordEncoder, export
from m2sm_train import load_frames

CHECK_STORIES = 10


class SentenceScorer(Summarizer):
    """Only the sentence logits: these runs have no frame head."""

    def forward(self, pooled, frames=None):
        return self.model(pooled, frames, None)[0]


def export_mode(name, run, embeddings, stories):
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, config, size = load_model(run, embeddings)
    trained = model.embedding.weight.detach().clone()
    uses_video = not config["config"]["no_video"]

    original = {s["id"]: logits_from_pytorch(model, s, uses_video) for s in stories}

    export(WordEncoder(model.word_lstm).eval(), (torch.zeros(1, 7, 100),), ["embedded"],
           {"outputs": ["pooled"], "dynamic": {"embedded": {1: "words"}}}, out_dir / "word_encoder.onnx")
    dynamic = {"pooled": {0: "sentences"}, "sentence_logits": {0: "sentences"}}
    inputs, names = (torch.zeros(5, 128),), ["pooled"]
    if uses_video:
        inputs, names = (*inputs, torch.zeros(9, 2048)), [*names, "frames"]
        dynamic["frames"] = {0: "frames"}
    export(SentenceScorer(model, False).eval(), inputs, names, {"outputs": ["sentence_logits"], "dynamic": dynamic},
           out_dir / "summarizer.onnx")
    trained.numpy().astype(np.float32).tofile(out_dir / "embeddings.bin")

    largest = check_export(out_dir, trained, stories, original, uses_video)
    print(f"{name} ({run}): largest difference from PyTorch {largest:.6f} over {len(stories)} stories", flush=True)
    if largest > 1e-3:
        raise SystemExit(f"{name}: the exported model disagrees with PyTorch")
    return {"run": run, "summary_size": size, "best_val": config["best_val"], "test_dailymail": config["test_dailymail"]}


@torch.no_grad()
def logits_from_pytorch(model, story, uses_video):
    sentence_ids = [torch.tensor(ids) for ids in story["sentence_ids"]]
    if uses_video:
        return model(sentence_ids, load_frames(story["id"]), None)[0].numpy()
    return model(sentence_ids)[0].numpy()


def check_export(out_dir, embeddings, stories, original, uses_video):
    """Runs the ONNX files the way worker.js does: one unpadded sentence at a time, then the summarizer."""
    providers = ["CPUExecutionProvider"]
    words = onnxruntime.InferenceSession(str(out_dir / "word_encoder.onnx"), providers=providers)
    summarizer = onnxruntime.InferenceSession(str(out_dir / "summarizer.onnx"), providers=providers)
    table = embeddings.numpy()
    largest = 0.0
    for story in stories:
        pooled = [words.run(None, {"embedded": table[np.array(ids)][None]})[0][0] for ids in story["sentence_ids"]]
        inputs = {"pooled": np.stack(pooled)}
        if uses_video:
            inputs["frames"] = load_frames(story["id"]).numpy()
        logits = summarizer.run(None, inputs)[0]
        largest = max(largest, float(np.abs(logits - original[story["id"]]).max()))
    return largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="text_only_lr0.01")
    parser.add_argument("--speech", default="speech_video_tuned")
    args = parser.parse_args()
    torch.set_num_threads(2)

    cache = torch.load(CACHE_PATH, weights_only=False)
    articles = [s for s in cache["stories"] if s["split_paper"] == "val"][:CHECK_STORIES]
    speech = [s for s in torch.load(SPEECH_CACHE_PATH, weights_only=False) if s["split_paper"] == "val"][:CHECK_STORIES]

    config_path = OUT_DIR / "config.json"
    settings = json.loads(config_path.read_text())
    settings["modes"] = {"text": export_mode("text", args.text, cache["embeddings"], articles),
                         "speech": export_mode("speech", args.speech, cache["embeddings"], speech)}
    config_path.write_text(json.dumps(settings))
    for folder in ("text", "speech"):
        for path in sorted((OUT_DIR / folder).iterdir()):
            print(f"  {folder}/{path.name}: {path.stat().st_size / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()
