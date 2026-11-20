"""Export one finished M2SM run and ResNet-50 to ONNX for the in-browser app in webapp/.

The word and sentence stages are split so the browser can run each sentence unpadded, which gives the same
states as the packed sequences used in training. ResNet-50 is quantized against frames from training stories.

Usage: python src/m2sm_export_web.py --run m2sm_full_lr0.01 [--calibration-stories 40]
"""
from pathlib import Path
import argparse
import json
import random

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from m2sm_data import CACHE_PATH, MAX_SENTENCE_WORDS, MAX_SENTENCES, MAX_TRANSCRIPT_WORDS, PROJECT_ROOT, feature_path
from m2sm_export_demo import load_model
from m2sm_features import Encoder, video_files

import av

OUT_DIR = PROJECT_ROOT / "webapp" / "models"
OPSET = 17
FRAMES_PER_STORY = 5


class WordEncoder(torch.nn.Module):
    """Embedded words of one sentence to its pooled state (Eq. 1, word level)."""

    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, embedded):
        return self.lstm(embedded)[0].mean(1)


class Summarizer(torch.nn.Module):
    """M2SM.forward from the pooled sentences on, with embedding done by the page."""

    def __init__(self, model, transcript):
        super().__init__()
        self.model, self.transcript = model, transcript
        model.embedding = torch.nn.Identity()
        model.encode_text = lambda pooled: model.sentence_lstm(pooled.unsqueeze(0))[0].squeeze(0)

    def forward(self, pooled, frames, words=None):
        return self.model(pooled, frames, words if self.transcript else None)


def export(module, inputs, names, axes, path):
    torch.onnx.export(module, inputs, str(path), input_names=names, output_names=axes["outputs"],
                      dynamic_axes=axes["dynamic"], opset_version=OPSET, dynamo=False)


def export_m2sm(run, embeddings, vocabulary, out_dir):
    model, config, size = load_model(run, embeddings)
    if config["config"]["no_video"] or config["config"]["no_bistream"]:
        raise SystemExit(f"{run} doesn't select frames, so the app can't use it")
    trained_embeddings = model.embedding.weight.detach().clone()      # training updates them, so not the cache's

    export(WordEncoder(model.word_lstm).eval(), (torch.zeros(1, 7, 100),), ["embedded"],
           {"outputs": ["pooled"], "dynamic": {"embedded": {1: "words"}}}, out_dir / "word_encoder.onnx")

    dynamic = {"pooled": {0: "sentences"}, "frames": {0: "frames"}, "words": {0: "words"},
               "sentence_logits": {0: "sentences"}, "frame_logits": {0: "frames"}}
    pooled, frames, words = torch.zeros(5, 128), torch.zeros(9, 2048), torch.zeros(11, 100)
    outputs = ["sentence_logits", "frame_logits"]
    export(Summarizer(model, True).eval(), (pooled, frames, words), ["pooled", "frames", "words"],
           {"outputs": outputs, "dynamic": dynamic}, out_dir / "summarizer_transcript.onnx")
    no_words = {k: v for k, v in dynamic.items() if k != "words"}
    export(Summarizer(model, False).eval(), (pooled, frames), ["pooled", "frames"],
           {"outputs": outputs, "dynamic": no_words}, out_dir / "summarizer_video.onnx")

    trained_embeddings.numpy().astype(np.float32).tofile(out_dir / "embeddings.bin")
    settings = {"run": run, "summary_size": size, "learning_rate": config["config"]["lr"],
                "max_sentences": MAX_SENTENCES, "max_sentence_words": MAX_SENTENCE_WORDS,
                "max_transcript_words": MAX_TRANSCRIPT_WORDS, "embedding_dim": embeddings.shape[1],
                "best_val": config["best_val"], "test_dailymail": config["test_dailymail"]}
    (out_dir / "config.json").write_text(json.dumps(settings))
    (out_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False))


def calibration_frames(stories, encoder, count):
    """A few evenly spaced frames from each sampled training story."""
    picked = random.Random(42).sample(stories, count)
    batches = []
    for story in picked:
        source, number = story["id"].split("/")
        folder = PROJECT_ROOT / "data" / "data" / "daily mail" / number
        frames = []
        with av.open(str(video_files(folder)[0])) as container:
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                if i % 150 == 0:
                    frames.append(encoder.transform(frame.to_image()))
                if len(frames) == FRAMES_PER_STORY:
                    break
        batches += [t.unsqueeze(0).numpy() for t in frames]
    return batches


class FrameReader(CalibrationDataReader):
    def __init__(self, batches):
        self.items = iter({"images": b} for b in batches)

    def get_next(self):
        return next(self.items, None)


def export_resnet(train_stories, calibration_stories, out_dir):
    encoder = Encoder("cpu")
    raw, prepared = out_dir / "resnet_raw.onnx", out_dir / "resnet_prepared.onnx"
    export(encoder.model, (torch.zeros(1, 3, 224, 224),), ["images"],
           {"outputs": ["features"], "dynamic": {"images": {0: "batch"}, "features": {0: "batch"}}}, raw)
    quant_pre_process(str(raw), str(prepared))
    batches = calibration_frames(train_stories, encoder, calibration_stories)
    quantize_static(str(prepared), str(out_dir / "resnet.onnx"), FrameReader(batches), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)
    prepared.unlink()
    print(f"Calibrated on {len(batches)} frames from training stories", flush=True)
    return raw, encoder


def compare_resnet(raw, stories, encoder, out_dir):
    """Cosine between float and quantized features, on test frames the calibration didn't see."""
    full = onnxruntime.InferenceSession(str(raw), providers=["CPUExecutionProvider"])
    small = onnxruntime.InferenceSession(str(out_dir / "resnet.onnx"), providers=["CPUExecutionProvider"])
    values = []
    for batch in calibration_frames(stories, encoder, 5):
        a, b = full.run(None, {"images": batch})[0], small.run(None, {"images": batch})[0]
        values.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"Quantized vs float ResNet cosine: mean {np.mean(values):.4f}, min {min(values):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--calibration-stories", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(2)

    cache = torch.load(CACHE_PATH, weights_only=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_m2sm(args.run, cache["embeddings"], cache["vocabulary"], OUT_DIR)

    daily_mail = [s for s in cache["stories"] if s["source"] == "dailymail" and feature_path(s["id"]).exists()]
    train = [s for s in daily_mail if s["split_paper"] == "train"]
    test = [s for s in daily_mail if s["split_paper"] == "test"]
    raw, encoder = export_resnet(train, args.calibration_stories, OUT_DIR)
    compare_resnet(raw, test, encoder, OUT_DIR)
    raw.unlink()
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name}: {path.stat().st_size / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()
