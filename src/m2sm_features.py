"""ResNet-50 features for every MM-AVS video and reference image (M2SM, Fu et al. 2020, Sec. 3.1).

Frames are sampled at 1 per second, a deliberate change from the paper's "one of five consecutive frames",
and stored as float16 next to their timestamps.
Resumable: stories whose .npz already exists are skipped.
"""
from pathlib import Path
import re
import time

import av
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
OUT_DIR = PROJECT_ROOT / "data" / "m2sm_features"
BATCH_SIZE = 64


class Encoder:
    def __init__(self, device):
        weights = ResNet50_Weights.IMAGENET1K_V1      # the weights available when the paper was written
        self.device = device
        self.transform = weights.transforms()
        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.eval().to(device)

    @torch.inference_mode()
    def encode(self, tensors):
        return self.model(torch.stack(tensors).to(self.device)).cpu().numpy().astype(np.float16)


def video_files(folder):
    if (folder / "video.mp4").exists():
        return [folder / "video.mp4"]
    segments = (folder / "video").glob("segment*.ts")
    return sorted(segments, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


def encode_videos(paths, encoder):
    features, times, batch, offset = [], [], [], 0.0
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            start, next_second, last = None, 0.0, 0.0
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                start = frame.time if start is None else start
                last = frame.time - start
                if last < next_second:
                    continue
                batch.append(encoder.transform(frame.to_image()))
                times.append(offset + last)
                next_second += 1.0
                if len(batch) == BATCH_SIZE:
                    features.append(encoder.encode(batch))
                    batch = []
        offset += last + 1.0
    if batch:
        features.append(encoder.encode(batch))
    frames = np.concatenate(features) if features else np.empty((0, 2048), np.float16)
    return frames, np.array(times, np.float32)


def encode_images(folder, encoder):
    paths = sorted(folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10 ** 6)
    paths += sorted(folder.glob("*.jpg"))
    features = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = []
        for path in paths[i:i + BATCH_SIZE]:
            try:
                batch.append(encoder.transform(Image.open(path).convert("RGB")))
            except OSError as error:
                print(f"  unreadable image {path}: {error}", flush=True)
        if batch:
            features.append(encoder.encode(batch))
    return np.concatenate(features) if features else np.empty((0, 2048), np.float16)


def story_folders():
    for source, parent in (("dailymail", DATA_ROOT / "daily mail"), ("cnn", DATA_ROOT / "cnn_data")):
        for folder in sorted((p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()),
                             key=lambda p: int(p.name)):
            yield source, folder


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = Encoder(device)
    stories = list(story_folders())
    print(f"{len(stories)} story folders | device {device}", flush=True)

    t0, done, failed = time.time(), 0, 0
    for n, (source, folder) in enumerate(stories, 1):
        out = OUT_DIR / f"{source}_{folder.name}.npz"
        if out.exists():
            continue
        try:
            frames, times = encode_videos(video_files(folder), encoder)
            images = encode_images(folder, encoder)
        except (av.FFmpegError, OSError, RuntimeError) as error:
            failed += 1
            print(f"  failed {source}/{folder.name}: {error}", flush=True)
            continue
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, frames=frames, times=times, images=images)
        tmp.rename(out)
        done += 1
        if done % 50 == 0:
            print(f"  {n}/{len(stories)} ({done} new, {failed} failed), {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"Finished: {done} new, {failed} failed, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
