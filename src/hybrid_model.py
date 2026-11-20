"""Hybrid of M2SM (Fu et al. 2020) and A2Summ (He et al. 2023) over article sentences, video frames and speech.

From A2Summ: RoBERTa sentence vectors, one transformer over all tokens with shared attention and a feed-forward
network per modality, [CLS] tokens, focal loss on supervised key frames and sentences, and the inter- and
intra-sample contrastive losses. Its alignment-guided attention is applied between frames and speech sentences,
whose Whisper timestamps make the time correspondence available on Daily Mail.
From M2SM: a sentence BiLSTM over the article before the transformer, the transcript as a bridge between article
and video (article -> speech -> frame paths exist inside the attention), and late+ fusion (Eq. 9) for the
sentence scores.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from m2sm_model import Fusion

ARTICLE, VIDEO, SPEECH = 0, 1, 2
MAX_POSITIONS = 1024
ALIGN_SLACK = 1.0        # seconds of tolerance when matching a frame to a speech sentence


class MultiWayBlock(nn.Module):
    """Shared attention over all tokens, then a feed-forward network per modality (A2Summ, Sec. 3.2)."""

    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))
        self.ffns = nn.ModuleList(nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
                                                nn.Linear(4 * hidden, hidden), nn.Dropout(dropout))
                                  for _ in range(3))

    def forward(self, x, modality, blocked):
        y = self.norm_attention(x)
        x = x + self.attention(y, y, y, attn_mask=blocked, need_weights=False)[0]
        out = x.clone()
        for m in range(3):
            where = modality == m
            if where.any():
                out[where] = x[where] + self.ffns[m](self.norms[m](x[where]))
        return out


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """A2Summ's classification loss (focal, alpha 0.25, gamma 2) on sigmoid scores."""
    p = logits.sigmoid()
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (-alpha_t * (1 - p_t) ** gamma * p_t.clamp(min=1e-7).log()).mean()


class Hybrid(nn.Module):
    def __init__(self, hidden=256, heads=4, layers=2, dropout=0.1, fc_dropout=0.5, use_video=True, use_speech=True,
                 sentence_lstm=True, fusion="lateplus", align=True, beta=0.3, ratio=8):
        super().__init__()
        self.use_video, self.use_speech, self.use_lstm, self.align = use_video, use_speech, sentence_lstm, align
        self.ratio = ratio
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(768, hidden), nn.Dropout(dropout)),
                                      nn.Sequential(nn.Linear(2048, hidden), nn.Dropout(dropout)),
                                      nn.Sequential(nn.Linear(768, hidden), nn.Dropout(dropout))])
        if sentence_lstm:
            self.article_lstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.positions = nn.Parameter(torch.zeros(3, MAX_POSITIONS, hidden))
        self.types = nn.Parameter(torch.zeros(3, hidden))
        self.cls = nn.Parameter(torch.zeros(3, hidden))
        for p in (self.positions, self.types, self.cls):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(MultiWayBlock(hidden, heads, dropout) for _ in range(layers))
        self.out_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))
        self.fusion = Fusion(fusion, hidden, hidden, beta) if fusion != "none" else None
        head = lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(fc_dropout), nn.Linear(hidden, 1))
        self.article_head = head() if self.fusion is None else None
        self.frame_head = head()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def tokens(self, sample):
        """Token states, modality ids and times for one story, [CLS] first within each modality."""
        parts, modality, times = [], [], []
        article = self.project[ARTICLE](sample["article"])
        if self.use_lstm:
            article = self.article_lstm(article.unsqueeze(0))[0].squeeze(0)
        sources = [(ARTICLE, article, None)]
        if self.use_video:
            sources.append((VIDEO, self.project[VIDEO](sample["frames"]), sample["frame_times"]))
        if self.use_speech and sample.get("speech") is not None:
            sources.append((SPEECH, self.project[SPEECH](sample["speech"]), sample["speech_times"]))
        for m, states, when in sources:
            states = torch.cat([self.cls[m].unsqueeze(0), states]) + self.positions[m, :len(states) + 1] + self.types[m]
            parts.append(states)
            modality.append(torch.full((len(states),), m))
            span = torch.full((len(states), 2), float("nan"))
            if when is not None:
                span[1:] = when if when.dim() == 2 else when.unsqueeze(1).expand(-1, 2)
            times.append(span)
        return torch.cat(parts), torch.cat(modality), torch.cat(times), article

    def blocked_pairs(self, modality, times):
        """True where attention is not allowed: frames and speech sentences only meet when their times overlap."""
        n = len(modality)
        blocked = torch.zeros(n, n, dtype=torch.bool)
        if not self.align:
            return blocked
        frame = (modality == VIDEO) & ~times[:, 0].isnan()
        speech = (modality == SPEECH) & ~times[:, 0].isnan()
        if not (frame.any() and speech.any()):
            return blocked
        f_idx, s_idx = frame.nonzero().squeeze(1), speech.nonzero().squeeze(1)
        t = times[f_idx, 0].unsqueeze(1)
        start, end = times[s_idx, 0].unsqueeze(0), times[s_idx, 1].unsqueeze(0)
        overlap = (t >= start - ALIGN_SLACK) & (t <= end + ALIGN_SLACK)
        lonely = ~overlap.any(0)                     # speech past the video's end: nearest frame instead
        if lonely.any():
            nearest = (t - start).abs().argmin(0)
            overlap[nearest[lonely], lonely.nonzero().squeeze(1)] = True
        block = ~overlap
        blocked[f_idx.unsqueeze(1), s_idx.unsqueeze(0)] = block
        blocked[s_idx.unsqueeze(1), f_idx.unsqueeze(0)] = block.T
        return blocked

    def forward(self, batch):
        """Scores and embeddings for a list of stories. Padding is per batch; padded keys are blocked."""
        encoded = [self.tokens(sample) for sample in batch]
        length = max(len(e[0]) for e in encoded)
        hidden = encoded[0][0].shape[1]
        x = torch.zeros(len(batch), length, hidden)
        modality = torch.full((len(batch), length), -1)
        blocked = torch.ones(len(batch), length, length, dtype=torch.bool)
        for i, (states, m, times, _) in enumerate(encoded):
            n = len(states)
            x[i, :n], modality[i, :n] = states, m
            blocked[i, :n, :n] = self.blocked_pairs(m, times)
        blocked[:, torch.arange(length), torch.arange(length)] = False      # every row keeps itself
        heads = self.blocks[0].attention.num_heads
        mask = blocked.repeat_interleave(heads, 0)
        for block in self.blocks:
            x = block(x, modality.clamp(min=0), mask)

        outputs = []
        for i, (states, m, _, article_in) in enumerate(encoded):
            y = x[i, :len(states)].clone()
            for k in range(3):
                if (m == k).any():
                    y[m == k] = self.out_norms[k](y[m == k])
            article = y[m == ARTICLE]
            cls_text, sentences = article[0], article[1:]
            if self.fusion is not None:
                sentence_logits = self.fusion(article_in, sentences)
            else:
                sentence_logits = self.article_head(sentences).squeeze(-1)
            out = {"sentence_logits": sentence_logits, "sentences": sentences, "cls_text": cls_text}
            if self.use_video:
                video = y[m == VIDEO]
                out.update(cls_video=video[0], frames=video[1:], frame_logits=self.frame_head(video[1:]).squeeze(-1))
            outputs.append(out)
        return outputs

    def hard_negatives(self, logits, labels, states):
        """Highest-scoring non-key units outside 4 steps of any key (A2Summ, Sec. 3.4 and Appendix B)."""
        near = torch.from_numpy(ndimage.binary_dilation(labels.numpy() > 0, iterations=4))
        count = max(1, len(labels) // self.ratio)
        order = [i for i in logits.argsort(descending=True).tolist() if not near[i]][:count]
        return states[order] if order else states[logits.argmin().unsqueeze(0)]

    def intra_loss(self, out, sample):
        """Key sentences and key frames of one story pulled together, hard non-key units pushed away (InfoNCE)."""
        sentence_labels, frame_labels = sample["sentence_labels"], sample.get("frame_labels")
        if frame_labels is None or sentence_labels.sum() == 0:
            return None
        temperature = self.log_temperature.exp()
        losses = []
        for anchors, keys, negatives in (
                (out["frames"][frame_labels > 0], out["sentences"][sentence_labels > 0],
                 self.hard_negatives(out["sentence_logits"].detach(), sentence_labels, out["sentences"])),
                (out["sentences"][sentence_labels > 0], out["frames"][frame_labels > 0],
                 self.hard_negatives(out["frame_logits"].detach(), frame_labels, out["frames"]))):
            anchor = F.normalize(anchors.mean(0), dim=0)
            positive = F.normalize(keys, dim=1) @ anchor * temperature
            negative = F.normalize(negatives, dim=1) @ anchor * temperature
            losses.append(-(positive - torch.logsumexp(torch.cat([positive, negative]), 0)).mean())
        return sum(losses) / 2

    def inter_loss(self, outputs):
        """CLIP-style loss between each story's video and text [CLS] tokens and the others in the batch."""
        if len(outputs) < 2:
            return None
        text = F.normalize(torch.stack([o["cls_text"] for o in outputs]), dim=1)
        video = F.normalize(torch.stack([o["cls_video"] for o in outputs]), dim=1)
        logits = text @ video.T * self.log_temperature.exp()
        target = torch.arange(len(outputs))
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
