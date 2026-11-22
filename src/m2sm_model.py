"""M2SM (Fu et al. 2020): hierarchical text encoder, bi-hop attention, late+ fusion, bi-stream heads.

Equation numbers in comments refer to the paper. Each component can be switched off for the ablation
and sub-module comparisons (Tables 5 and 6, Fig. 4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def bilstm_states(lstm, sequences):
    """Runs a BiLSTM over a list of variable-length tensors. Returns padded states and lengths."""
    lengths = torch.tensor([len(s) for s in sequences])
    packed = pack_padded_sequence(pad_sequence(sequences, batch_first=True), lengths, batch_first=True,
                                  enforce_sorted=False)
    states, _ = pad_packed_sequence(lstm(packed)[0], batch_first=True)
    return states, lengths


class BilinearAttention(nn.Module):
    """Single feature projection plus dual feature interaction (Eq. 4-6)."""

    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def weights(self, queries, keys):
        q = torch.tanh(self.key(keys)).unsqueeze(0)            # 1 x K x D
        r = torch.tanh(self.query(queries)).unsqueeze(1)       # Q x 1 x D
        return self.v(q * r + q + r).squeeze(-1).softmax(-1)

    def forward(self, queries, keys, values):
        return self.weights(queries, keys) @ values            # Eq. 7


class ConcatAttention(nn.Module):
    """The conventional additive attention of Eq. 3, the paper's concat-product."""

    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, queries, keys, values):
        energy = torch.tanh(self.query(queries).unsqueeze(1) + self.key(keys).unsqueeze(0))
        return self.v(energy).squeeze(-1).softmax(-1) @ values


def feed_forward(in_dim, hidden, out_dim=1):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))


class Fusion(nn.Module):
    """Early, tensor, late and late+ fusion of a unit state with its cross-modal context (Sec. 3.3)."""

    TENSOR_DIM = 32     # outer product of the projections stays small; the paper gives no size

    def __init__(self, kind, dim, hidden, beta):
        super().__init__()
        self.kind, self.beta = kind, beta
        if kind == "early":
            self.net = feed_forward(2 * dim, hidden)
        elif kind == "tensor":
            self.project_a = nn.Linear(dim, self.TENSOR_DIM)
            self.project_b = nn.Linear(dim, self.TENSOR_DIM)
            self.net = feed_forward((self.TENSOR_DIM + 1) ** 2, hidden)
        elif kind in ("late", "lateplus"):
            self.f = feed_forward(dim, hidden)
            self.g = feed_forward(dim, hidden)
            self.net = feed_forward(2, hidden)
        else:
            raise ValueError(kind)

    def forward(self, a, b):
        if self.kind == "early":
            return self.net(torch.cat([a, b], -1)).squeeze(-1)
        if self.kind == "tensor":
            one = a.new_ones(len(a), 1)
            left = torch.cat([self.project_a(a), one], -1).unsqueeze(2)
            right = torch.cat([self.project_b(b), one], -1).unsqueeze(1)
            return self.net((left * right).flatten(1)).squeeze(-1)
        f, g = self.f(a).sigmoid(), self.g(b).sigmoid()
        if self.kind == "lateplus":               # Eq. 9
            w_a = (1 - g).clamp(min=1e-4) ** self.beta
            w_b = (1 - f).clamp(min=1e-4) ** self.beta
            f, g = w_a * f, w_b * g
        return self.net(torch.cat([f, g], -1)).squeeze(-1)


class M2SM(nn.Module):
    def __init__(self, embeddings, hidden=64, video=True, transcript=True, bistream=True,
                 attention="bilinear", fusion="lateplus", beta=0.3, sentence_dim=None):
        super().__init__()
        self.use_video, self.use_transcript, self.use_bistream = video, transcript, bistream
        self.attention_kind = attention
        dim = 2 * hidden

        self.embedding = nn.Embedding.from_pretrained(embeddings.clone(), freeze=False, padding_idx=0)
        if sentence_dim:        # fixed sentence vectors (e.g. RoBERTa) replace the GloVe word BiLSTM
            self.sentence_project = nn.Sequential(nn.Linear(sentence_dim, dim), nn.Tanh())
        else:
            self.word_lstm = nn.LSTM(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
        self.sentence_lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)

        if not video:
            self.text_head = feed_forward(dim, hidden)
            return

        self.frame_lstm = nn.LSTM(2048, hidden, batch_first=True, bidirectional=True)
        make = {"bilinear": BilinearAttention, "concat": ConcatAttention}.get(attention)
        if make:
            self.sentence_to_frame = make(dim)
            if bistream:
                self.frame_to_sentence = make(dim)
            if transcript:
                self.transcript_lstm = nn.LSTM(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
                self.sentence_to_word = make(dim)
                self.word_to_frame = make(dim)
                if bistream:
                    self.frame_to_word = make(dim)
                    self.word_to_sentence = make(dim)
        self.text_fusion = Fusion(fusion, dim, hidden, beta)
        if bistream:
            self.frame_fusion = Fusion(fusion, dim, hidden, beta)

    def encode_text(self, sentence_ids):
        """Word and sentence BiLSTMs with average pooling between them (Eq. 1).
        Given a [sentences x D] tensor of fixed sentence vectors, a projection replaces the word level."""
        if isinstance(sentence_ids, torch.Tensor) and sentence_ids.is_floating_point():
            return self.sentence_lstm(self.sentence_project(sentence_ids).unsqueeze(0))[0].squeeze(0)
        embedded = [self.embedding(ids) for ids in sentence_ids]
        states, lengths = bilstm_states(self.word_lstm, embedded)
        mask = (torch.arange(states.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        pooled = (states * mask).sum(1) / lengths.unsqueeze(1)
        return self.sentence_lstm(pooled.unsqueeze(0))[0].squeeze(0)

    def forward(self, sentence_ids, frames=None, transcript_ids=None):
        """Returns one logit per sentence and, for bi-stream, one per frame."""
        s = self.encode_text(sentence_ids)
        if not self.use_video:
            return self.text_head(s).squeeze(-1), None

        m = self.frame_lstm(frames.unsqueeze(0))[0].squeeze(0)     # Eq. 2
        if self.attention_kind == "none":                          # last state as the video feature
            sentence_context = m[-1].expand_as(s)
            frame_context = s[-1].expand_as(m)
        else:
            has_words = self.use_transcript and transcript_ids is not None and len(transcript_ids) > 0
            if has_words:                                          # bi-hop: sentence -> word -> frame (Eq. 8)
                t = self.transcript_lstm(self.embedding(transcript_ids).unsqueeze(0))[0].squeeze(0)
                word_context = self.word_to_frame(t, m, m)
                sentence_context = self.sentence_to_word(s, t, word_context)
            else:                                                  # no transcript for this story: single hop
                sentence_context = self.sentence_to_frame(s, m, m)
            frame_context = None
            if self.use_bistream:
                if has_words:                                      # the same two hops, reversed
                    frame_context = self.frame_to_word(m, t, self.word_to_sentence(t, s, s))
                else:
                    frame_context = self.frame_to_sentence(m, s, s)

        sentence_logits = self.text_fusion(s, sentence_context)
        frame_logits = self.frame_fusion(m, frame_context) if self.use_bistream else None
        return sentence_logits, frame_logits


def video_reward(selected, features):
    """Diversity plus representativeness rewards over L2-normalized ResNet features (Eq. 11)."""
    picks = features[selected]
    if len(picks) == 0:
        return features.new_zeros(())
    if len(picks) == 1:
        diversity = features.new_zeros(())
    else:
        dissimilarity = 1 - picks @ picks.T
        diversity = (dissimilarity.sum() - dissimilarity.diagonal().sum()) / (len(picks) * (len(picks) - 1))
    nearest = torch.cdist(features, picks).min(1).values
    return diversity + torch.exp(-nearest.mean())


def reinforce_loss(frame_logits, features, baseline, episodes=5, target_rate=0.5, rate_weight=0.01):
    """REINFORCE over Bernoulli frame selection, as in DR-DSN (Zhou et al. 2018), which Sec. 3.4 follows."""
    probabilities = frame_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    loss, rewards = 0.0, []
    for _ in range(episodes):
        selected = torch.bernoulli(probabilities.detach()).bool()
        if not selected.any():
            selected[probabilities.argmax()] = True
        reward = video_reward(selected, features).detach()
        log_probability = torch.where(selected, probabilities.log(), (1 - probabilities).log()).mean()
        loss = loss - log_probability * (reward - baseline)
        rewards.append(reward.item())
    return loss / episodes + rate_weight * (probabilities.mean() - target_rate) ** 2, sum(rewards) / episodes


def normalized(features):
    return F.normalize(features, dim=-1)
