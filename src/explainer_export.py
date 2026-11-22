"""Data for the "How it works" screens in webapp/: one JSON file plus video thumbnails, all computed offline.

Uses the trained hybrid model (outputs/hybrid/runs/hybrid) on the app's five sample stories, which were drawn at
random from the test split before any of this was built. Also adds dataset-level facts (where key sentences sit,
GloVe neighbours) and the test results with paired bootstrap intervals.

Usage: python src/explainer_export.py [--out webapp/explainer]
"""
import argparse
import json
import re
import shutil

import numpy as np
import torch
from rouge_score import tokenize as rouge_tokenize
from transformers import AutoTokenizer

from hybrid_model import ARTICLE, SPEECH, VIDEO
from hybrid_train import RUN_DIR, OUT_DIR as HYBRID_OUT, build_samples, make_model, parse_args as hybrid_args
from m2sm_baselines import vsumm_choice
from m2sm_data import CACHE_PATH, PROJECT_ROOT, score_summary, scorer, tokenize
from m2sm_export_demo import save_frames
from m2sm_sentence_vectors import MODEL_NAME

SAMPLES = PROJECT_ROOT / "webapp" / "samples" / "samples.json"
M2SM_OUT = PROJECT_ROOT / "outputs" / "m2sm" / "runs"
SKIP_STORIES = {"dailymail/4202"}      # the user asked to drop this one: its clown-face frames look like ghosts
STAGE_SENTENCES = 8                   # sentences shown in the animation
BARCODE_DIMS = 32                     # leading RoBERTa dimensions drawn as a sentence's "barcode"
TIMELINE_FRAMES = 12
SIMILARITY_SENTENCES = 12
POSITIONS = 30
WORDS = ["police", "president", "football", "music", "election", "hospital", "film", "court", "money", "school",
         "children", "fire", "war", "phone", "queen", "doctor", "car", "rain", "happy", "london"]
ABLATIONS = [("hybrid", "Full hybrid"), ("hybrid_no_speech", "Speech switched off"),
             ("hybrid_no_video", "Video switched off"), ("hybrid_no_align", "Time alignment switched off"),
             ("hybrid_no_contrastive", "Contrastive losses switched off"),
             ("hybrid_no_sentence_lstm", "Sentence BiLSTM switched off"),
             ("hybrid_no_lateplus", "Late+ fusion switched off")]


def r(values, digits=3):
    return [round(float(v), digits) for v in values]


def glove_neighbours(cache, words, count=8):
    vocabulary, vectors = cache["vocabulary"], cache["embeddings"].numpy()
    names = {i: w for w, i in vocabulary.items()}
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    out = {}
    for word in words:
        if word not in vocabulary:
            continue
        similarity = unit @ unit[vocabulary[word]]
        ranked = [i for i in np.argsort(-similarity) if i in names and names[i] != word and names[i].isalpha()]
        out[word] = [{"word": names[i], "similarity": round(float(similarity[i]), 2)} for i in ranked[:count]]
    return out


def key_sentence_positions(stories):
    """For each position in an article, how often the authors' labels mark it as a key sentence (train split)."""
    hits, totals = np.zeros(POSITIONS), np.zeros(POSITIONS)
    for story in stories:
        for i, label in enumerate(story["labels"][:POSITIONS]):
            totals[i] += 1
            hits[i] += label
    return r(100 * hits / totals.clip(min=1), 1)


def roberta_pieces(tokenizer, text):
    """RoBERTa's sub-word pieces as spans of the original text, marking pieces that start a new word."""
    spans = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    pieces, last = [], None
    for start, end in spans:
        if end > start and (start, end) != last:      # one character can span several byte-level tokens
            pieces.append({"text": text[start:end].strip(), "new_word": start == 0 or text[start - 1] == " "})
        last = (start, end)
    return [p for p in pieces if p["text"]]


def rouge_words(summary, reference):
    """Words of each text with a flag for whether ROUGE-1 counts them as shared (lowercased and stemmed)."""
    stem = lambda word: rouge_tokenize.tokenize(word, scorer._tokenizer._stemmer)
    reference_stems = {s for w in reference.split() for s in stem(w)}
    summary_stems = {s for w in " ".join(summary).split() for s in stem(w)}
    mark = lambda text, other: [{"w": w, "hit": any(s in other for s in stem(w))} for w in text.split()]
    return {"summary": [mark(s, reference_stems) for s in summary], "reference": mark(reference, summary_stems)}


@torch.no_grad()
def run_with_attention(model, sample):
    """Hybrid outputs plus the last layer's attention, averaged over heads."""
    captured = []
    attention = model.blocks[-1].attention
    original = attention.forward

    def forward(*args, **kwargs):
        kwargs.update(need_weights=True, average_attn_weights=True)
        out = original(*args, **kwargs)
        captured.append(out[1])
        return out

    attention.forward = forward
    try:
        out = model([sample])[0]
        _, modality, _, _ = model.tokens(sample)
    finally:
        attention.forward = original
    weights = captured[-1][0][:len(modality), :len(modality)]
    where = {m: (modality == m).nonzero().squeeze(1)[1:] for m in (ARTICLE, VIDEO, SPEECH)}
    return out, weights, where, model.blocked_pairs(*model.tokens(sample)[1:3])


def story_data(sample, text_model, frame_model, roberta, size, frames_dir):
    story = sample["story"]
    out, weights, where, blocked = run_with_attention(text_model, sample)
    frame_scores = frame_model([sample])[0]["frame_logits"].sigmoid()
    scores = out["sentence_logits"].sigmoid().tolist()
    chosen = sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:size])
    lead = list(range(min(3, len(scores))))

    times = sample["frame_times"].numpy()
    n_frames = len(times)
    timeline = sorted(set(np.linspace(0, n_frames - 1, min(TIMELINE_FRAMES, n_frames)).round().astype(int).tolist()))
    top_frames = frame_scores.topk(min(4, n_frames)).indices.tolist()
    unit = torch.nn.functional.normalize(sample["frames"], dim=1)
    vsumm = vsumm_choice(unit, min(4, n_frames), 42)
    wanted = sorted(set(timeline + top_frames + vsumm))
    names = save_frames(story["id"], [float(times[i]) for i in wanted], frames_dir)
    thumbnails = {i: names.get(float(times[i])) for i in wanted}

    vectors = torch.nn.functional.normalize(sample["article"][:SIMILARITY_SENTENCES], dim=1)
    first = story["sentences"][0]
    speech = None
    if "speech" in sample and len(where[SPEECH]):
        speech_rows = where[SPEECH]
        speech = {"sentences": [s for s in story.get("speech_sentences", [])],
                  "times": sample["speech_times"].tolist(),
                  "frame_attention": [r(weights[i, where[VIDEO]].tolist()) for i in speech_rows],
                  "allowed": [(~blocked[i, where[VIDEO] - 0]).tolist() for i in speech_rows]}

    return {
        "id": story["id"], "title": sample["title"], "sentences": story["sentences"],
        "reference": story["reference"], "human": [h for h in story["reference"].split("\n") if h.strip()],
        "tokens": {"sentence": first, "simple": tokenize(first)[:40], "roberta": roberta_pieces(roberta, first)[:48]},
        "similarity": r((vectors @ vectors.T).flatten().tolist(), 2), "similarity_size": len(vectors),
        "barcodes": [r((v / v.abs().max()).tolist(), 2) for v in sample["article"][:STAGE_SENTENCES, :BARCODE_DIMS]],
        "scores": r(scores), "chosen": chosen, "lead": lead, "size": size,
        "frame_times": r(times, 1), "frame_scores": r(frame_scores.tolist()), "top_frames": sorted(top_frames),
        "vsumm_frames": sorted(vsumm), "thumbnails": {str(i): n for i, n in thumbnails.items() if n},
        "timeline": [i for i in timeline if thumbnails.get(i)],
        "sentence_frame_attention": [r(weights[i, where[VIDEO]].tolist()) for i in where[ARTICLE]],
        "sentence_speech_attention": ([r(weights[i, where[SPEECH]].tolist()) for i in where[ARTICLE]]
                                      if speech else None),
        "speech": speech,
        "rouge": {"model": {**{k: round(100 * v, 1) for k, v in score_summary(
                                story["reference"], [story["sentences"][i] for i in chosen]).items()},
                            **rouge_words([story["sentences"][i] for i in chosen], story["reference"])},
                  "lead3": {**{k: round(100 * v, 1) for k, v in score_summary(
                                story["reference"], [story["sentences"][i] for i in lead]).items()},
                            **rouge_words([story["sentences"][i] for i in lead], story["reference"])}},
    }


def paired(a, b, seed=42):
    d = a - b
    picks = np.random.default_rng(seed).integers(0, len(d), (10000, len(d)))
    low, high = np.percentile(d[picks].mean(1), [2.5, 97.5], axis=0)
    return [{"diff": round(float(x), 2), "low": round(float(l), 2), "high": round(float(h), 2)}
            for x, l, h in zip(d.mean(0), low, high)]


def results(stories):
    """Per-story ROUGE on the Daily Mail test split for the headline rows, with bootstrap intervals."""
    test = sorted([s for s in stories if s["split_paper"] == "test" and s["source"] == "dailymail"],
                  key=lambda s: s["id"])
    metrics = ("r1", "r2", "rl")
    rows_of = lambda pick: np.array([[100 * score_summary(s["reference"], pick(s))[m] for m in metrics] for s in test])
    top = lambda values, k: sorted(sorted(range(len(values)), key=lambda i: -values[i])[:k])

    def m2sm(run):
        size = json.loads((M2SM_OUT / run / "results.json").read_text())["summary_size"]
        scores = json.loads((M2SM_OUT / run / "test_scores.json").read_text())
        return rows_of(lambda s: [s["sentences"][i] for i in top(scores[s["id"]]["scores"], size)])

    def hybrid(run):
        size = json.loads((HYBRID_OUT / run / "results.json").read_text())["state"]["size"]
        scores = json.loads((HYBRID_OUT / run / "test_scores.json").read_text())
        return rows_of(lambda s: [s["sentences"][i] for i in top(scores[s["id"]]["sentences"], size)])

    lead3 = rows_of(lambda s: s["sentences"][:3])
    main = [("Lead3 (first 3 sentences)", lead3), ("Text only, GloVe (our M2SM copy)", m2sm("text_only_lr0.01")),
            ("Full M2SM (text + video + transcript)", m2sm("m2sm_full_lr0.01")), ("Full hybrid", hybrid("hybrid"))]
    table = [{"name": name, **dict(zip(metrics, r(rows.mean(0), 2))), "vs_lead3": paired(rows, lead3)}
             for name, rows in main]
    full = hybrid("hybrid")
    ablations = []
    for run, name in ABLATIONS:
        rows = full if run == "hybrid" else hybrid(run)
        result = json.loads((HYBRID_OUT / run / "results.json").read_text())
        ablations.append({"name": name, **dict(zip(metrics, r(rows.mean(0), 2))),
                          "cos": round(result.get("test_cos", {}).get("cos") or 0, 2) or None,
                          "change": None if run == "hybrid" else paired(rows, full)})
    baselines = json.loads((PROJECT_ROOT / "outputs" / "m2sm" / "baselines.json").read_text())
    cos = [{"name": n, "cos": round(v["cos"], 2)} for n, v in baselines["video"].items()]
    cos += [{"name": "Full hybrid", "cos": ablations[0]["cos"]}]
    return {"n": len(test), "main": table, "ablations": ablations, "cos": sorted(cos, key=lambda c: -c["cos"]),
            "oracle": dict(zip(metrics, r([baselines["text_dailymail"]["oracle"][m] for m in metrics], 2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "webapp" / "explainer"))
    out_dir = __import__("pathlib").Path(parser.parse_args().out)
    torch.set_num_threads(4)

    import sys
    sys.argv = [sys.argv[0], "--name", "hybrid"]
    args = hybrid_args()
    samples = {s["id"]: s for s in build_samples(args)}
    cache = torch.load(CACHE_PATH, weights_only=False)
    speech_cache = {s["id"]: s for s in torch.load(PROJECT_ROOT / "data" / "m2sm_speech.pt", weights_only=False)}

    text_model, frame_model = make_model(args), make_model(args)
    text_model.load_state_dict(torch.load(RUN_DIR / "hybrid" / "best_text.pt"))
    frame_model.load_state_dict(torch.load(RUN_DIR / "hybrid" / "best_frames.pt"))
    text_model.eval(), frame_model.eval()
    size = json.loads((HYBRID_OUT / "hybrid" / "results.json").read_text())["state"]["size"]
    roberta = AutoTokenizer.from_pretrained(MODEL_NAME)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    stories = []
    for item in json.loads(SAMPLES.read_text())["samples"]:
        if item["id"] in SKIP_STORIES:
            continue
        sample = samples[item["id"]]
        sample["title"] = item["title"]
        if item["id"] in speech_cache:
            sample["story"] = {**sample["story"], "speech_sentences": speech_cache[item["id"]]["sentences"]}
        with torch.no_grad():
            stories.append(story_data(sample, text_model, frame_model, roberta, size, frames_dir))

    train = [s for s in cache["stories"] if s["split_paper"] == "train" and s["source"] == "dailymail"]
    payload = {"model": "hybrid (M2SM + A2Summ)", "stories": stories,
               "key_positions": key_sentence_positions(train), "glove": glove_neighbours(cache, WORDS),
               "results": results(cache["stories"])}
    (out_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False))
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 2 ** 20
    print(f"Wrote {len(stories)} stories to {out_dir} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
