"""Frozen RoBERTa-base sentence vectors for every article sentence and Whisper sentence (A2Summ uses RoBERTa features).

Each sentence is encoded on its own and mean-pooled over its tokens, leaving out <s> and </s>; A2Summ's files don't say
which pooling they used. Stored as float16, one [sentences x 768] tensor per story, in the same order as the cache.
Resumable: finished stories are kept in the output file and skipped.

Usage: python src/m2sm_sentence_vectors.py [--threads 6] [--limit N]
"""
import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer

from m2sm_data import CACHE_PATH, SPEECH_CACHE_PATH, VECTOR_PATHS

MODEL_NAME = "FacebookAI/roberta-base"
MAX_TOKENS = 128
BATCH_SIZE = 64
SAVE_EVERY = 200


@torch.no_grad()
def encode(sentences, tokenizer, model):
    """Mean of the last hidden states over each sentence's own tokens, in input order."""
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    vectors = [None] * len(sentences)
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        tokens = tokenizer([sentences[i] for i in batch], padding=True, truncation=True, max_length=MAX_TOKENS,
                           return_tensors="pt", return_special_tokens_mask=True)
        special = tokens.pop("special_tokens_mask").bool()
        states = model(**tokens).last_hidden_state
        keep = (tokens["attention_mask"].bool() & ~special).unsqueeze(-1)
        pooled = (states * keep).sum(1) / keep.sum(1).clamp(min=1)
        for i, vector in zip(batch, pooled):
            vectors[i] = vector
    return torch.stack(vectors).half()


def encode_all(name, stories, tokenizer, model, limit):
    path = VECTOR_PATHS[name]
    done = torch.load(path) if path.exists() else {}
    todo = [s for s in stories if s["id"] not in done][:limit]
    began, sentences = time.time(), 0
    for n, story in enumerate(todo, 1):
        done[story["id"]] = encode(story["sentences"], tokenizer, model)
        sentences += len(story["sentences"])
        if n % SAVE_EVERY == 0 or n == len(todo):
            torch.save(done, path)
            rate = sentences / (time.time() - began)
            print(f"  {name}: {len(done)} of {len(stories)} stories, {rate:.0f} sentences/s", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, help="pilot only: encode at most this many new stories per set")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    articles = torch.load(CACHE_PATH, weights_only=False)["stories"]
    speech = torch.load(SPEECH_CACHE_PATH, weights_only=False)
    print(f"{len(articles)} articles ({sum(len(s['sentences']) for s in articles):,} sentences), "
          f"{len(speech)} speech stories ({sum(len(s['sentences']) for s in speech):,} sentences)", flush=True)
    encode_all("article", articles, tokenizer, model, args.limit)
    encode_all("speech", speech, tokenizer, model, args.limit)


if __name__ == "__main__":
    main()
