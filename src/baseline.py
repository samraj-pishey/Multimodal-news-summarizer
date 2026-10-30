
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
MODEL_NAME = "facebook/bart-large-cnn"

# Start with 10 articles to verify evaluation works.
# We can scale up after confirming memory and runtime.
MAX_ARTICLES = 10
SEED = 42


def load_articles():
    samples = []

    for article_path in DATA_ROOT.rglob("article.txt"):
        reference_path = article_path.with_name("highlight.txt")

        if not reference_path.exists():
            continue

        article = article_path.read_text(
            encoding="utf-8", errors="ignore"
        ).strip()
        reference = reference_path.read_text(
            encoding="utf-8", errors="ignore"
        ).strip()

        if article and reference:
            samples.append({
                "id": str(article_path.relative_to(DATA_ROOT)),
                "article": article,
                "reference": reference,
            })

    # Stable ordering before sampling makes results reproducible.
    samples.sort(key=lambda x: x["id"])
    random.seed(SEED)
    random.shuffle(samples)

    return samples


def main():
    samples = load_articles()

    if not samples:
        raise RuntimeError(
            f"No article.txt/highlight.txt pairs found under {DATA_ROOT}"
        )

    selected = samples[:min(MAX_ARTICLES, len(samples))]
    print(f"Found {len(samples)} usable article-summary pairs.")
    print(f"Evaluating {len(selected)} articles.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for i, sample in enumerate(selected, start=1):
        inputs = tokenizer(
            sample["article"],
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=4,
                no_repeat_ngram_size=3,
            )

        prediction = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        scores = scorer.score(sample["reference"], prediction)

        for metric in totals:
            totals[metric] += scores[metric].fmeasure

        print(f"\n[{i}/{len(selected)}] {sample['id']}")
        print(f"Generated: {prediction[:300]}")
        print(
            "ROUGE-1: {:.4f} | ROUGE-2: {:.4f} | ROUGE-L: {:.4f}".format(
                scores["rouge1"].fmeasure,
                scores["rouge2"].fmeasure,
                scores["rougeL"].fmeasure,
            )
        )

    print("\n========== MEAN ROUGE SCORES ==========")
    for metric, total in totals.items():
        print(f"{metric.upper()}: {total / len(selected):.4f}")


if __name__ == "__main__":
    main()
