
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"

MODEL_NAME = "facebook/bart-large-cnn"

def find_sample():
    """Find an article that has a reference summary."""
    for article_path in DATA_ROOT.rglob("article.txt"):
        summary_path = article_path.with_name("highlight.txt")

        if summary_path.exists():
            article = article_path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
            reference = summary_path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()

            if article and reference:
                return article_path, article, reference

    raise FileNotFoundError(
        "No matching article.txt and highlight.txt files found."
    )

def main():
    print("Loading an MM-AVS sample...")
    article_path, article, reference = find_sample()

    print(f"Article: {article_path}")
    print(f"Article length: {len(article)} characters")

    device = (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print("\nDownloading/loading pretrained BART...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # BART has a limited input length; truncate the article.
    inputs = tokenizer(
        article,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    print("\nGenerating summary...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

    generated = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )

    print("\n" + "=" * 60)
    print("GENERATED SUMMARY")
    print("=" * 60)
    print(generated)

    print("\n" + "=" * 60)
    print("REFERENCE SUMMARY")
    print("=" * 60)
    print(reference)

if __name__ == "__main__":
    main()
