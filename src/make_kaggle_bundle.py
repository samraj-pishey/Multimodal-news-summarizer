from pathlib import Path
import csv
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLITS_CSV = OUTPUT_DIR / "splits.csv"
BUNDLE_DIR = OUTPUT_DIR / "kaggle_bundle"


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def main():
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        split: open(BUNDLE_DIR / f"{split}.jsonl", "w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    counts = {split: 0 for split in files}

    with open(SPLITS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            folder = PROJECT_ROOT / row["folder"]
            # CNN folders spell it "artitle.txt".
            article = read_text(folder / "article.txt") or read_text(folder / "artitle.txt")
            summary = read_text(folder / "highlight.txt")
            if not article or not summary:
                raise ValueError(f"{row['id']} is in splits.csv but has no article/summary")

            record = {
                "id": row["id"],
                "source": row["source"],
                "group_id": int(row["group_id"]),
                "is_group_rep": row["is_group_rep"] == "True",
                "title": read_text(folder / "title.txt"),
                "article": article,
                "summary": summary,
            }
            files[row["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[row["split"]] += 1

    for fh in files.values():
        fh.close()

    # Keep the split settings next to the data so the notebook can log them.
    shutil.copy(OUTPUT_DIR / "splits_summary.json", BUNDLE_DIR / "splits_summary.json")

    for split, n in counts.items():
        size_mb = (BUNDLE_DIR / f"{split}.jsonl").stat().st_size / 1e6
        print(f"{split:<6} {n:>5} records  {size_mb:5.1f} MB")
    print(f"\nBundle written to {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
