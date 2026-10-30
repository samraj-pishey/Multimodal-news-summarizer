from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import random
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCES = {"cnn": "cnn_data", "dailymail": "daily mail"}
OFFICIAL_TEST_IDS = DATA_ROOT / "test_id_dm.txt"

SEED = 42
# Daily Mail test comes from the official list; CNN has no official split.
DM_RATIOS = {"train": 0.9, "val": 0.1}
CNN_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Articles sharing more than this fraction of 8-word shingles are treated
# as the same story. Kept low on purpose: over-grouping only costs a little
# randomness, under-grouping leaks test stories into training.
SHINGLE_SIZE = 8
NEAR_DUP_JACCARD = 0.3
# Shingles appearing in more documents than this are boilerplate, not evidence.
MAX_SHINGLE_DF = 50


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def load_examples():
    examples = []

    for source, folder_name in SOURCES.items():
        for folder in sorted((DATA_ROOT / folder_name).iterdir()):
            if not folder.is_dir():
                continue

            # CNN folders spell it "artitle.txt".
            article = read_text(folder / "article.txt") or read_text(folder / "artitle.txt")
            summary = read_text(folder / "highlight.txt")
            if not article or not summary:
                continue

            images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            ts_segments = list((folder / "video").glob("*.ts"))

            examples.append({
                "id": f"{source}/{folder.name}",
                "source": source,
                "folder": str(folder.relative_to(PROJECT_ROOT)),
                "article": article,
                "summary": summary,
                "title": read_text(folder / "title.txt"),
                "n_article_words": len(article.split()),
                "n_summary_words": len(summary.split()),
                "n_images": len(images),
                "has_captions": bool(read_text(folder / "img_caption.txt")),
                "has_transcript": bool(read_text(folder / "transcript.txt")),
                "has_audio": (folder / "audio.mp3").exists(),
                "has_video": (folder / "video.mp4").exists() or bool(ts_segments),
            })

    return examples


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def group_duplicates(examples):
    """Return a story-group index per example, merging exact and near duplicates."""
    uf = UnionFind(len(examples))

    # Exact matches on article text or on the reference summary.
    for field in ("article", "summary"):
        first_seen = {}
        for i, ex in enumerate(examples):
            key = normalize(ex[field])
            if key in first_seen:
                uf.union(i, first_seen[key])
            else:
                first_seen[key] = i

    # Near duplicates via shingle overlap, counted through an inverted index.
    shingles = []
    postings = defaultdict(list)
    for i, ex in enumerate(examples):
        words = normalize(ex["article"]).split()
        doc = {
            " ".join(words[k:k + SHINGLE_SIZE])
            for k in range(max(1, len(words) - SHINGLE_SIZE + 1))
        }
        shingles.append(doc)
        for s in doc:
            postings[s].append(i)

    shared = Counter()
    for docs in postings.values():
        if 1 < len(docs) <= MAX_SHINGLE_DF:
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    shared[docs[a], docs[b]] += 1

    near_pairs = 0
    for (i, j), n in shared.items():
        jaccard = n / (len(shingles[i]) + len(shingles[j]) - n)
        if jaccard > NEAR_DUP_JACCARD:
            uf.union(i, j)
            near_pairs += 1

    roots = {}
    group_ids = [roots.setdefault(uf.find(i), len(roots)) for i in range(len(examples))]
    return group_ids, near_pairs


def assign_splits(examples, group_ids, official_test):
    groups = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)

    split_of_group = {}
    pools = {"dailymail": [], "cnn": []}
    for g, members in groups.items():
        # Any official test article pulls its whole story group into test.
        if any(examples[i]["id"] in official_test for i in members):
            split_of_group[g] = "test"
        elif any(examples[i]["source"] == "dailymail" for i in members):
            pools["dailymail"].append(g)
        else:
            pools["cnn"].append(g)

    rng = random.Random(SEED)
    for pool, ratios in ((pools["dailymail"], DM_RATIOS), (pools["cnn"], CNN_RATIOS)):
        pool.sort()
        rng.shuffle(pool)
        total = sum(len(groups[g]) for g in pool)
        cutoffs, running = [], 0.0
        for name, ratio in ratios.items():
            running += ratio
            cutoffs.append((running * total, name))

        filled = 0
        for g in pool:
            split_of_group[g] = next(name for limit, name in cutoffs if filled < limit)
            filled += len(groups[g])

    # One representative per group, so evaluation counts each story once.
    representatives = {g: min(members, key=lambda i: examples[i]["id"]) for g, members in groups.items()}
    return [split_of_group[g] for g in group_ids], {representatives[g] for g in groups}


def main():
    examples = load_examples()
    official_test = {
        f"dailymail/{x.strip()}"
        for x in OFFICIAL_TEST_IDS.read_text().split(",")
        if x.strip()
    }

    group_ids, near_pairs = group_duplicates(examples)
    splits, representatives = assign_splits(examples, group_ids, official_test)

    OUTPUT_DIR.mkdir(exist_ok=True)
    columns = [
        "id", "source", "folder", "split", "group_id", "is_group_rep",
        "in_official_test", "n_article_words", "n_summary_words", "n_images",
        "has_captions", "has_transcript", "has_audio", "has_video",
    ]
    with open(OUTPUT_DIR / "splits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for i, ex in enumerate(examples):
            writer.writerow({
                **ex,
                "split": splits[i],
                "group_id": group_ids[i],
                "is_group_rep": i in representatives,
                "in_official_test": ex["id"] in official_test,
            })

    counts = defaultdict(Counter)
    for i, ex in enumerate(examples):
        counts[splits[i]][ex["source"]] += 1
        if i in representatives:
            counts[splits[i]]["unique_stories"] += 1

    loaded_ids = {ex["id"] for ex in examples}
    summary = {
        "seed": SEED,
        "dm_ratios": DM_RATIOS,
        "cnn_ratios": CNN_RATIOS,
        "near_dup_jaccard": NEAR_DUP_JACCARD,
        "shingle_size": SHINGLE_SIZE,
        "total_examples": len(examples),
        "unique_stories": len(representatives),
        "near_duplicate_pairs": near_pairs,
        "official_test_ids": len(official_test),
        "official_test_ids_missing_locally": sorted(official_test - loaded_ids),
        "counts": {split: dict(c) for split, c in sorted(counts.items())},
    }
    (OUTPUT_DIR / "splits_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Loaded {len(examples)} article-summary pairs "
          f"({len(representatives)} unique stories, {near_pairs} near-duplicate pairs).")
    print(f"Official test IDs missing locally: {summary['official_test_ids_missing_locally']}")
    print(f"\n{'split':<6} {'dailymail':>10} {'cnn':>6} {'total':>6} {'unique':>7}")
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split:<6} {c['dailymail']:>10} {c['cnn']:>6} "
              f"{c['dailymail'] + c['cnn']:>6} {c['unique_stories']:>7}")
    print(f"\nWrote {OUTPUT_DIR / 'splits.csv'}")


if __name__ == "__main__":
    main()
