from pathlib import Path
import base64
import csv
import html
import json
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
RUN_DIR = PROJECT_ROOT / "outputs" / "full_run_v3"
OUT_DIR = PROJECT_ROOT / "outputs" / "annotation"

SEED = 42
N_STORIES = 100
N_ANNOTATORS = 4
N_OVERLAP = 20          # stories labelled by two people, for inter-annotator agreement


def load_candidates():
    titles = {}
    for line in open(RUN_DIR / "clean_test.jsonl"):
        row = json.loads(line)
        titles[row["id"]] = row["title"]
    stories = []
    for r in csv.DictReader(open(RUN_DIR / "test_predictions.csv")):
        if r["source"] != "dailymail":
            continue
        folder = DATA_ROOT / "daily mail" / r["id"].split("/")[1]
        images = sorted(folder.glob("*.png"), key=lambda p: int(p.stem)) + sorted(folder.glob("*.jpg"))
        if len(images) >= 2:
            stories.append({"id": r["id"], "title": titles[r["id"]], "summary": r["finetuned"], "images": images})
    return sorted(stories, key=lambda s: s["id"])


def assign(stories, rng):
    """Split stories across annotators; the first N_OVERLAP also go to a second annotator."""
    rng.shuffle(stories)
    chosen = stories[:N_STORIES]
    shards = {a: [] for a in range(N_ANNOTATORS)}
    for i, s in enumerate(chosen):
        first = i % N_ANNOTATORS
        shards[first].append(s)
        if i < N_OVERLAP:
            shards[(first + 1) % N_ANNOTATORS].append(s)
    return chosen, shards


def story_html(s, rng):
    order = list(s["images"])
    rng.shuffle(order)    # hide the original position so annotators don't favour the lead image
    imgs = "".join(
        f'<label class="opt"><input type="radio" name="{s["id"]}" value="{p.name}">'
        f'<img alt="candidate image" src="data:image/{p.suffix[1:]};base64,{base64.b64encode(p.read_bytes()).decode()}"></label>'
        for p in order)
    return (f'<section class="story" data-id="{s["id"]}"><h2>{html.escape(s["title"])}</h2>'
            f'<p class="summary">{html.escape(s["summary"])}</p><div class="grid">{imgs}'
            f'<label class="opt none"><input type="radio" name="{s["id"]}" value="none">None of these fits</label></div></section>')


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cover Image Labelling</title>
<style>
:root{--bg:#fbfbfa;--fg:#151515;--muted:#5b5b57;--line:#d9d8d3;--pick:#2a78d6;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#efeee9;--muted:#b9b8b0;--line:#3a3a37;--pick:#3987e5;--card:#232322}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:1}
.story{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.story h2{font-size:18px;margin:0 0 8px}.summary{color:var(--muted);margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.opt{display:flex;flex-direction:column;gap:6px;border:2px solid var(--line);border-radius:8px;padding:6px;cursor:pointer}
.opt img{width:100%;height:auto;border-radius:4px}.opt.none{justify-content:center;align-items:center;min-height:80px}
.opt:has(input:checked){border-color:var(--pick)}button,input[type=text]{font:inherit;padding:6px 10px}
</style></head><body>
<header><strong>Cover image labelling</strong> · __WHO__ · <span id="count"></span><br>
For each story, pick the image that best illustrates the summary as a cover picture. If none fits, choose “None of these fits”.
<div style="margin-top:8px"><input id="name" type="text" placeholder="Your name"> <button id="dl">Download answers (CSV)</button></div></header>
<main>__STORIES__</main>
<script>
const KEY = "cover-labels-__SHARD__";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const radios = [...document.querySelectorAll("input[type=radio]")];
radios.forEach(r => { if (saved[r.name] === r.value) r.checked = true; });
const total = document.querySelectorAll(".story").length;
function update() {
  const done = new Set(radios.filter(r => r.checked).map(r => r.name)).size;
  document.getElementById("count").textContent = done + " / " + total + " done";
}
radios.forEach(r => r.addEventListener("change", () => {
  saved[r.name] = r.value;
  try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
  update();
}));
update();
document.getElementById("dl").addEventListener("click", () => {
  const who = document.getElementById("name").value.trim() || "__WHO__";
  const rows = [["annotator", "shard", "story_id", "choice"]];
  document.querySelectorAll(".story").forEach(s => {
    const pick = s.querySelector("input:checked");
    rows.push([who, "__SHARD__", s.dataset.id, pick ? pick.value : ""]);
  });
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  a.download = "cover_labels___SHARD__.csv";
  a.click();
});
</script></body></html>"""


def main():
    rng = random.Random(SEED)
    candidates = load_candidates()
    chosen, shards = assign(candidates, rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a, stories in shards.items():
        name = f"annotator_{a + 1}"
        page = (PAGE.replace("__STORIES__", "".join(story_html(s, rng) for s in stories))
                    .replace("__WHO__", name).replace("__SHARD__", name))
        (OUT_DIR / f"label_{name}.html").write_text(page, encoding="utf-8")
        print(f"{name}: {len(stories)} stories")
    (OUT_DIR / "annotation_sample.json").write_text(json.dumps({
        "seed": SEED, "candidates": len(candidates), "stories": [s["id"] for s in chosen],
        "overlap": [s["id"] for s in chosen[:N_OVERLAP]],
        "shards": {f"annotator_{a + 1}": [s["id"] for s in v] for a, v in shards.items()}}, indent=1))
    print(f"{len(chosen)} of {len(candidates)} eligible test stories; files in {OUT_DIR}")


if __name__ == "__main__":
    main()
