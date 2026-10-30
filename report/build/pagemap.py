import sys, json, re, fitz
d = fitz.open(sys.argv[1]); out = {}
# Chapter pages are numbered from 1 starting at the page that holds "Chapter 1: Introduction" heading.
start = next(i for i, p in enumerate(d) if p.get_text().lstrip().startswith("Chapter 1: Introduction"))
for i, p in enumerate(d):
    if i < start: continue
    for m in re.finditer(r"^(Figure|Table) ([0-9A]+\.[0-9]+):", p.get_text(), re.M):
        out.setdefault(f"{m.group(1)} {m.group(2)}", i - start + 1)
json.dump(out, open(sys.argv[2], "w"), indent=0); print(out)
