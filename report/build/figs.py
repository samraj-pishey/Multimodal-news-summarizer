import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[2]
res = json.load(open(sys.argv[1]))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# Fig 5.1: test ROUGE with 95% CI
metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]
fig, ax = plt.subplots(figsize=(6.3, 3.4), dpi=200)
w = 0.36
for k, (name, col, lab) in enumerate((("pretrained", ORANGE, "Pretrained BART-large-CNN"),
                                      ("finetuned", BLUE, "Fine-tuned on MM-AVS"))):
    m = [res["test"][name][x]["mean"] for x in metrics]
    lo = [m[i] - res["test"][name][x]["ci95"][0] for i, x in enumerate(metrics)]
    hi = [res["test"][name][x]["ci95"][1] - m[i] for i, x in enumerate(metrics)]
    xs = [i + (k - 0.5) * (w + 0.02) for i in range(4)]
    ax.bar(xs, m, w, color=col, label=lab, yerr=[lo, hi], capsize=3, error_kw={"ecolor": INK, "lw": 0.8})
    for x, v in zip(xs, m):
        ax.text(x, v + 0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(range(4), labels)
ax.set_ylabel("F1 score (test, 418 stories)")
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, loc="upper right", fontsize=9)
clean(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/rouge_test.png"); plt.close(fig)

# Fig 5.2: validation curves (two panels, own axes)
h = res["history"]
ep = [r["epoch"] for r in h]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.8), dpi=200)
a1.plot(ep, [r["val_rouge_mean"] for r in h], color=BLUE, lw=2, marker="o", ms=5)
best = res["best_epoch"]
a1.annotate("selected", (best, h[best - 1]["val_rouge_mean"]), textcoords="offset points", xytext=(7, -3), fontsize=9, color=INK)
a1.set_title("Validation mean ROUGE-1/2/L", fontsize=10.5, color=INK)
a1.set_xlabel("Epoch"); a1.set_xticks(ep)
a2.plot(ep, [r["train_loss"] for r in h], color=BLUE, lw=2, marker="o", ms=5, label="Training loss")
a2.plot(ep, [r["val_loss"] for r in h], color=ORANGE, lw=2, marker="s", ms=5, label="Validation loss")
a2.set_title("Cross-entropy loss", fontsize=10.5, color=INK)
a2.set_xlabel("Epoch"); a2.set_xticks(ep)
a2.legend(frameon=False, fontsize=8.5)
for a in (a1, a2): clean(a)
fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png"); plt.close(fig)

# Fig 3.1: pipeline diagram
fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, text, done=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                 fc="#eaf2fc" if done else "#ffffff", ec=BLUE if done else MUTED, lw=1.2,
                 ls="-" if done else (0, (4, 3))))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=INK, wrap=True)
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
box(2, 82, 28, 13, "MM-AVS raw data\n(article, highlights,\nimages, audio, video)")
box(36, 82, 28, 13, "Duplicate grouping\n(exact + 8-word shingle\nJaccard > 0.3)")
box(70, 82, 28, 13, "Leakage-free splits\ntrain / val / test\n(official DM test IDs)")
box(2, 58, 28, 13, "Text cleaning\n(re-split glued highlights,\nre-space articles)")
box(36, 58, 28, 13, "Pretrained BART-large-CNN\n(zero-shot baseline)")
box(70, 58, 28, 13, "Fine-tuned BART\n(best epoch on\nvalidation)")
box(36, 34, 28, 13, "Evaluation: ROUGE F1,\nbootstrap 95% CI,\npaired bootstrap")
box(2, 10, 22, 13, "CLIP image\nrelevance", done=False)
box(27, 10, 22, 13, "Whisper audio\ntranscripts", done=False)
box(52, 10, 22, 13, "Video key\nframes", done=False)
box(77, 10, 21, 13, "Multimodal\ncomparison", done=False)
arrow(30, 88.5, 36, 88.5); arrow(64, 88.5, 70, 88.5); arrow(84, 82, 84, 77)
ax.annotate("", (16, 71), (84, 77), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, connectionstyle="arc3,rad=0.0"))
arrow(30, 64.5, 36, 64.5); arrow(64, 64.5, 70, 64.5)
arrow(50, 58, 50, 47); arrow(84, 58, 60, 47)
arrow(50, 34, 50, 24.5)
ax.text(50, 29.5, "", fontsize=8)
ax.plot([], [], color=BLUE, lw=6, alpha=0.3, label="Completed")
ax.plot([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Planned")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/pipeline.png"); plt.close(fig)
print("figures written")
