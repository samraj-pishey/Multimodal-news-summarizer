import json
import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, PageBreak)

OUT, RES, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
res = json.load(open(RES))
pre, ft = res["test"]["pretrained"], res["test"]["finetuned"]

INK, MUTED, ACCENT, RULE, SHADE = (colors.HexColor(c) for c in ("#111111", "#555555", "#2a78d6", "#d0d0d0", "#f1f4f8"))
base = dict(fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=INK, alignment=TA_LEFT)
S = {
    "title": ParagraphStyle("title", **{**base, "fontName": "Helvetica-Bold", "fontSize": 16, "leading": 20}),
    "sub": ParagraphStyle("sub", **{**base, "textColor": MUTED, "fontSize": 9}),
    "h": ParagraphStyle("h", **{**base, "fontName": "Helvetica-Bold", "fontSize": 11, "leading": 14,
                               "textColor": ACCENT, "spaceBefore": 8, "spaceAfter": 3}),
    "p": ParagraphStyle("p", **{**base, "spaceAfter": 4}),
    "b": ParagraphStyle("b", **{**base, "leftIndent": 11, "bulletIndent": 2, "spaceAfter": 2}),
    "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.4, "leading": 10.4}),
    "cellb": ParagraphStyle("cellb", **{**base, "fontSize": 8.4, "leading": 10.4, "fontName": "Helvetica-Bold"}),
    "cap": ParagraphStyle("cap", **{**base, "fontSize": 8, "leading": 10, "textColor": MUTED}),
}
P = lambda t, s="p": Paragraph(t, S[s])
B = lambda t: Paragraph(t, S["b"], bulletText="•")
W = A4[0] - 3.6 * cm


def table(rows, widths, header=True, shade_first_col=False):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SHADE))
    t.setStyle(TableStyle(style))
    return t


f3 = lambda x: f"{x:.3f}"
pts = lambda m: f"+{(ft[m]['mean'] - pre[m]['mean']) * 100:.1f}"

story = [
    P("Multimodal News Summarizer — where we are", "title"),
    P("Team progress summary · 29 October 2026 · text baseline complete; image, audio and video stages next", "sub"),
    Spacer(1, 6),
    table([["In one paragraph"],
           [f"We have the full MM-AVS dataset, leakage-free splits, and a final text-only baseline. BART-large-CNN "
            f"fine-tuned on MM-AVS beats the pretrained model on all four ROUGE scores on 418 unseen test stories "
            f"(ROUGE-1 {f3(pre['rouge1']['mean'])} → <b>{f3(ft['rouge1']['mean'])}</b>, ROUGE-Lsum "
            f"{f3(pre['rougeLsum']['mean'])} → <b>{f3(ft['rougeLsum']['mean'])}</b>). Getting there meant fixing two "
            f"dataset problems that would otherwise have given misleading numbers. Every multimodal experiment from "
            f"now on is compared against this baseline, on the same test stories, with the same metrics."]],
          [1.0]),

    P("1. Status of the pipeline", "h"),
    table([["Stage", "Status", "Notes"],
           ["Get full MM-AVS dataset", "Done", "Daily Mail 4000–5999 + CNN: 2,224 article–summary pairs, ~27 GB with media"],
           ["Leakage-free splits", "Done", "src/build_splits.py → outputs/splits.csv"],
           ["Pretrained BART baseline", "Done", "facebook/bart-large-cnn, zero-shot"],
           ["Fine-tune BART on MM-AVS", "Done (run 3 = final)", "Kaggle T4, ~36 min training, 8.2 GB peak GPU memory"],
           ["CLIP image relevance", "Next", "score images + captions against article/summary"],
           ["Whisper audio, video key frames", "Planned", "same splits, same metrics, one modality at a time"],
           ["Multimodal comparison + report", "Planned", "report draft exists (report/); first submission 2 Nov"]],
          [0.30, 0.20, 0.50]),

    P("2. What we found in the data (and fixed)", "h"),
    B("<b>Duplicate stories.</b> 2,224 pairs but only <b>1,935 distinct stories</b>; one story appears 12 times. "
      "We grouped copies (exact match + 8-word-shingle Jaccard &gt; 0.3) and kept every group inside one split, "
      "so no test story is seen in training. Test = official MM-AVS Daily Mail test IDs + 20% of CNN: "
      "<b>418 unique test stories</b> (369 DM, 49 CNN); train 1,358, validation 159."),
    B("<b>Fused reference summaries.</b> Daily Mail highlights are stored with no separator "
      "(<i>“…last FridayRaiders seen on CCTV…”</i>). Our first run learned to write like that (70% of outputs had fused words) "
      "and looked worse than pretrained on ROUGE-Lsum. We re-split highlights with a heuristic that protects real "
      "camel-case words (iPhone, DailyMailTV, McConnell). After that, fused words in outputs: <b>0%</b>."),
    B("<b>Fused article sentences</b> (<i>“sideline.Feki was…”</i>) in every article — re-spaced in run 3. Known leftover: "
      "straight quotes get the space on the wrong side in 49/418 test articles; to be fixed before the CLIP stage."),

    P("3. Results — final text baseline (run 3, 418 test stories)", "h"),
    table([["Metric", "Pretrained", "Fine-tuned", "Gain (points)", "95% CI fine-tuned"]]
          + [[n, f3(pre[m]["mean"]), f"<b>{f3(ft[m]['mean'])}</b>", pts(m),
              f"{f3(ft[m]['ci95'][0])}–{f3(ft[m]['ci95'][1])}"]
             for m, n in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L"), ("rougeLsum", "ROUGE-Lsum"))],
          [0.22, 0.18, 0.18, 0.18, 0.24]),
    Spacer(1, 3),
    P("A paired bootstrap prefers fine-tuned in ≥99.9% of resamples for every metric. Run 2 (before article cleaning, "
      "4 epochs) gave the same numbers within 0.6 points, so the result is reproducible. Validation picked epoch 3; "
      "epochs 4–6 over-fit.", "p"),
    Image(f"{FIG}/rouge_test.png", width=10 * cm, height=10 * cm * 680 / 1260),
    P("Test ROUGE F1, pretrained (orange) vs fine-tuned (blue); error bars are bootstrap 95% confidence intervals.", "cap"),
]

page2 = [
    P("4. What the summaries actually look like", "h"),
    table([["Property (test set)", "Human reference", "Pretrained", "Fine-tuned"],
           ["Sentences copied word-for-word from the article", "2%", "47%", "26%"],
           ["Stories with a name/number not in the article", "49%*", "3%", "13%"],
           ["Average length (words)", "59", "61", "69"],
           ["Outputs cut off mid-sentence (160-token limit)", "–", "1.4%", "5.7%"]],
          [0.46, 0.18, 0.18, 0.18]),
    Spacer(1, 2),
    P("*References use background knowledge (e.g. first names the article never states); shown for scale only.", "cap"),
    P("Fine-tuning makes BART write more like the human highlights (less copying), and that is where the ROUGE gain comes from — "
      "but paraphrasing also introduces errors ROUGE cannot see. Verified examples: <i>“shot Princess, 8”</i> (Princess is the "
      "dog; the 8-year-old is the owner), an invented surname, and a summary that reversed the article's meaning. "
      "<b>We should always report an error analysis next to ROUGE.</b>", "p"),

    P("5. Ground rules for the next stages", "h"),
    B("<b>Use the cleaned text in <font face='Courier'>outputs/full_run_v3/clean_{train,val,test}.jsonl</font></b> — "
      "it is exactly what the model saw. Do not re-read raw highlight.txt files without the same cleaning."),
    B("<b>Never tune anything on test.</b> Choose thresholds, prompts, image counts, etc. on validation "
      "(val predictions are in <font face='Courier'>val_predictions.csv</font>); run test once per final variant."),
    B("<b>Differences under ~0.6 ROUGE points are noise</b> (that is how much two runs of the same setup differ). "
      "A modality only “helps” if it beats this and survives the paired bootstrap."),
    B("<b>Read outputs, not just scores.</b> Both data bugs were found by looking at predictions."),
    B("<b>Compute budget:</b> one Kaggle T4 run of the full notebook ≈ 50–60 min of the 30 h/week free quota; "
      "always do a pilot run (PILOT = True) first."),

    P("6. Where things are", "h"),
    table([["Path", "What it is"],
           ["src/build_splits.py, src/make_kaggle_bundle.py", "Build splits; build the text-only bundle uploaded to Kaggle as dataset “mmavs-text”"],
           ["notebooks/finetune_bart_kaggle.ipynb", "Cleaning, baseline, fine-tuning, evaluation (Kaggle notebook, version 3)"],
           ["outputs/full_run_v1 … v3/", "results.json, test predictions with per-story ROUGE; v3 adds cleaned splits + val predictions"],
           ["Kaggle output bart-mmavs-best/", "Fine-tuned checkpoint (1.6 GB) — stays on Kaggle, not in the repo"],
           ["report/", "University report draft (.docx), PDF preview, and the scripts that build it"]],
          [0.40, 0.60]),
    Spacer(1, 2),
    P("data/ and the Kaggle bundle are git-ignored (too large); the new scripts and notebook are not committed yet.", "cap"),

    P("7. Next steps", "h"),
    B("Fix the quote spacing in the article cleaner (small, CPU only)."),
    B("<b>CLIP stage:</b> score each image and caption against the article and the baseline summary; evaluate image selection and "
      "whether adding relevant captions improves the text summary (tuned on validation)."),
    B("<b>Whisper</b> for stories without transcripts, then <b>video key frames</b> scored like images."),
    B("Final comparison of all variants on the same 418 test stories + a small manual faithfulness check."),
    B("<b>Report (due 2 Nov):</b> fill in names/USNs/guide placeholders, Chapter 7 (personal), and the published MM-AVS "
      "baseline numbers in Section 5.2.3."),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 1.1 * cm, "Multimodal News Summarizer · team progress summary")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page} of 2")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
                        bottomMargin=1.7 * cm, title="Multimodal News Summarizer — team progress summary")
doc.build(story + [PageBreak()] + page2, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
