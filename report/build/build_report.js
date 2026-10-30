const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, HeadingLevel, PageBreak, Footer, PageNumber, NumberFormat, BorderStyle,
  TableOfContents, ShadingType, LevelFormat, TabStopType, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "fig");
const OUT = process.argv[2];

const FONT = "Times New Roman";
const RED = "C00000";
const cm = (x) => Math.round(x * 567); // DXA
const TEXT_WIDTH = 11906 - cm(3.81) - cm(2.54); // A4 width minus margins

// ---------- inline helpers ----------
// Strings may contain **bold**, *italic* and [[placeholder]] (red, to be filled by the student).
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("[[")) out.push(new TextRun({ text: "[" + t.slice(2, -2) + "]", color: RED, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) =>
  new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 360, ...(opts.spacing || {}) }, indent: opts.indent, keepNext: opts.keepNext,
    pageBreakBefore: opts.pageBreakBefore });
const C = (text, run = {}, after = 120) =>
  new Paragraph({ children: runs(text, run), alignment: AlignmentType.CENTER, spacing: { after, line: 276 } });
const bullet = (text, level = 0) =>
  new Paragraph({ children: runs(text), numbering: { reference: "bullets", level },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 360 } });
const numbered = (text, ref = "steps") =>
  new Paragraph({ children: runs(text), numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 360 } });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ children: [], spacing: { after: 0 } }));

const chapter = (num, title) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER,
    children: [new TextRun(`Chapter ${num}: ${title}`)], spacing: { after: 240 } });
const unnumberedChapter = (title) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER,
    children: [new TextRun(title)], spacing: { after: 240 } });
const section = (title) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(title)], keepNext: true });
const subsection = (title) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(title)], keepNext: true });
const frontTitle = (title) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: title, bold: true, underline: {}, size: 28, allCaps: true })] });

// ---------- figures & tables ----------
function figure(file, widthCm, caption) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  const wpx = widthCm * 37.8;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: wpx * h / w } })] }),
    (CAPTIONS.push(caption), new Paragraph({ style: "FigureCaption", children: runs(caption) })),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round(w * TEXT_WIDTH / total));
  scaled[scaled.length - 1] += TEXT_WIDTH - scaled.reduce((a, b) => a + b, 0);
  const cell = (text, i, head) =>
    new TableCell({ borders, width: { size: scaled[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      shading: head ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ alignment: (opts.align && opts.align[i]) || AlignmentType.LEFT, spacing: { after: 0, line: 240 },
        children: runs(String(text), { size: 20, bold: head || undefined }) })] });
  return [
    (CAPTIONS.push(caption), new Paragraph({ style: "TableCaption", keepNext: true, children: runs(caption) })),
    new Table({ width: { size: TEXT_WIDTH, type: WidthType.DXA }, columnWidths: scaled,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
             ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] }),
    new Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}
const R = AlignmentType.RIGHT, L = AlignmentType.LEFT, CE = AlignmentType.CENTER;

// ---------- results (v2 run; replace when v3 lands) ----------
const res = JSON.parse(fs.readFileSync(process.argv[3]));
const f3 = (x) => x.toFixed(3);
const ci = (name, m) => `${f3(res.test[name][m].mean)} (${f3(res.test[name][m].ci95[0])}–${f3(res.test[name][m].ci95[1])})`;
const METRICS = [["rouge1", "ROUGE-1"], ["rouge2", "ROUGE-2"], ["rougeL", "ROUGE-L"], ["rougeLsum", "ROUGE-Lsum"]];

// ============ FRONT PAGES (no page numbers) ============
function titleBlock() {
  return [
    ...blank(1),
    C("[[Insert RV University logo]]", { size: 24 }, 360),
    C("School of Computer Science and Engineering", { bold: true, size: 28 }, 120),
    C("November, 2026", { size: 24 }, 480),
    C("MULTIMODAL NEWS SUMMARIZATION ON MM-AVS: A LEAKAGE-CONTROLLED BART BASELINE AND DATA-QUALITY STUDY",
      { bold: true, size: 40 }, 360),
    C("Submitted in partial fulfillment for the award of the degree of", { italics: true, size: 28 }, 240),
    C("[[Name of the degree]]", { bold: true, size: 44 }, 360),
    C("by", { italics: true, size: 28 }, 240),
    C("[[NAME OF THE CANDIDATE(S)]]", { bold: true, size: 32, allCaps: true }, 60),
    C("[[USN]]", { bold: true, size: 32 }, 360),
    C("Guided by", { italics: true, size: 28 }, 120),
    C("[[GUIDE NAME]]", { bold: true, size: 32, allCaps: true }, 0),
  ];
}

const sigCell = (lines, w) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
  children: lines.map((l, i) => new Paragraph({ spacing: { after: 0, line: 276 }, children: runs(l, i === 0 ? { bold: true } : {}) })),
});
const noBorderTable = (cols, rowsOfLines) => {
  const w = Math.floor(TEXT_WIDTH / cols);
  return new Table({ width: { size: w * cols, type: WidthType.DXA }, columnWidths: Array(cols).fill(w),
    borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE },
               right: { style: BorderStyle.NONE }, insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } },
    rows: rowsOfLines.map((r) => new TableRow({ children: r.map((lines) => sigCell(lines, w)) })) });
};

const TITLE = "Multimodal News Summarization on MM-AVS: A Leakage-Controlled BART Baseline and Data-Quality Study";

const certificate = [
  pageBreak(),
  C("[[Insert RV University logo]]", { size: 24 }, 240),
  C("School of Computer Science and Engineering", { bold: true, size: 28 }, 60),
  C("RV University", {}, 60),
  C("RV Vidyaniketan, 8th Mile, Mysore Road, Bengaluru, Karnataka, India - 560059", {}, 360),
  C("CERTIFICATE", { bold: true, size: 28 }, 240),
  P(`This is to certify that the project work titled “${TITLE}” is performed by [[Name]] ([[USN]]), a bonafide student of [[Name of the degree]] at the School of Computer Science and Engineering, RV University, Bengaluru in partial fulfillment for the award of degree [[Name of the degree]] in Computer Science & Engineering, during the Academic year **2026-2027**.`),
  ...blank(2),
  noBorderTable(3, [[
    ["[[Guide Name]]", "Guide", "[[Assistant Professor]]", "SoCSE", "RV University", "Date:"],
    ["Dr. Rashmi S", "[[B.Tech(H)]]", "Program Director", "SoCSE", "RV University", "Date:"],
    ["Dr. G Shobha", "", "Dean", "SoCSE", "RV University", "Date:"],
  ]]),
  ...blank(3),
  noBorderTable(2, [
    [["Signature of the Mentor"], ["Signature of the Guide"]],
  ]),
  ...blank(2),
  noBorderTable(2, [
    [["Name of the Examiner", "", "1.", "", "2."], ["Signature of Examiner", "", "1.", "", "2."]],
  ]),
];

const declaration = [
  pageBreak(),
  C("[[Insert RV University logo]]", { size: 24 }, 240),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: "DECLARATION", bold: true, underline: {}, size: 28 })] }),
  P(`I hereby declare that the thesis entitled “${TITLE}” submitted by me, for the award of the degree [[Name of the degree]] in RV University is a record of Bonafide work carried out by me under the supervision of [[Guide Name]].`, { indent: { firstLine: 720 } }),
  P("I further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university.", { indent: { firstLine: 720 } }),
  ...blank(6),
  new Paragraph({ children: [new TextRun("Place: Bangalore")], spacing: { after: 120 } }),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: TEXT_WIDTH }],
    children: [new TextRun("Date:"), new TextRun({ text: "\tSignature of the Candidate", bold: true })] }),
];

// ============ FRONT MATTER (roman numerals) ============
const abstract = [
  frontTitle("Abstract"),
  P("News today is published as a bundle of text, photographs, audio and video, yet most automatic summarizers read only the article text. This project builds the text foundation for a multimodal news summarizer on MM-AVS, a public dataset of Daily Mail and CNN stories that pairs each article with human-written highlights, images with captions, audio, video and transcripts."),
  P("Before any model was trained, the dataset was audited. The 2,224 article–summary pairs contain only 1,935 distinct stories: 139 stories were scraped repeatedly, up to 12 times. A duplicate-grouping procedure (exact matching plus 8-word shingle Jaccard similarity) was therefore used to build train, validation and test splits in which no story crosses a split boundary, while keeping the official MM-AVS Daily Mail test list. The audit also found that the reference highlights of most Daily Mail stories are stored fused together without any separator (for example “last FridayRaiders seen on CCTV”), and that every article has sentences joined without spaces. A first fine-tuning run trained on the raw references learned to reproduce these fused words in 70% of its Daily Mail outputs and appeared to lose on ROUGE-Lsum; a heuristic re-splitting procedure, validated against camel-case words that occur in the articles, removed the artefact completely."),
  P(`With clean references, BART-large-CNN fine-tuned on 1,358 training stories improves over the pretrained model on all four ROUGE measures on 418 held-out test stories (ROUGE-1 ${f3(res.test.pretrained.rouge1.mean)} → ${f3(res.test.finetuned.rouge1.mean)}, ROUGE-2 ${f3(res.test.pretrained.rouge2.mean)} → ${f3(res.test.finetuned.rouge2.mean)}, ROUGE-Lsum ${f3(res.test.pretrained.rougeLsum.mean)} → ${f3(res.test.finetuned.rougeLsum.mean)}); a paired bootstrap prefers the fine-tuned model in at least 99.9% of resamples. Error analysis shows the trade-off behind the gain: the fine-tuned model copies far less verbatim text but introduces unsupported names or numbers in 13% of stories, including verified entity and meaning errors. These results form the text-only baseline against which the planned image (CLIP), audio (Whisper) and video stages will be compared.`),
];

const acknowledgement = [
  frontTitle("Acknowledgement"),
  P("It is my pleasure to express with deep sense of gratitude to [[Internal Guide name]], [[Designation, School of Computer Science and Engineering, RV University]], for his/her constant guidance, continual encouragement, understanding; more than all, he/she taught me patience in my endeavor. My association with him / her is not confined to academics only, but it is a great opportunity on my part of work with an intellectual and expert in the field of natural language processing."),
  P("I would like to express my gratitude to Prof Dr Ram Kumar Kakani, Vice Chancellor RV University, and Prof Dr Shobha G, Dean SoCSE, RV University, for providing with an environment to work in and for his inspiration during the tenure of the course."),
  P("In jubilant mood I express ingeniously my whole-hearted thanks to Dr. Rashmi S Program Director, [[B.Tech(H)]] SoCSE, RV University, all teaching staff and members working as limbs of our university for their not-self-centered enthusiasm coupled with timely encouragements showered on me with zeal, which prompted the acquirement of the requisite knowledge to finalize my course study successfully. I would like to thank my parents for their support."),
  P("I would like to extend my sincere thanks to my mentor, [[Mentor Name]], [[Designation, School of Computer Science and Engineering, RV University]], for their unwavering support, thoughtful guidance, and motivational presence throughout the course of this project. Their mentorship not only enriched my learning but also inspired me to approach challenges with confidence and clarity. Working with [[him/her/them]] has been a meaningful experience, and I am truly grateful for the valuable time and insights shared with me during this journey."),
  P("It is indeed a pleasure to thank my friends who persuaded and encouraged me to take up and complete this task. At last but not least, I express my gratitude and appreciation to all those who have helped me directly or indirectly toward the successful completion of this project."),
  ...blank(2),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: TEXT_WIDTH }],
    children: [new TextRun({ text: "Place: Bangalore", bold: true }), new TextRun({ text: "\t[Name of the student]", bold: true, color: RED })] }),
  new Paragraph({ children: [new TextRun({ text: "Date:", bold: true })] }),
];

// Captions are collected as they are created; page numbers come from a rendered PDF (argv[4]).
const CAPTIONS = [];
const PAGE_MAP = process.argv[4] && fs.existsSync(process.argv[4]) ? JSON.parse(fs.readFileSync(process.argv[4])) : {};
function captionList(kind) {
  return CAPTIONS.filter((c) => c.startsWith(kind)).map((c) => {
    const [label, ...rest] = c.split(": ");
    return new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 1000, hanging: 1000 },
      tabStops: [{ type: TabStopType.LEFT, position: 1000 }, { type: TabStopType.RIGHT, position: TEXT_WIDTH - 20, leader: "dot" }],
      children: [new TextRun(label.replace(/^(Figure|Table) /, "") + "\t" + rest.join(": ").replace(/\*/g, "") + "\t" + (PAGE_MAP[label] || ""))] });
  });
}

const contents = () => [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "CONTENTS", bold: true, size: 28 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ pageBreakBefore: true, spacing: { after: 240 },
    children: [new TextRun({ text: "LIST OF FIGURES", bold: true, size: 28 })] }),
  ...captionList("Figure"),
  new Paragraph({ pageBreakBefore: true, spacing: { after: 240 },
    children: [new TextRun({ text: "LIST OF TABLES", bold: true, size: 28 })] }),
  ...captionList("Table"),
];

const ACRONYMS = [
  ["ASR", "Automatic Speech Recognition"], ["BART", "Bidirectional and Auto-Regressive Transformer"],
  ["BPE", "Byte-Pair Encoding"], ["CI", "Confidence Interval"], ["CLIP", "Contrastive Language–Image Pre-training"],
  ["CNN", "Cable News Network (news source; not convolutional neural network in this report)"], ["DM", "Daily Mail"],
  ["F1", "Harmonic mean of precision and recall"], ["fp16", "16-bit floating point (half precision)"],
  ["GPU", "Graphics Processing Unit"], ["LCS", "Longest Common Subsequence"],
  ["MM-AVS", "Multi-Modal Article–Video Summarization dataset (Fu et al., 2021)"],
  ["MSMO", "Multimodal Summarization with Multimodal Output"], ["NLP", "Natural Language Processing"],
  ["ROUGE", "Recall-Oriented Understudy for Gisting Evaluation"], ["T4", "NVIDIA Tesla T4 GPU (16 GB)"],
  ["USN", "University Seat Number"],
];
const acronyms = [
  new Paragraph({ pageBreakBefore: true, spacing: { after: 240 },
    children: [new TextRun({ text: "LIST OF ACRONYMS", bold: true, size: 28 })] }),
  ...ACRONYMS.map(([a, b]) => new Paragraph({ tabStops: [{ type: TabStopType.LEFT, position: 1700 }], spacing: { after: 80, line: 276 },
    children: [new TextRun(a), new TextRun("\t" + b)] })),
];

// ============ CHAPTERS ============
const pre = res.test.pretrained, ft = res.test.finetuned;
const ch1 = [
  chapter(1, "Introduction"),
  section("1.1 INTRODUCTION"),
  P("Automatic summarization condenses a document into a short text that keeps its most important content. For news, this task has been studied mainly as a text-to-text problem: a model reads an article and produces a few highlight sentences. Online news, however, is no longer text-only. A typical Daily Mail story arrives with several photographs and their captions, an embedded video, an audio track and sometimes a transcript. Multimodal summarization asks whether these additional signals can make summaries more informative, and how the right image or video frame can be selected to accompany the text."),
  P("This report describes the first phase of a multimodal news summarization project built on the MM-AVS dataset (Fu et al., 2021). The phase covers everything that a credible multimodal comparison depends on: obtaining and auditing the full dataset, constructing leakage-free experimental splits, establishing a strong text-only baseline with the pretrained BART-large-CNN model (Lewis et al., 2020), fine-tuning that model on MM-AVS, and analysing its errors. The image, audio and video stages are designed and scheduled but not yet evaluated; they are described in Chapter 8 as future work."),
  P("A central lesson of this phase is that data quality, not model choice, dominated the early results. The audit uncovered heavy duplication across the dataset and fused reference summaries; both would have produced misleading numbers if left untreated. The report therefore gives as much weight to the data pipeline as to the model."),
  section("1.2 PROBLEM STATEMENT"),
  P("Given a news article and its associated media, generate a short abstractive summary that captures the key facts of the story. In this phase the input is restricted to the article text, and the goal is to establish how well a state-of-the-art pretrained summarizer performs on MM-AVS before and after domain fine-tuning, measured on held-out stories that the model has never seen in any form."),
  section("1.3 OBJECTIVES"),
  numbered("Obtain the complete MM-AVS dataset and audit it for completeness, duplication and formatting defects.", "obj"),
  numbered("Build reproducible train, validation and test splits that keep the official Daily Mail test list and prevent duplicate stories from leaking across splits.", "obj"),
  numbered("Measure the zero-shot performance of pretrained BART-large-CNN on the test split.", "obj"),
  numbered("Fine-tune BART-large-CNN on MM-AVS within free GPU limits, selecting the checkpoint on validation data only.", "obj"),
  numbered("Evaluate both models with ROUGE, confidence intervals and paired significance testing, and analyse errors qualitatively.", "obj"),
  numbered("Prepare the cleaned data and baseline outputs for the planned image, audio and video experiments.", "obj"),
  section("1.4 SCOPE OF THE PROJECT"),
  P("The work reported here is limited to text summarization in English on the 2,224 article–summary pairs of MM-AVS. Only the BART family is evaluated; other summarizers such as PEGASUS (Zhang et al., 2020) are discussed but not run. Evaluation uses automatic ROUGE metrics supported by a small manual error analysis; a full human evaluation is outside the current scope. All training runs on a single free-tier Kaggle T4 GPU, and the development machine is an 8 GB laptop, which shaped several design choices (gradient checkpointing, mixed precision and a single random seed). Multimodal modelling is in scope for the project as a whole but is not yet part of the reported results."),
];

const LIT = [
  ["2004", "Lin", "ROUGE n-gram and LCS overlap metrics", "Standard automatic evaluation for summarization"],
  ["2015", "Hermann et al.", "CNN/Daily Mail reading-comprehension corpus", "Source of the news-highlight summarization benchmark"],
  ["2016", "Nallapati et al.", "Attentional encoder–decoder RNNs", "First strong abstractive results on CNN/DM"],
  ["2017", "See et al.", "Pointer-generator network with coverage", "Copying from source; reduces repetition"],
  ["2017", "Li et al.", "Multimodal summarization of asynchronous text, image, audio, video", "Early joint use of four modalities"],
  ["2018", "Zhu et al.", "MSMO: multimodal summary with image output", "Summary plus selected image; Daily Mail data"],
  ["2019", "Liu & Lapata", "BERTSum: pretrained encoders for summarization", "Transfer learning for extractive/abstractive"],
  ["2020", "Lewis et al.", "BART denoising sequence-to-sequence pretraining", "State-of-the-art abstractive summarization"],
  ["2020", "Zhang et al.", "PEGASUS gap-sentence pretraining", "Summarization-specific pretraining objective"],
  ["2020", "Maynez et al.; Kryscinski et al.", "Faithfulness and factual-consistency analysis", "Abstractive models hallucinate unsupported facts"],
  ["2021", "Radford et al.", "CLIP contrastive image–text pretraining", "Zero-shot image–text relevance scoring"],
  ["2021", "Fu et al.", "MM-AVS dataset", "Full-scale multimodal dataset used in this project"],
  ["2023", "Radford et al.", "Whisper large-scale weakly supervised ASR", "Robust transcription for the audio stage"],
];
const ch2 = [
  chapter(2, "Background"),
  section("2.1 LITERATURE SURVEY"),
  P("This section reviews the methods that led to the current project, in chronological order. Table 2.1 summarizes the survey."),
  subsection("2.1.1 EVALUATION AND DATASETS (2004–2015)"),
  P("Lin (2004) introduced ROUGE, a family of recall-oriented overlap measures between a candidate summary and human references. ROUGE-1 and ROUGE-2 count shared unigrams and bigrams, while ROUGE-L uses the longest common subsequence (LCS) to reward in-order overlap. The summary-level variant ROUGE-Lsum computes a union LCS over sentences and therefore depends on correct sentence boundaries, a detail that became important in this project. Hermann et al. (2015) collected CNN and Daily Mail articles with their bullet-point highlights for reading comprehension; the corpus was soon adopted as the standard news summarization benchmark and is the textual ancestor of MM-AVS."),
  subsection("2.1.2 NEURAL ABSTRACTIVE SUMMARIZATION (2016–2017)"),
  P("Nallapati et al. (2016) applied attentional recurrent encoder–decoders to CNN/Daily Mail and showed that neural models can write new sentences rather than only extract them. See et al. (2017) added a pointer mechanism that copies words directly from the source and a coverage penalty against repetition; the pointer-generator became the reference abstractive model for several years. These models made the extractive–abstractive trade-off explicit: copying preserves facts, while generation produces more concise text at the risk of errors."),
  subsection("2.1.3 MULTIMODAL SUMMARIZATION (2017–2018)"),
  P("Li et al. (2017) proposed summarizing asynchronous collections of text, images, audio and video by aligning speech transcripts and image captions with the text and selecting salient content across modalities. Zhu et al. (2018) defined multimodal summarization with multimodal output (MSMO): the system writes a text summary and also picks the most relevant image, using a Daily Mail corpus with images and captions. Both works showed that visual signals can guide content selection, but relied on recurrent models trained from scratch."),
  subsection("2.1.4 PRETRAINED TRANSFORMERS (2019–2020)"),
  P("Transformer models (Vaswani et al., 2017) pretrained on large unlabeled corpora changed the field. Liu and Lapata (2019) fine-tuned BERT encoders for summarization. Lewis et al. (2020) introduced BART, a denoising sequence-to-sequence model whose bidirectional encoder and autoregressive decoder suit summarization naturally; BART-large fine-tuned on CNN/Daily Mail (released as BART-large-CNN) reported ROUGE-1/2/L of 44.16/21.28/40.90. Zhang et al. (2020) designed PEGASUS with a gap-sentence objective tailored to summarization. In the same year, Maynez et al. (2020) and Kryscinski et al. (2020) showed that such abstractive models frequently produce statements that are not supported by the source, so ROUGE alone can overstate quality."),
  subsection("2.1.5 VISION–LANGUAGE AND SPEECH MODELS (2021–2023)"),
  P("Radford et al. (2021) trained CLIP to embed images and text in a shared space using 400 million image–caption pairs, enabling zero-shot image–text relevance scores. Fu et al. (2021) released MM-AVS, extending Daily Mail and CNN stories with videos, audio and transcripts in addition to images and captions, so that all modalities can be studied on the same articles. Radford et al. (2023) released Whisper, a speech recognizer trained on 680,000 hours of weakly supervised audio that transcribes noisy real-world speech robustly. CLIP and Whisper are the planned components of this project's image and audio stages."),
  ...table("Table 2.1: Year-wise summary of the literature survey", ["Year", "Authors", "Method", "Relevance to this project"],
    LIT, [8, 20, 36, 36]),
  section("2.2 TOOLS AND TECHNIQUES"),
  subsection("2.2.1 BART-LARGE-CNN"),
  P("BART-large has 12 encoder and 12 decoder layers (about 406 million parameters) and reads at most 1,024 subword tokens. The facebook/bart-large-cnn checkpoint has already been fine-tuned on CNN/Daily Mail, which makes it a strong zero-shot baseline for MM-AVS but also raises a possible contamination concern discussed in Chapter 6."),
  subsection("2.2.2 SOFTWARE AND HARDWARE"),
  bullet("**Python 3.12, PyTorch 2.11 and Hugging Face Transformers 5.16** (Paszke et al., 2019; Wolf et al., 2020) for modelling."),
  bullet("**rouge-score** (Google's reference implementation) with Porter stemming for evaluation."),
  bullet("**Kaggle Notebooks** with an NVIDIA Tesla T4 (16 GB) GPU, a free quota of 30 GPU-hours per week, used for all training and test inference."),
  bullet("**AdamW** optimizer (Loshchilov & Hutter, 2019) with linear warm-up and decay, automatic mixed precision (fp16) and gradient checkpointing to fit BART-large into T4 memory."),
  bullet("**Git** for version control and an 8 GB MacBook Air for data processing, auditing and analysis."),
  subsection("2.2.3 STATISTICAL TECHNIQUES"),
  P("Reported scores carry 95% confidence intervals from a non-parametric bootstrap over test stories (Efron & Tibshirani, 1993) with 1,000 resamples. Differences between two systems are tested with a paired bootstrap (Koehn, 2004): test stories are resampled with replacement and the fraction of resamples in which the fine-tuned model has the higher mean is reported. Near-duplicate detection uses word shingles and Jaccard resemblance (Broder, 1997)."),
];

const ch3 = [
  chapter(3, "Methodology"),
  section("3.1 PROPOSED METHODOLOGY"),
  P("The methodology is a staged pipeline (Figure 3.1). The completed stages turn the raw MM-AVS download into clean, leakage-free data and produce a measured text-only baseline. The planned stages add one modality at a time so that the contribution of each can be measured against that baseline under identical splits and metrics."),
  ...figure("pipeline.png", 15.5, "Figure 3.1: Overall project pipeline. Solid boxes are completed; dashed boxes are planned."),
  subsection("3.1.1 DUPLICATE-AWARE SPLITTING"),
  P("MM-AVS was built by scraping news sites, and the same story often appears in several folders, sometimes with small edits. If one copy lands in training and another in test, the model is evaluated on a story it has memorized. To prevent this, stories are grouped with a union–find structure. Two articles join the same group if (a) their normalized article texts are identical, (b) their reference summaries are identical, or (c) the Jaccard similarity of their sets of 8-word shingles exceeds 0.3. Pairs are found efficiently with an inverted index; shingles occurring in more than 50 documents are ignored as boilerplate. The threshold is deliberately low: grouping two different stories only costs a little randomness, whereas missing a duplicate leaks test data."),
  P("Splits are then assigned to whole groups. Every group containing an article from the official MM-AVS Daily Mail test list goes to test. Remaining Daily Mail groups are split 90/10 into train and validation, and CNN groups (which have no official split) 70/10/20 into train, validation and test, with a fixed random seed of 42. Within each group one representative is chosen; models are trained and scored on representatives only, so each story counts once."),
  subsection("3.1.2 TEXT CLEANING"),
  P("The audit (Section 4.1) revealed that Daily Mail highlights are stored without separators and that articles contain sentences run together without spaces. Two cleaning functions are applied identically for every model:"),
  bullet("**Reference re-splitting.** A highlight boundary is inserted where a lowercase letter, digit or closing quotation mark is immediately followed by a capitalized word, an acronym or the words “A”/“I”, and where sentence punctuation is followed directly by a capital letter. A boundary is not inserted if it falls inside a camel-case word that appears at least twice in the correctly spaced articles (for example iPhone, YouTube, DailyMailTV, MeToo) or after a Mc/Mac prefix. Each recovered highlight is terminated with a full stop, so that sentence-level metrics see real sentence boundaries."),
  bullet("**Article re-spacing.** A space is inserted after sentence punctuation that is directly followed by a capital letter or opening quotation mark, non-breaking spaces are normalized, the “(CNN)” dateline is removed and a small list of CNN boilerplate lines (copyright and attribution notes) is dropped. Straight double quotes are ambiguous, so a quote directly after sentence punctuation is assumed to close the sentence; this is wrong when it opens the next one (Chapter 8)."),
  subsection("3.1.3 BASELINE AND FINE-TUNING"),
  P("The pretrained BART-large-CNN model is first evaluated on the test set with fixed generation settings. The same model is then fine-tuned on the training representatives with a standard cross-entropy objective. After each epoch the model generates summaries for the validation set; the epoch with the highest mean of ROUGE-1, ROUGE-2 and ROUGE-L is saved, reloaded and evaluated once on the test set. Generation settings were fixed before any test evaluation and are identical for both models, so the comparison isolates the effect of fine-tuning."),
  subsection("3.1.4 EVALUATION PROTOCOL"),
  P("ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum F1 are computed per story with stemming, with one sentence per line for ROUGE-Lsum. Means are reported with bootstrap 95% confidence intervals and the two models are compared with a paired bootstrap. Because ROUGE measures overlap rather than correctness, every run is followed by an error analysis of the saved predictions: formatting artefacts, length, the share of sentences copied verbatim from the article, and names or numbers that do not occur in the article, with manual verification of suspicious cases."),
  subsection("3.1.5 PLANNED MULTIMODAL STAGES"),
  P("The next stages reuse the cleaned splits and the text baseline: CLIP scores the relevance of each image and caption to the article and generated summary; Whisper transcribes the audio track where no transcript exists; key frames are sampled from the videos and scored the same way as images. Each modality is added separately and evaluated on the same test stories, and any tuning is performed on validation data only."),
];

const DS = [
  ["Article–summary pairs", "1,972", "252", "2,224"],
  ["Mean article length (words)", "675 (median 474)", "1,003 (median 912)", "–"],
  ["Mean reference length (words)", "62", "30", "–"],
  ["Images per article (mean)", "2.9", "0", "5,672 images"],
  ["Articles with image captions", "91%", "0%", "–"],
  ["Articles with transcript", "46%", "73%", "–"],
  ["Articles with audio / video", "99% / 99%", "0% / 81%", "–"],
];
const SPLITS = [
  ["Train", "1,280", "177", "1,457", "1,358"],
  ["Validation", "142", "25", "167", "159"],
  ["Test", "550", "50", "600", "418 (369 DM + 49 CNN)"],
  ["Total", "1,972", "252", "2,224", "1,935"],
];
const ISSUES = [
  ["Repeated scraping of the same story", "139 stories have 2–12 copies (428 articles; 892 near-duplicate pairs)", "Duplicate-aware group splits; one representative per story"],
  ["Official test IDs absent locally", "4 of 394 official Daily Mail test IDs", "Reported; test uses the 390 available"],
  ["CNN article file misspelled", "CNN folders name the article artitle.txt", "Loader accepts both file names"],
  ["Fused Daily Mail highlights", "e.g. “last FridayRaiders seen…”; 327 of 369 DM test references affected", "Highlight re-splitting (Section 3.1.2)"],
  ["Sentences fused in articles", "Every article; median 11 (DM) and 15 (CNN) per article", "Article re-spacing (Section 3.1.2)"],
  ["CNN dateline and boilerplate", "“(CNN)The…”, copyright and attribution lines", "Removed during cleaning"],
  ["Long articles", "About 25% exceed BART's 1,024-token input", "Truncated; noted as a limitation"],
];
const HP = [
  ["Model", "facebook/bart-large-cnn (406M parameters)"],
  ["Max input / target length", "1,024 / 192 tokens"],
  ["Optimizer", "AdamW, learning rate 2e-5, weight decay 0.01, gradient clipping 1.0"],
  ["Schedule", "Linear warm-up over 10% of steps, then linear decay"],
  ["Batch size", "2 per step × 4 accumulation steps = effective 8"],
  ["Epochs", "4 (runs 1–2); 6 (run 3); best epoch chosen on validation"],
  ["Precision / memory", "fp16 mixed precision, gradient checkpointing"],
  ["Generation (both models)", "beam size 4, no-repeat 3-gram, length penalty 2.0, 20–160 new tokens"],
  ["Seed", "42 (data split and training-data shuffling)"],
  ["Hardware", "Kaggle NVIDIA Tesla T4 16 GB; peak memory 8.2 GB; ~6 min per epoch"],
];
const ch4 = [
  chapter(4, "Implementation"),
  section("4.1 DATASETS"),
  P("MM-AVS (Fu et al., 2021) is distributed as one folder per story containing the article, the highlight summary, a title, images with captions, and audio, video and transcript files where available. The Daily Mail portion (folders 4000–5999) was downloaded in full from the authors' cloud share, together with the official Daily Mail test ID list; the CNN portion was already available locally. After discarding folders without an article or a summary, 2,224 article–summary pairs remain (Table 4.1). The full multimodal download occupies about 27 GB."),
  ...table("Table 4.1: MM-AVS statistics after loading", ["Statistic", "Daily Mail", "CNN", "Total"], DS, [34, 24, 24, 18], { align: [L, CE, CE, CE] }),
  P("Table 4.2 shows the splits produced by the duplicate-aware procedure. The test split contains 600 articles but only 418 distinct stories, which illustrates the scale of duplication; all scores in this report are computed over the 418 story representatives. A verification step confirmed that no duplicate group spans two splits; the most similar test–train pair has a shingle Jaccard similarity of 0.296 and, on inspection, describes a related but different story."),
  ...table("Table 4.2: Leakage-free splits (articles and unique stories)", ["Split", "Daily Mail", "CNN", "Articles", "Unique stories"], SPLITS, [18, 16, 12, 16, 38], { align: [L, CE, CE, CE, CE] }),
  P("The data audit identified the defects listed in Table 4.3. The two text-fusion problems were discovered only after the first full training run, through error analysis of its outputs; Section 5.2.1 describes their effect."),
  ...table("Table 4.3: Data-quality issues found and how they were handled", ["Issue", "Extent", "Treatment"], ISSUES, [27, 40, 33]),
  section("4.2 IMPLEMENTATION"),
  P("The implementation is split into small, single-purpose components so that every result can be regenerated from the raw data:"),
  bullet("**build_splits.py** loads every story folder, records which modalities are present, runs duplicate grouping and split assignment, and writes splits.csv (one row per article with its split, group and representative flag) and splits_summary.json."),
  bullet("**make_kaggle_bundle.py** converts the splits into three compact JSON-lines files (train 6.6 MB, validation 0.8 MB, test 3.5 MB) containing only the text fields, so that training does not require uploading the 27 GB multimodal download."),
  bullet("**finetune_bart_kaggle.ipynb** is a self-contained notebook that cleans the text, evaluates the pretrained model, fine-tunes with validation-based checkpoint selection, evaluates the selected checkpoint and saves predictions, per-story scores and a results.json file that records all settings, library versions and the training history."),
  P("A hand-written training loop is used instead of the Transformers Trainer class to keep behaviour stable across library versions and to make mixed precision, gradient accumulation and validation-based selection explicit. Each new notebook version is first exercised in a pilot mode (48 training and 16 test stories) to catch errors cheaply before a full run."),
  section("4.3 EXPERIMENTAL SETUP"),
  P("All hyperparameters are listed in Table 4.4. They follow common practice for fine-tuning BART on news summarization and were not tuned on the test set. Three full runs have been carried out; they differ only in data cleaning and the number of epochs, as summarized in Table 5.3."),
  ...table("Table 4.4: Training and generation settings", ["Setting", "Value"], HP, [30, 70]),
];

const perSource = [
  ["Daily Mail (369)", "0.406 → 0.433", "0.174 → 0.191", "0.271 → 0.291", "0.359 → 0.384", "60%"],
  ["CNN (49)", "0.357 → 0.366", "0.153 → 0.162", "0.246 → 0.265", "0.326 → 0.329", "57%"],
];
const RUNS = [
  ["Run 1", "Raw highlights (only “one.Two” fixed); raw articles; 4 epochs", "0.385 / 0.398", "0.306 / 0.277", "Superseded: model learned fused words"],
  ["Run 2", "Re-split highlights; raw articles; 4 epochs", "0.398 / 0.428", "0.352 / 0.381", "Superseded by run 3 (equivalent ROUGE)"],
  ["Run 3", "Re-split highlights; cleaned articles; 6 epochs (epoch 3 selected)", `${f3(pre.rouge1.mean)} / ${f3(ft.rouge1.mean)}`, `${f3(pre.rougeLsum.mean)} / ${f3(ft.rougeLsum.mean)}`, "Final text baseline; reported in this chapter"],
];
const OUTPUTS = [
  ["Outputs with fused words (DM)", "71% (references, raw)", "0%", "70% (run 1) → 0% (runs 2–3)"],
  ["Sentences per summary", "3.2", "3.4", "3.7"],
  ["Words per summary", "59", "61", "69"],
  ["Sentences copied verbatim from article", "2%", "47%", "26%"],
  ["Stories with a name/number absent from article", "49%*", "3%", "13%"],
  ["Outputs ending mid-sentence", "–", "1.4%", "5.7%"],
];
const ERRORS = [
  ["dailymail/5166", "Meaning reversal", "States the proposed date “would honour Aboriginals”; the article says the proposer does not want a separate day for this.", "Not repeated"],
  ["dailymail/4096", "Entity confusion", "“shot Princess, 8”: Princess is the dog; the 8-year-old is the girl who owned it.", "Persists"],
  ["dailymail/5701", "Invented detail", "Adds the surname “Molo” to a player the article names only as “Feki”.", "Not repeated"],
  ["dailymail/5323", "Inconsistent name", "Spells the same officer “Bostwick” and then “Bostick”.", "Not repeated"],
  ["dailymail/4334", "Non-word", "Generates “Incrediblely”.", "Not repeated"],
];
const ch5 = [
  chapter(5, "Results"),
  P("This chapter reports run 3, the final text-only baseline: references re-split, articles cleaned, and up to six epochs with the best epoch chosen on validation. Its scores are statistically indistinguishable from run 2 (every ROUGE difference between the two runs is below 0.6 points and none is significant under a paired bootstrap), so article cleaning and extra epochs neither helped nor hurt ROUGE. Run 3 is adopted because its pipeline is the cleanest and its outputs contain fewer unsupported names or numbers (13% of stories against 17% in run 2)."),
  section("5.1 RESULT SUMMARY TABLES"),
  P(`Table 5.1 gives the main result. Fine-tuning improves every metric, by ${((ft.rouge1.mean - pre.rouge1.mean) * 100).toFixed(1)} ROUGE-1 points, ${((ft.rouge2.mean - pre.rouge2.mean) * 100).toFixed(1)} ROUGE-2 points, ${((ft.rougeL.mean - pre.rougeL.mean) * 100).toFixed(1)} ROUGE-L points and ${((ft.rougeLsum.mean - pre.rougeLsum.mean) * 100).toFixed(1)} ROUGE-Lsum points. The confidence intervals of the two systems overlap slightly for ROUGE-2 and ROUGE-L, but the paired bootstrap, which accounts for the fact that both systems are scored on the same stories, prefers the fine-tuned model in at least ${(Math.min(...METRICS.map(([m]) => res.test.paired_bootstrap_p_finetuned_better[m])) * 100).toFixed(1)}% of resamples for every metric. Figure 5.1 shows the same numbers graphically.`),
  ...table("Table 5.1: Test results, 418 stories, F1 with bootstrap 95% confidence intervals",
    ["Metric", "Pretrained BART-large-CNN", "Fine-tuned on MM-AVS", "Paired bootstrap P(fine-tuned better)"],
    METRICS.map(([m, n]) => [n, ci("pretrained", m), ci("finetuned", m), res.test.paired_bootstrap_p_finetuned_better[m].toFixed(3)]),
    [16, 30, 30, 24], { align: [L, CE, CE, CE] }),
  ...figure("rouge_test.png", 14.5, "Figure 5.1: Test ROUGE F1 of the pretrained and fine-tuned models; error bars are bootstrap 95% confidence intervals."),
  P("Table 5.2 splits the test set by source. Almost all of the gain comes from Daily Mail, which makes up 88% of the training data; on the 49 CNN test stories the gains are smaller and the fine-tuned model wins on only about half of the stories, so no firm conclusion can be drawn for CNN."),
  ...table("Table 5.2: Test ROUGE by source (pretrained → fine-tuned) and share of stories where fine-tuning scores higher on ROUGE-1",
    ["Source (stories)", "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "Fine-tuned wins"], perSource, [20, 16, 16, 16, 17, 15], { align: [L, CE, CE, CE, CE, CE] }),
  P(`Figure 5.2 shows the training history. Validation ROUGE rose from ${res.history[0].val_rouge_mean.toFixed(4)} after epoch 1 to a peak of ${res.history[res.best_epoch - 1].val_rouge_mean.toFixed(4)} at epoch ${res.best_epoch}, then fell to between ${Math.min(...res.history.slice(res.best_epoch).map((h) => h.val_rouge_mean)).toFixed(4)} and ${Math.max(...res.history.slice(res.best_epoch).map((h) => h.val_rouge_mean)).toFixed(4)} over the remaining epochs, while validation loss climbed from ${res.history[0].val_loss.toFixed(2)} to ${res.history[res.history.length - 1].val_loss.toFixed(2)} and training loss kept falling. The model therefore over-fits after epoch ${res.best_epoch}, and validation-based selection correctly kept epoch ${res.best_epoch}. (In run 2, trained for four epochs, the best epoch was the last one; run 3 was extended to six epochs to check whether training had stopped too early, and it had not.) Training took ${res.training_minutes.toFixed(0)} minutes with a peak GPU memory of ${res.peak_gpu_memory_gb.toFixed(1)} GB.`),
  ...figure("training_curves.png", 15, "Figure 5.2: Run 3 training history: validation ROUGE (left) and training and validation loss (right)."),
  ...table("Table 5.3: History of full training runs (test ROUGE-1 and ROUGE-Lsum, pretrained / fine-tuned)",
    ["Run", "Data and settings", "ROUGE-1", "ROUGE-Lsum", "Status"], RUNS, [9, 37, 15, 15, 24], { align: [L, L, CE, CE, L] }),
  section("5.2 ANALYSIS OF RESULTS"),
  subsection("5.2.1 EFFECT OF THE FUSED REFERENCES"),
  P("Run 1 trained on references in which most Daily Mail highlights were fused together. The fine-tuned model learned this format faithfully: 70% of its Daily Mail outputs contained fused words such as “homeThe couple”, and they averaged 1.3 sentences against 3.1 for the pretrained model. ROUGE-1 and ROUGE-2 still improved, but ROUGE-Lsum, which compares summaries sentence by sentence, fell from 0.306 to 0.277, and the paired bootstrap preferred the pretrained model in every resample. Taken at face value, run 1 would have suggested that fine-tuning harms summary structure. After re-splitting (run 2), fused words disappeared completely, both models scored higher because the references themselves became readable, and fine-tuning improved ROUGE-Lsum by 2.9 points. The episode shows that a data artefact can reverse the conclusion of an experiment, and that inspecting model outputs, not only scores, is necessary."),
  subsection("5.2.2 OUTPUT CHARACTERISTICS AND ERRORS"),
  P("Table 5.4 compares simple properties of the summaries. The pretrained model is largely extractive: 47% of its sentences appear verbatim in the article. The fine-tuned model copies about half as much (26%), which is closer to the human references (2%), but it writes longer summaries and 5.7% of its outputs reach the 160-token generation limit and end mid-sentence."),
  ...table("Table 5.4: Properties of test summaries (run 3, 418 stories)", ["Property", "Reference", "Pretrained", "Fine-tuned"], OUTPUTS, [40, 20, 20, 20], { align: [L, CE, CE, CE] }),
  P("*The reference figure is high mainly because human highlights use first names or background facts that the article states differently (for example “Theresa May” where the article writes “Mrs May”); it is shown for scale, not as an error rate."),
  P("Most unsupported tokens in fine-tuned outputs are harmless background knowledge of the same kind. Manual checking of run 2 outputs, however, also found genuine errors, listed in Table 5.5. Four of the five do not recur in run 3, but the entity confusion persists, which shows that individual errors vary between training runs and that error rates must be measured rather than inferred from a few examples. The pretrained model makes almost none of these errors because it copies source sentences; the fine-tuned model's paraphrasing is what raises ROUGE and also what introduces errors. This agrees with the hallucination findings of Maynez et al. (2020)."),
  ...table("Table 5.5: Factual errors found in run 2 fine-tuned outputs, verified against the articles, and whether they recur in run 3", ["Story", "Error type", "Description (run 2)", "Run 3"], ERRORS, [17, 16, 52, 15]),
  subsection("5.2.3 COMPARISON WITH EXISTING METHODS"),
  P(`Lewis et al. (2020) report ROUGE-1/2/L of 44.16/21.28/40.90 for BART-large on the CNN/Daily Mail test set, where their ROUGE-L corresponds to the summary-level ROUGE-Lsum used here. The pretrained model's scores on MM-AVS are lower (ROUGE-1 ${f3(pre.rouge1.mean)}, ROUGE-Lsum ${f3(pre.rougeLsum.mean)}), which is expected: MM-AVS articles are from a later period, a quarter of them exceed the input limit, and Daily Mail highlights in MM-AVS are longer (62 words on average) than typical CNN/Daily Mail highlights. Results are not directly comparable with the numbers published with MM-AVS (Fu et al., 2021) either, because this project de-duplicates the test set, evaluates each story once and repairs the fused references, all of which change the scores. [[Add the published MM-AVS text-only baseline numbers here after checking them in the original paper, with a note on these differences.]]`),
];

const ch6 = [
  chapter(6, "Inferences"),
  section("6.1 KEY FINDINGS"),
  numbered("**Data quality dominated early results.** Duplicate stories and fused references are each large enough to change conclusions. Without duplicate grouping, up to 12 copies of a story could be split between training and test; without re-splitting, fine-tuning appeared to damage summary structure.", "find"),
  numbered("**Fine-tuning helps, modestly but reliably.** On clean data, domain fine-tuning raises ROUGE-1 by about 2.5 points and ROUGE-Lsum by about 2.2 points over an already strong pretrained summarizer, with paired-bootstrap support of at least 99.9%; two independent training runs (runs 2 and 3) agree on this.", "find"),
  numbered("**The gain comes with a faithfulness cost.** The fine-tuned model is more abstractive and closer in style to the references, but it introduces unsupported content in about one story in eight (13%), including entity and meaning errors that ROUGE does not penalize.", "find"),
  numbered("**Small ROUGE differences are noise.** Cleaning the input articles and training for longer changed no score by more than 0.6 points, and training past three epochs over-fits. Differences of this size should not be read as improvements when the multimodal variants are compared.", "find"),
  numbered("**The pretrained baseline may be optimistic.** BART-large-CNN was fine-tuned on CNN/Daily Mail, which could overlap with some MM-AVS stories; any contamination would favour the pretrained model, so the measured fine-tuning gain is, if anything, a lower bound.", "find"),
  numbered("**Free compute is sufficient for the text stage.** A full fine-tune of a 406M-parameter model takes about 25 minutes on a free Kaggle T4 using mixed precision and gradient checkpointing, leaving the weekly quota for multimodal experiments.", "find"),
  section("6.2 FULFILLMENT OF OBJECTIVES"),
  ...table("Table 6.1: Status of the objectives in Section 1.3", ["Objective", "Status", "Evidence"], [
    ["1. Obtain and audit MM-AVS", "Completed", "Section 4.1, Tables 4.1 and 4.3"],
    ["2. Leakage-free splits", "Completed", "Section 3.1.1, Table 4.2"],
    ["3. Zero-shot baseline", "Completed", "Table 5.1"],
    ["4. Fine-tune within free GPU limits", "Completed", "Tables 4.4, 5.1, 5.3"],
    ["5. Evaluation and error analysis", "Completed", "Section 5.2, Tables 5.4 and 5.5"],
    ["6. Prepare data for multimodal stages", "Completed", "Cleaned splits and validation/test predictions saved by run 3"],
  ], [40, 25, 35]),
];

const ch7 = [
  chapter(7, "Impression Report"),
  P("[[This chapter is personal. The text below is a draft based on the work done; edit it in your own words.]]"),
  section("7.1 PERSONAL LEARNING EXPERIENCE"),
  P("The most valuable lesson of this phase was that a model's numbers are only as trustworthy as the data and the evaluation behind them. The first full training run produced a result that looked like a modelling problem, a drop in ROUGE-Lsum after fine-tuning, but reading the model's outputs revealed that the cause was fused reference summaries in the dataset. Tracing a surprising number back to its root cause, rather than tuning around it, changed how I approach experiments."),
  P("Working within a small budget was also instructive. Training a 406-million-parameter model on an 8 GB laptop is not possible, so the work had to be planned around free cloud GPU time: compact data bundles, pilot runs before full runs, and careful memory management on the GPU."),
  section("7.2 SKILL ENHANCEMENT"),
  bullet("Fine-tuning and evaluating transformer summarization models with PyTorch and Hugging Face Transformers, including mixed precision, gradient accumulation and checkpoint selection."),
  bullet("Dataset auditing: duplicate detection with shingling and union–find, leakage control and data cleaning with regular expressions."),
  bullet("Experimental rigour: separating validation from test, bootstrap confidence intervals, paired significance tests and qualitative error analysis."),
  bullet("Working with cloud notebooks and GPU quotas, version control and reproducible scripts."),
  section("7.3 INDUSTRY/ACADEMIC RELEVANCE"),
  P("News organisations, aggregators and search engines summarize large volumes of multimedia content every day, and faithfulness errors of the kind found here are a known risk when such systems are deployed. Academically, the project contributes a leakage-controlled evaluation protocol and documented cleaning procedures for MM-AVS, which other students and researchers can reuse to obtain comparable results."),
];

const ch8 = [
  chapter(8, "Conclusion & Future Work"),
  P("This phase set out to establish a trustworthy text-only baseline for multimodal news summarization on MM-AVS. It obtained the complete dataset, found that 289 of its 2,224 article–summary pairs are extra copies of other stories, and built splits in which no story is shared between training and test. It found and repaired two text defects, fused reference highlights and fused article sentences, the first of which had reversed the outcome of an early experiment. On clean data, fine-tuning BART-large-CNN on MM-AVS improves ROUGE by two to two and a half points over the pretrained model, with strong statistical support that was reproduced across two training runs."),
  P("The results also have clear limits. Improvements in ROUGE do not mean better summaries in every respect: the fine-tuned model introduces unsupported or wrong statements in a noticeable minority of stories, and ROUGE does not detect them. The test set has only 49 CNN stories, too few for conclusions about that source. Only one random seed was run because of compute limits, about a quarter of articles are truncated at 1,024 tokens, and the pretrained baseline may benefit from overlap with its own training data. The article cleaning also has a known flaw: straight double quotes cannot always be told apart as opening or closing, and in 49 of the 418 test articles a space is placed on the wrong side of an opening quote; since cleaning the articles did not change the scores, this was recorded rather than re-run. The multimodal stages, which are the project's main goal, have not yet been evaluated."),
  P("Planned future work, in order:"),
  numbered("Correct the straight-quote spacing in the article cleaning before the cleaned text is reused by the multimodal stages.", "future"),
  numbered("**Image stage:** use CLIP to score each image and its caption against the article and summary, evaluate image selection and test whether relevant captions improve the text summary.", "future"),
  numbered("**Audio stage:** transcribe audio with Whisper where MM-AVS has no transcript and measure whether transcripts add information not present in the article.", "future"),
  numbered("**Video stage:** sample key frames and score them with CLIP in the same way as images.", "future"),
  numbered("**Comparison and faithfulness:** compare all variants on the same test stories, add a factual-consistency measure and a small human evaluation, and run additional seeds where the quota allows.", "future"),
];

const REFS = [
  "Broder, A. Z. (1997). On the resemblance and containment of documents. In *Proceedings of Compression and Complexity of Sequences 1997* (pp. 21–29). IEEE.",
  "Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
  "Fu, X., Wang, J., & Yang, Z. (2021). MM-AVS: A full-scale dataset for multi-modal summarization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 5922–5926). Association for Computational Linguistics.",
  "Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. In *Advances in Neural Information Processing Systems 28* (pp. 1693–1701).",
  "Koehn, P. (2004). Statistical significance tests for machine translation evaluation. In *Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing* (pp. 388–395). Association for Computational Linguistics.",
  "Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020). Evaluating the factual consistency of abstractive text summarization. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 9332–9346). Association for Computational Linguistics.",
  "Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 7871–7880). Association for Computational Linguistics.",
  "Li, H., Zhu, J., Ma, C., Zhang, J., & Zong, C. (2017). Multi-modal summarization for asynchronous collection of text, image, audio and video. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 1092–1102). Association for Computational Linguistics.",
  "Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out* (pp. 74–81). Association for Computational Linguistics.",
  "Liu, Y., & Lapata, M. (2019). Text summarization with pretrained encoders. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3730–3740). Association for Computational Linguistics.",
  "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*.",
  "Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1906–1919). Association for Computational Linguistics.",
  "Nallapati, R., Zhou, B., dos Santos, C., Gulçehre, Ç., & Xiang, B. (2016). Abstractive text summarization using sequence-to-sequence RNNs and beyond. In *Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning* (pp. 280–290). Association for Computational Linguistics.",
  "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).",
  "Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. (2021). Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning* (pp. 8748–8763). PMLR.",
  "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. In *Proceedings of the 40th International Conference on Machine Learning* (pp. 28492–28518). PMLR.",
  "See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics* (pp. 1073–1083). Association for Computational Linguistics.",
  "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008).",
  "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 38–45). Association for Computational Linguistics.",
  "Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 11328–11339). PMLR.",
  "Zhu, J., Li, H., Liu, T., Zhou, Y., Zhang, J., & Zong, C. (2018). MSMO: Multimodal summarization with multimodal output. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 4154–4164). Association for Computational Linguistics.",
];
const refs = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER,
    children: [new TextRun("REFERENCES")], spacing: { after: 240 } }),
  ...REFS.map((r) => new Paragraph({ children: runs(r), alignment: AlignmentType.LEFT,
    indent: { left: 720, hanging: 720 }, spacing: { after: 120, line: 360 } })),
];

const code = (lines) => lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 },
  children: [new TextRun({ text: l, font: "Courier New", size: 18 })] }));
const appendix = [
  unnumberedChapter("Appendix"),
  section("APPENDIX 1: PROJECT FILES"),
  ...table("Table A1.1: Project files and their roles", ["File", "Role"], [
    ["src/build_splits.py", "Loads MM-AVS, groups duplicates, assigns leakage-free splits; writes outputs/splits.csv and splits_summary.json"],
    ["src/make_kaggle_bundle.py", "Writes text-only train/val/test JSON-lines files for Kaggle"],
    ["notebooks/finetune_bart_kaggle.ipynb", "Cleaning, pretrained baseline, fine-tuning, evaluation and result export"],
    ["outputs/full_run_v1/, outputs/full_run_v2/", "Saved test predictions, per-story scores and results.json of each run"],
  ], [38, 62]),
  section("APPENDIX 2: HIGHLIGHT RE-SPLITTING (EXCERPT)"),
  P("The core of the reference re-splitting procedure described in Section 3.1.2:"),
  ...code([
    "GLUED = re.compile(r\"(?:(?<=[a-z0-9%)])|(?<=[\\w,]['\\u2019\\\"]))\"",
    "                   r\"(?=[A-Z][a-z]|[AI]\\b|[A-Z]{2,}\\b)\")",
    "",
    "def split_highlights(text):",
    "    text = GLUED_STOP.sub(\"\\n\", text.replace(\"\\xa0\", \" \"))",
    "    cuts = [m.start() for m in GLUED.finditer(text)",
    "            if not inside_real_word(text, m.start())]",
    "    for pos in reversed(cuts):",
    "        text = text[:pos] + \"\\n\" + text[pos:]",
    "    ...",
    "",
    "def clean_summary(text):",
    "    highlights = [re.sub(r\"\\s+\", \" \", h) for h in split_highlights(text)]",
    "    return \" \".join(h if h[-1] in \".!?\" else h + \".\" for h in highlights)",
  ]),
  section("APPENDIX 3: EXAMPLE SUMMARIES"),
  P("Story dailymail/4002 (test set). **Reference** (after re-splitting): Police called to couple's £6 million home in Great Tew, Oxfordshire, last Friday. Raiders seen on CCTV in balaclavas before being chased off by restaurant guests. Couple forced to immediately increase security at the countryside mansion."),
  P("**Run 1 fine-tuned output (fused references):** “David and Victoria Beckham have been forced to up security after a break in at their home**The** couple, who have four children, are said to be distraught over the attempted burglary**Police** were called to the couple's £6 million home…” The bold words mark fused sentence boundaries learned from the raw references."),
];

// ---------- document ----------
const footerFor = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT] })] })] });
const page = (fmt, start) => ({
  page: { size: { width: 11906, height: 16838 },
    margin: { left: cm(3.81), right: cm(2.54), top: cm(2.54), bottom: cm(2.54) },
    pageNumbers: fmt ? { start, formatType: fmt } : undefined },
});

const doc = new Document({
  creator: "Multimodal News Summarizer project",
  title: TITLE,
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 0, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, allCaps: true }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, allCaps: true }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
      { id: "FigureCaption", name: "Figure Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 20 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 240 } } },
      { id: "TableCaption", name: "Table Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 20, bold: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ...["obj", "find", "future", "steps"].map((reference) => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL,
      text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
  ] },
  sections: [
    { properties: page(), children: titleBlock() },                         // cover page
    { properties: page(), children: [...titleBlock(), ...certificate, ...declaration] }, // title page, certificate, declaration
    { properties: page(NumberFormat.LOWER_ROMAN, 1), footers: { default: footerFor() },
      children: [...abstract, pageBreak(), ...acknowledgement, ...contents(), ...acronyms] },
    { properties: page(NumberFormat.DECIMAL, 1), footers: { default: footerFor() },
      children: [...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...refs, ...appendix] },
  ],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length, "bytes"); });
