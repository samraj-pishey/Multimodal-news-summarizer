// "How it works": an animated walk-through of the hybrid model on one real test story. Each scene animates one
// step on a single stage (article on the left, video on the right) with one short caption. Everything shown was
// computed offline by src/explainer_export.py.
const $ = (id) => document.getElementById(id);
const MODEL = "#c9513d";
const BASELINE = "#1f6fa8";
const SHOWN_FRAMES = 6;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const LINKS = {
  tokens: ["Hugging Face course", "https://huggingface.co/learn/llm-course/chapter1/1"],
  numbers: ["StatQuest: Word Embedding", "https://www.youtube.com/watch?v=viZrOnJclY0"],
  order: ["StatQuest: LSTM", "https://www.youtube.com/watch?v=YCzL96nL7j0"],
  attention: ["3Blue1Brown: Attention", "https://www.youtube.com/watch?v=eMlx5fFNoYc"],
  transformer: ["StatQuest: Transformers", "https://www.youtube.com/watch?v=zxQyTK8quyY"],
};

const SCENES = [
  { concept: "The task", caption: "A real news article and its video. Our goal: pick the few sentences that matter most.", run: sceneStory },
  { concept: "Step 1 · Tokens", caption: "The computer can't read words, so it first cuts each sentence into small pieces called tokens.", link: "tokens", run: sceneTokens },
  { concept: "Step 2 · Numbers", caption: "RoBERTa turns every sentence into 768 numbers. The barcode shows the first 32: similar sentences get similar barcodes.", link: "numbers", run: sceneNumbers },
  { concept: "Step 3 · Reading in order", caption: "A BiLSTM reads the sentences forwards, then backwards, so each sentence knows where it sits. In news, the first sentences matter most.", link: "order", run: sceneOrder },
  { concept: "Step 4 · Attention", caption: "Each sentence looks across the video and decides which frames relate to it. Thicker lines mean more attention.", link: "attention", run: sceneAttention },
  { concept: "Step 5 · Scores", caption: "Putting it all together, the model gives every sentence a score from 0 to 1.", link: "transformer", run: sceneScores },
  { concept: "Step 6 · Summary", caption: "The highest-scoring sentences become the summary, kept in article order. Nothing is rewritten.", run: sceneSummary },
  { concept: "Step 7 · Grading", caption: "We compare with the journalist's own summary. ROUGE-1 counts the words both share.", run: sceneGrade },
  { concept: "What we found", caption: "Tested on 387 unseen articles: adding video and speech didn't help pick article sentences. Knowing why is part of the result.", run: sceneFindings },
];

const state = { data: null, story: 0, scene: 0, playing: !REDUCED, run: 0 };

// Helpers -----------------------------------------------------------------------------------------------------------
function el(tag, { dataset, ...props } = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  Object.assign(node.dataset, dataset);
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}
function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}
const short = (text, n) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

class Cancelled extends Error {}
// Waits that respect pause, scene changes and "show instantly" (fast) mode.
function timer(run, fast) {
  return async (ms) => {
    if (fast || REDUCED) return;
    let left = ms;
    while (left > 0) {
      await new Promise((r) => setTimeout(r, 50));
      if (run !== state.run) throw new Cancelled();
      if (state.playing) left -= 50;
    }
  };
}

// Stage ---------------------------------------------------------------------------------------------------------------
function story() { return state.data.stories[state.story]; }

function shownFrames(s) {
  const list = s.timeline;
  const step = (list.length - 1) / (SHOWN_FRAMES - 1);
  return [...new Set(Array.from({ length: SHOWN_FRAMES }, (_, i) => list[Math.round(i * step)]))];
}

function buildStage() {
  const s = story();
  const rows = s.sentences.slice(0, 8).map((text, i) => el("div", { className: "a-row", dataset: { i } },
    el("span", { className: "a-num", textContent: i + 1 }),
    el("div", { className: "a-main" },
      el("p", { className: "a-text", textContent: short(text, 120) }),
      el("div", { className: "a-tokens" }),
      el("div", { className: "a-code" })),
    el("div", { className: "a-score" }, el("i"), el("b"))));
  const frames = shownFrames(s).map((i) => el("figure", { className: "a-frame", dataset: { i } },
    el("img", { src: `explainer/frames/${s.thumbnails[i]}`, alt: `Video frame at ${s.frame_times[i].toFixed(0)} seconds` }),
    el("figcaption", { textContent: `${s.frame_times[i].toFixed(0)} s` })));
  const speech = s.speech ? el("div", { className: "a-speech" },
    el("p", { className: "label", textContent: "What is said in the video (Whisper)" }),
    s.speech.sentences.slice(0, 3).map((t) => el("p", { className: "a-bubble", textContent: `“${short(t, 90)}”` }))) : null;

  $("a-stage").replaceChildren(
    el("section", { className: "a-card a-article" }, el("p", { className: "label", textContent: "Article" }), rows),
    el("section", { className: "a-card a-video" }, el("p", { className: "label", textContent: "Video" }),
      el("div", { className: "a-frames" }, frames), speech),
    svg("svg", { class: "a-lines", "aria-hidden": "true" }),
    el("section", { className: "a-card a-out", hidden: true }));
}

const rowsEl = () => [...document.querySelectorAll("#a-stage .a-row")];
const framesEl = () => [...document.querySelectorAll("#a-stage .a-frame")];

// Scenes. Each one can run animated or "fast" (to rebuild the state when jumping to a later scene). ---------------------
async function sceneStory(wait) {
  for (const row of rowsEl()) { row.classList.add("in"); await wait(140); }
  for (const frame of framesEl()) { frame.classList.add("in"); await wait(110); }
  document.querySelectorAll("#a-stage .a-bubble").forEach((b) => b.classList.add("in"));
}

async function sceneTokens(wait) {
  const s = story();
  const row = rowsEl()[0];
  row.classList.add("focus");
  const box = row.querySelector(".a-tokens");
  box.replaceChildren();
  for (const piece of s.tokens.roberta.slice(0, 18)) {
    box.append(el("span", { className: `a-chip${piece.new_word ? "" : " joined"}`, textContent: piece.text }));
    await wait(120);
  }
  if (s.tokens.roberta.length > 18) box.append(el("span", { className: "a-more", textContent: `+${s.tokens.roberta.length - 18} more` }));
  await wait(900);
}

async function sceneNumbers(wait) {
  const s = story();
  rowsEl()[0].classList.remove("focus");
  rowsEl()[0].querySelector(".a-tokens").replaceChildren();
  for (const [i, row] of rowsEl().entries()) {
    const code = row.querySelector(".a-code");
    code.replaceChildren(...(s.barcodes[i] || []).map((v) => el("i", {
      style: `background:${v >= 0 ? MODEL : BASELINE};opacity:${(0.15 + 0.85 * Math.abs(v)).toFixed(2)}` })));
    row.classList.add("coded");
    await wait(260);
  }
}

async function sceneOrder(wait) {
  const rows = rowsEl();
  for (const direction of [rows, [...rows].reverse()]) {
    for (const row of direction) { row.classList.add("reading"); await wait(170); row.classList.remove("reading"); }
  }
  rows.forEach((row, i) => row.style.setProperty("--lead", i < 3 ? 1 : 0));
  document.querySelector("#a-stage .a-article").classList.add("lead");
  await wait(600);
}

function drawLine(lines, from, to, width, stage) {
  const a = from.getBoundingClientRect(), b = to.getBoundingClientRect(), box = stage.getBoundingClientRect();
  const x1 = a.right - box.left, y1 = a.top + a.height / 2 - box.top;
  const x2 = b.left - box.left, y2 = b.top + b.height / 2 - box.top;
  const path = svg("path", { d: `M${x1},${y1} C${x1 + 60},${y1} ${x2 - 60},${y2} ${x2},${y2}`, "stroke-width": width.toFixed(1) });
  lines.append(path);
  const length = path.getTotalLength();
  path.style.strokeDasharray = length;
  path.style.strokeDashoffset = REDUCED ? 0 : length;
  requestAnimationFrame(() => { path.style.strokeDashoffset = 0; });
}

async function sceneAttention(wait, fast) {
  const s = story();
  const stage = $("a-stage");
  const lines = stage.querySelector(".a-lines");
  lines.replaceChildren();
  lines.classList.toggle("instant", fast || REDUCED);
  const frames = framesEl();
  const ids = frames.map((f) => Number(f.dataset.i));
  const rows = rowsEl();
  const all = rows.flatMap((_, r) => ids.map((f) => s.sentence_frame_attention[r]?.[f] ?? 0));
  const top = Math.max(...all) || 1;
  for (const [r, row] of rows.entries()) {
    row.classList.add("looking");
    const weights = ids.map((f) => s.sentence_frame_attention[r]?.[f] ?? 0);
    const best = weights.map((w, k) => [w, k]).sort((x, y) => y[0] - x[0]).slice(0, 2);
    for (const [w, k] of best) {
      drawLine(lines, row, frames[k], 1 + 5 * (w / top), stage);
      frames[k].classList.add("hit");
    }
    await wait(380);
    row.classList.remove("looking");
    frames.forEach((f) => f.classList.remove("hit"));
  }
  await wait(500);
}

async function sceneScores(wait) {
  const s = story();
  $("a-stage").querySelector(".a-lines").replaceChildren();
  for (const [i, row] of rowsEl().entries()) {
    const score = s.scores[i] ?? 0;
    const bar = row.querySelector(".a-score i");
    bar.style.width = `${(100 * score).toFixed(0)}%`;
    row.querySelector(".a-score b").textContent = score.toFixed(2);
    row.classList.add("scored");
    if (s.chosen.includes(i)) row.classList.add("chosen");
    await wait(220);
  }
  await wait(500);
}

async function sceneSummary(wait, fast) {
  const s = story();
  const out = $("a-stage").querySelector(".a-out");
  const list = el("ol", { className: "a-summary" });
  out.replaceChildren(el("p", { className: "label", textContent: `Our summary: ${s.chosen.length} sentences` }), list);
  reveal(out, fast);
  for (const i of s.chosen) {
    list.append(el("li", { className: "pop" }, el("span", { className: "a-num", textContent: i + 1 }), s.sentences[i]));
    await wait(450);
  }
  const keyFrames = new Set(s.top_frames);
  framesEl().forEach((f) => f.classList.toggle("key", keyFrames.has(Number(f.dataset.i))));
  await wait(400);
}

async function sceneGrade(wait, fast) {
  const s = story();
  const out = $("a-stage").querySelector(".a-out");
  const ours = s.rouge.model.summary.flat();
  const theirs = s.rouge.model.reference;
  const words = (items) => items.map((w) => el("span", { className: w.hit ? "w hit" : "w", textContent: `${w.w} ` }));
  const meter = el("strong", { className: "a-meter", textContent: "0.0" });
  out.replaceChildren(
    el("div", { className: "a-grade" },
      el("div", {}, el("p", { className: "label", textContent: "Our summary" }), el("p", { className: "a-words" }, words(ours))),
      el("div", {}, el("p", { className: "label", textContent: "Journalist's summary" }), el("p", { className: "a-words" }, words(theirs)))),
    el("p", { className: "a-score-line" }, "ROUGE-1 for this story: ", meter,
      el("span", { textContent: `  (first 3 sentences would get ${s.rouge.lead3.r1.toFixed(1)})` })));
  reveal(out, fast);
  const hits = [...out.querySelectorAll(".w.hit")];
  for (const [k, word] of hits.entries()) {
    word.classList.add("lit");
    meter.textContent = ((s.rouge.model.r1 * (k + 1)) / hits.length).toFixed(1);
    await wait(Math.max(25, 1600 / hits.length));
  }
  meter.textContent = s.rouge.model.r1.toFixed(1);
  await wait(600);
}

async function sceneFindings(wait, fast) {
  const res = state.data.results;
  const find = (name) => res.main.find((m) => m.name.startsWith(name)) ?? res.ablations.find((a) => a.name === name);
  const rows = [["First 3 sentences (lead3)", find("Lead3").r1, BASELINE],
                ["Our hybrid (text + video + speech)", find("Full hybrid").r1, MODEL],
                ["Our hybrid with speech switched off", find("Speech switched off").r1, MODEL]];
  const out = $("a-stage").querySelector(".a-out");
  const bars = rows.map(([name, value, color]) => {
    const fill = el("i", { style: `background:${color}` });
    return { fill, value, node: el("div", { className: "a-big" }, el("span", { textContent: name }),
      el("span", { className: "a-track" }, fill), el("b", { textContent: value.toFixed(1) })) };
  });
  const low = 38, high = 42;
  out.replaceChildren(el("p", { className: "label", textContent: `ROUGE-1 on ${res.n} test articles (higher is better)` }),
    ...bars.map((b) => b.node),
    el("div", { className: "a-scale" }, el("span"), el("span", {}, ...[low, 39, 40, 41, high].map((v) => el("span", { textContent: v }))), el("span")),
    el("ul", { className: "a-takeaways" },
      el("li", { textContent: "Journalists put key facts first, so \"take the first 3 sentences\" is a very strong yardstick." }),
      el("li", { textContent: "The journalists' summaries are written from the article; the video's speech often covers only part of the story." }),
      el("li", { textContent: "So switching speech off actually helped. We report that honestly." })));
  reveal(out, fast);
  for (const b of bars) {
    b.fill.style.left = `${(100 * (b.value - low)) / (high - low)}%`;
    await wait(500);
  }
  out.append(el("p", { className: "hint", textContent: "The scale runs from 38 to 42 so the small differences are visible. Differences under about half a point can be luck." }));
}

// Shows the result box and, while animating, scrolls it into view under the pinned caption.
function reveal(out, fast) {
  out.hidden = false;
  if (!fast && !REDUCED) out.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Player ------------------------------------------------------------------------------------------------------------
async function play(sceneIndex) {
  const run = ++state.run;
  state.scene = sceneIndex;
  buildStage();
  renderChrome();
  try {
    for (let k = 0; k < sceneIndex; k++) await SCENES[k].run(timer(run, true), true);
    for (let k = sceneIndex; k < SCENES.length; k++) {
      state.scene = k;
      renderChrome();
      await SCENES[k].run(timer(run, false), false);
      if (k < SCENES.length - 1) await timer(run, false)(1400);
      if (!state.playing) { await waitForPlay(run); }
    }
    state.playing = false;
    renderChrome();
  } catch (error) {
    if (!(error instanceof Cancelled)) throw error;
  }
}

async function waitForPlay(run) {
  while (!state.playing) {
    await new Promise((r) => setTimeout(r, 80));
    if (run !== state.run) throw new Cancelled();
  }
}

function renderChrome() {
  const scene = SCENES[state.scene];
  $("a-concept").textContent = scene.concept;
  $("a-caption").textContent = scene.caption;
  $("a-link").replaceChildren(...(scene.link ? ["Learn more: ", el("a", { href: LINKS[scene.link][1], target: "_blank", rel: "noopener", textContent: LINKS[scene.link][0] })] : []));
  $("a-play").textContent = state.playing ? "❚❚ Pause" : state.scene === SCENES.length - 1 ? "↺ Replay" : "▶ Play";
  $("a-dots").querySelectorAll("button").forEach((b, i) => {
    b.classList.toggle("on", i === state.scene);
    b.classList.toggle("done", i < state.scene);
    b.setAttribute("aria-current", i === state.scene ? "step" : "false");
  });
}

async function open() {
  if (!state.data) {
    const response = await fetch("explainer/data.json");
    if (!response.ok) throw new Error(`explainer data: ${response.status}`);
    state.data = await response.json();
    $("a-story").replaceChildren(...state.data.stories.map((s, i) => el("option", { value: i, textContent: short(s.title, 110) })));
    $("a-dots").replaceChildren(...SCENES.map((s, i) => {
      const b = el("button", { type: "button", title: s.concept });
      b.setAttribute("aria-label", `Go to: ${s.concept}`);
      b.addEventListener("click", () => { state.playing = true; play(i); });
      return b;
    }));
  }
  state.playing = !REDUCED;
  play(0);
}

$("a-play").addEventListener("click", () => {
  if (!state.playing && state.scene === SCENES.length - 1) { state.playing = true; play(0); return; }
  state.playing = !state.playing;
  renderChrome();
});
$("a-prev").addEventListener("click", () => { state.playing = true; play(Math.max(0, state.scene - 1)); });
$("a-next").addEventListener("click", () => { state.playing = true; play(Math.min(SCENES.length - 1, state.scene + 1)); });
$("a-story").addEventListener("change", (e) => { state.story = Number(e.target.value); state.playing = true; play(0); });
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.data && !$("explain-panel").hidden) play(state.scene); }, 250);
});
document.querySelector('.nav[data-mode="explain"]').addEventListener("click", () => {
  open().catch((error) => { $("a-caption").textContent = `Couldn't load the walk-through (${error.message}).`; });
});
