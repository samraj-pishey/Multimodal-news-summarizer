import * as pdfjs from "./vendor/pdf.min.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdf.worker.min.mjs", import.meta.url).href;

const MAX_VIDEO_SECONDS = 600;
const MAX_AUDIO_FILE_BYTES = 500 * 2 ** 20;
const BATCH_SIZE = 8;
const FRAMES_SHOWN = 4;
const MIN_SENTENCE_WORDS = 4;
const MIN_SPEECH_SENTENCE_WORDS = 3;     // these two match m2sm_data.py, which the speech model trained on
const MIN_SPEECH_WORDS = 5;
const AUDIO_RATE = 16000;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const ABBREVIATION = /(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|Gen|Col|Lt|St|Mt|Jr|Sr|No|vs|etc|e\.g|i\.e|a\.m|p\.m|a\.k\.a|approx|Inc|Ltd|Corp|Co|Dept|Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec)|\b[A-Z](?:\.[A-Z])*)\.$/;
const UPLOAD_STEPS = ["Loading models", "Reading the PDF", "Sampling video frames", "Transcribing the audio", "Selecting sentences and frames"];
const ARTICLE_STEPS = ["Loading the model", "Reading the PDF", "Selecting sentences"];
const VIDEO_STEPS = ["Loading models", "Sampling video frames", "Transcribing the audio", "Selecting sentences and frames"];
const SAMPLE_STEPS = ["Loading the model", "Reading the video features", "Selecting sentences and frames"];

const $ = (id) => document.getElementById(id);
const state = { result: null, length: 4 };

// Worker calls --------------------------------------------------------------------------------------------------
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 0;
let modelsLoaded = null;

worker.onmessage = ({ data }) => {
  const call = pending.get(data.id);
  if ("progress" in data) return call.onProgress?.(data.progress);
  pending.delete(data.id);
  data.error ? call.reject(new Error(data.error)) : call.resolve(data.result);
};
worker.onerror = (event) => {
  for (const call of pending.values()) call.reject(new Error(event.message || "The model worker failed to start."));
  pending.clear();
};

function ask(type, payload = {}, { transfer = [], onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

// Progress --------------------------------------------------------------------------------------------------------
function setSteps(names) {
  $("steps").replaceChildren(...names.map((name) => {
    const item = document.createElement("li");
    item.textContent = name;
    item.dataset.name = name;
    return item;
  }));
  $("steps").hidden = false;
}

function setStep(index, detail = "") {
  [...$("steps").children].forEach((item, i) => {
    item.className = i < index ? "done" : i === index ? "active" : "";
    item.textContent = item.dataset.name + (i === index && detail ? `: ${detail}` : "");
  });
}

// PDF -------------------------------------------------------------------------------------------------------------
async function pdfLines(file) {
  const document_ = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let n = 1; n <= document_.numPages; n++) {
    const { items } = await (await document_.getPage(n)).getTextContent();
    const lines = [];
    let line = "";
    for (const item of items) {
      line += item.str ?? "";
      if (item.hasEOL) {
        lines.push(line.trim());
        line = "";
      }
    }
    lines.push(line.trim());
    pages.push(lines.filter(Boolean));
  }
  return pages;
}

// Lines on most pages (running headers and footers) and bare page numbers aren't article text.
function bodyLines(pages) {
  const counts = new Map();
  const key = (line) => line.toLowerCase().replace(/\d+/g, "#");
  for (const page of pages) for (const line of new Set(page.map(key))) counts.set(line, (counts.get(line) ?? 0) + 1);
  const repeated = (line) => pages.length > 1 && counts.get(key(line)) >= Math.max(2, pages.length / 2);
  return pages.flat().filter((line) => !repeated(line) && !/^(?:page\s*)?\d+(?:\s*(?:of|\/)\s*\d+)?$/i.test(line));
}

// A short line with no closing punctuation, standing between paragraphs, is a headline or byline, not a sentence.
function dropHeadings(lines) {
  const ends = /[.!?:;"”’)]$/;
  let previousWasHeading = false;
  return lines.filter((line, i) => {
    const short = line.split(/\s+/).length <= 12;
    const startsParagraph = i === 0 || ends.test(lines[i - 1]) || previousWasHeading;
    const nextStartsNew = i + 1 >= lines.length || /^[A-Z0-9"“‘]/.test(lines[i + 1]);
    previousWasHeading = short && !ends.test(line) && startsParagraph && nextStartsNew;
    return !previousWasHeading;
  });
}

function splitSentences(lines) {
  const text = dropHeadings(lines).join("\n").replace(/([a-z])-\n([a-z])/g, "$1$2").replace(/\s+/g, " ")
    .replace(/([a-z0-9)'"”’][.!?])([A-Z])/g, "$1 $2").trim();      // scraped articles glue sentences: "sideline.Feki"
  const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
  const merged = [];
  for (const { segment } of segmenter.segment(text)) {
    const sentence = segment.trim();
    if (merged.length && ABBREVIATION.test(merged[merged.length - 1])) merged[merged.length - 1] += ` ${sentence}`;
    else if (sentence) merged.push(sentence);
  }
  return merged.filter((sentence) => sentence.split(/\s+/).length >= MIN_SENTENCE_WORDS);
}

async function readArticle(file) {
  let pages;
  try {
    pages = await pdfLines(file);
  } catch (error) {
    throw new Error(`Couldn't read the PDF (${error.message}). Is it password protected?`);
  }
  const sentences = splitSentences(bodyLines(pages));
  if (sentences.length < 2) {
    throw new Error("No article text found in this PDF. Scanned PDFs have no selectable text; try one exported from a web page or a word processor.");
  }
  return sentences;
}

// Video -----------------------------------------------------------------------------------------------------------
function once(target, event, failure = "error") {
  return new Promise((resolve, reject) => {
    const done = (handler) => (value) => {
      target.removeEventListener(event, onEvent);
      target.removeEventListener(failure, onFailure);
      handler(value);
    };
    const onEvent = done(resolve);
    const onFailure = done(() => reject(new Error("This browser can't decode the video. H.264 MP4 works in every major browser.")));
    target.addEventListener(event, onEvent);
    target.addEventListener(failure, onFailure);
  });
}

async function seek(video, time) {
  const seeked = once(video, "seeked");
  video.currentTime = time;
  await seeked;
}

// Resize the short side to 256 and center-crop 224, as ResNet50_Weights.IMAGENET1K_V1.transforms() does.
function preparePixels(video, context) {
  const { videoWidth: width, videoHeight: height } = video;
  const crop = 224 / (256 / Math.min(width, height));
  context.drawImage(video, (width - crop) / 2, (height - crop) / 2, crop, crop, 0, 0, 224, 224);
  const { data } = context.getImageData(0, 0, 224, 224);
  const pixels = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    for (let c = 0; c < 3; c++) pixels[c * 224 * 224 + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return pixels;
}

async function sampleFrames(file, step) {
  const video = $("player");
  URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(file);
  await once(video, "loadedmetadata");
  if (!video.videoWidth || !Number.isFinite(video.duration)) {
    throw new Error("This file has no video track this browser can decode. H.264 MP4 works in every major browser.");
  }
  const canvas = Object.assign(document.createElement("canvas"), { width: 224, height: 224 });
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";

  const seconds = Math.min(video.duration, MAX_VIDEO_SECONDS);
  const times = [];
  for (let t = 0; t < seconds; t += 1) times.push(t);
  await ask("reset");
  for (let start = 0; start < times.length; start += BATCH_SIZE) {
    const batch = times.slice(start, start + BATCH_SIZE);
    const pixels = new Float32Array(batch.length * 3 * 224 * 224);
    for (let i = 0; i < batch.length; i++) {
      await seek(video, batch[i]);
      pixels.set(preparePixels(video, context), i * 3 * 224 * 224);
    }
    await ask("encodeFrames", { pixels, count: batch.length }, { transfer: [pixels.buffer] });
    step(`${Math.min(start + BATCH_SIZE, times.length)} of ${times.length} seconds`);
  }
  return { times, truncated: video.duration > MAX_VIDEO_SECONDS };
}

async function thumbnail(video, time) {
  await seek(video, time);
  const scale = Math.min(1, 480 / video.videoWidth);
  const canvas = Object.assign(document.createElement("canvas"), {
    width: Math.round(video.videoWidth * scale), height: Math.round(video.videoHeight * scale),
  });
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

// Mono at 16 kHz, what Whisper expects. Throws if the file has no audio track.
async function readAudio(file) {
  const context = new OfflineAudioContext(1, AUDIO_RATE, AUDIO_RATE);
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  const samples = new Float32Array(Math.min(decoded.length, MAX_VIDEO_SECONDS * AUDIO_RATE));
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const channel = decoded.getChannelData(c);
    for (let i = 0; i < samples.length; i++) samples[i] += channel[i] / decoded.numberOfChannels;
  }
  return samples;
}

const NO_SPEECH = { text: "", segments: [] };

// Whisper's text plus its timed segments ({ start, end, text } in seconds).
async function transcribeVideo(file, step, notes) {
  if (file.size > MAX_AUDIO_FILE_BYTES) {
    notes.push("The video is over 500 MB, so its audio was skipped and the model used the frames alone.");
    return NO_SPEECH;
  }
  let audio;
  try {
    audio = await readAudio(file);
  } catch {
    notes.push("No audio could be read from the video, so the model used the frames alone.");
    return NO_SPEECH;
  }
  try {
    const onProgress = (percent) => step(percent === null ? "running" : `downloading Whisper ${percent}%`);
    const speech = await ask("transcribe", { audio }, { transfer: [audio.buffer], onProgress });
    if (!speech.text) notes.push("No speech was found in the video, so the model used the frames alone.");
    return speech;
  } catch (error) {
    notes.push(`Transcription failed (${error.message}), so the model used the frames alone.`);
    return NO_SPEECH;
  }
}

// Same rule as speech_sentences() in m2sm_data.py.
function speechSentences(text) {
  return text.split(/(?<=[.!?])\s+/).map((sentence) => sentence.trim())
    .filter((sentence) => sentence.split(/\s+/).length >= MIN_SPEECH_SENTENCE_WORDS);
}

// Results ---------------------------------------------------------------------------------------------------------
function topIndices(values, count) {
  return values.map((value, i) => [value, i]).sort((a, b) => b[0] - a[0]).slice(0, count).map(([, i]) => i);
}

const sigmoid = (logit) => 1 / (1 + Math.exp(-logit));
const tidy = (text) => text.replace(/([a-z0-9)'"”’][.!?])([A-Z])/g, "$1 $2");

// Bars are scaled to the story's own top score, so they compare within one story only.
function renderStrip(strip, readout, logits, chosen, describe) {
  const values = logits.map(sigmoid);
  const top = Math.max(...values) || 1;
  strip.replaceChildren(...values.map((value, i) => {
    const button = Object.assign(document.createElement("button"), { type: "button", className: chosen.has(i) ? "on" : "" });
    button.setAttribute("aria-label", describe(i, value));
    const bar = document.createElement("i");
    bar.style.height = `${Math.max(4, (100 * value) / top)}%`;
    button.append(bar);
    const show = () => { readout.textContent = describe(i, value); };
    button.addEventListener("mouseenter", show);
    button.addEventListener("focus", show);
    return button;
  }));
}

function renderSummary() {
  const { sentences, sentenceLogits, rouge, source = "article" } = state.result;
  const count = Math.min(state.length, sentences.length);
  const picks = topIndices(sentenceLogits, count).sort((a, b) => a - b);
  const chosen = new Set(picks);
  $("length-value").textContent = count;
  $("summary").replaceChildren(...picks.map((i) => Object.assign(document.createElement("li"), { textContent: tidy(sentences[i]) })));

  const outside = picks.filter((value) => value >= count).length;
  $("lead-note").textContent = outside === 0
    ? `For this story the model chose the ${source}'s first ${count === 1 ? "sentence" : `${count} sentences`}.`
    : `Kept in ${source} order. ${outside} of ${count} come from outside the ${source}'s first ${count}.`;

  renderStrip($("sentence-strip"), $("sentence-readout"), sentenceLogits, chosen,
    (i, value) => `Sentence ${i + 1}, score ${value.toFixed(2)}: ${tidy(sentences[i])}`);
  $("strip-end").textContent = `Sentence ${sentences.length}`;

  const words = (list) => list.reduce((total, text) => total + text.split(/\s+/).filter(Boolean).length, 0);
  const kept = words(picks.map((i) => sentences[i]));
  const total = words(sentences);
  const chip = (text) => Object.assign(document.createElement("span"), { textContent: text });
  $("summary-stats").replaceChildren(
    chip(`${count} of ${sentences.length} sentences`),
    chip(`${kept} of ${total} words`),
    chip(`${Math.max(0, Math.round(100 * (1 - kept / Math.max(1, total))))}% shorter`));

  if (rouge) {
    const row = (name, scores) => {
      const tr = document.createElement("tr");
      tr.append(...[name, scores.r1.toFixed(1), scores.r2.toFixed(1), scores.rl.toFixed(1)].map(
        (text) => Object.assign(document.createElement("td"), { textContent: text })));
      return tr;
    };
    $("rouge").replaceChildren(row("Model", rouge.model[count]), row(`First ${count} sentence${count === 1 ? "" : "s"}`, rouge.lead[count]));
    $("rouge-note").textContent = outside === 0
      ? `Here the model picked exactly the first ${count === 1 ? "sentence" : `${count} sentences`}, so both rows match. News puts key facts first, which makes this baseline hard to beat.`
      : "";
  }

  $("article").replaceChildren(...sentences.flatMap((sentence, i) => {
    const part = document.createElement(chosen.has(i) ? "mark" : "span");
    part.textContent = tidy(sentence);
    return [part, document.createTextNode(" ")];
  }));
}

function formatTime(seconds) {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function showResult(result) {
  state.result = result;
  window.m2sm = result;
  state.length = Math.min(result.defaultLength, result.sentences.length);
  const hasText = result.sentences.length > 0;
  const fromSpeech = result.source === "speech";
  $("result-title").textContent = result.title ?? (hasText ? "Your summary" : "Your key frames");
  $("text-results").hidden = !hasText;
  $("length-control").hidden = !hasText;
  $("frames-block").hidden = result.images.length === 0;
  $("sentence-hint").textContent = `One bar per sentence, in ${fromSpeech ? "the order spoken" : "article order"}. Hover a bar to read it.`;
  $("full-text-title").textContent = fromSpeech ? "Full transcript, with selected sentences marked" : "Full article, with selected sentences marked";
  if (hasText) {
    $("length").max = Math.min(8, result.sentences.length);
    $("length").value = state.length;
    renderSummary();
  }

  $("compare").classList.toggle("solo", !result.human);
  $("human-column").hidden = !result.human;
  $("rouge-block").hidden = !result.rouge;
  if (result.human) {
    $("human").replaceChildren(...result.human.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
  }

  $("frames").replaceChildren(...result.images.map(({ time, src }) => {
    const figure = document.createElement("figure");
    const image = Object.assign(document.createElement("img"), { src, alt: `Video frame at ${formatTime(time)}` });
    figure.append(image, Object.assign(document.createElement("figcaption"), { textContent: formatTime(time) }));
    return figure;
  }));

  $("transcript").textContent = result.transcript;
  renderSpeech(result);
  $("transcript-block").hidden = !result.transcript || fromSpeech || !$("speech-block").hidden;
  const inputs = result.inputs ?? ["article"];
  $("inputs-used").replaceChildren(Object.assign(document.createElement("span"), { textContent: `Read: ${inputs.join(" · ")}` }));
  resetAbstract(result);
  document.querySelectorAll(".jump a[data-for]").forEach((link) => { link.hidden = $(link.dataset.for).hidden; });
  $("welcome").hidden = true;
  $("results").hidden = false;
}

// What was said: the speech as timed lines. Clicking a line shows that moment of the video: a fresh frame for
// uploads, or the nearest saved frame for sample stories (which keep only a few).
function renderSpeech(result) {
  const segments = result.speech ?? [];
  const block = $("speech-block");
  block.hidden = !segments.length && !result.transcript;
  $("speech-preview").hidden = true;
  if (block.hidden) return;
  $("speech-hint").textContent = segments.length
    ? (result.speechNote ?? "Written out by Whisper in your browser.") + " Click a line to see that moment of the video."
    : "The speech for this story has no timestamps, so it is shown as plain text.";
  if (!segments.length) {
    $("speech-list").replaceChildren(Object.assign(document.createElement("li"), { className: "plain", textContent: result.transcript }));
    return;
  }
  $("speech-list").replaceChildren(...segments.map((segment) => {
    const item = document.createElement("li");
    const button = Object.assign(document.createElement("button"), { type: "button" });
    button.append(Object.assign(document.createElement("span"), { className: "time", textContent: `${formatTime(segment.start)}–${formatTime(segment.end)}` }),
      Object.assign(document.createElement("span"), { textContent: segment.text }));
    button.addEventListener("click", () => previewMoment(result, segment, item));
    item.append(button);
    return item;
  }));
}

async function previewMoment(result, segment, item) {
  document.querySelectorAll("#speech-list li").forEach((li) => li.classList.toggle("on", li === item));
  const figure = $("speech-preview");
  const image = figure.querySelector("img");
  const caption = figure.querySelector("figcaption");
  if (result.live) {
    image.src = await thumbnail($("player"), segment.start);
    caption.textContent = `The video at ${formatTime(segment.start)}`;
  } else {
    const frames = result.previewFrames ?? result.images;
    if (!frames.length) return;
    const nearest = frames.reduce((best, frame) => (Math.abs(frame.time - segment.start) < Math.abs(best.time - segment.start) ? frame : best));
    image.src = nearest.src;
    caption.textContent = `Nearest saved frame: ${formatTime(nearest.time)}`;
  }
  image.alt = caption.textContent;
  figure.hidden = false;
}

// Abstractive summary: hidden until the user asks for it. Sample stories carry a BART-large-CNN summary made
// offline; uploads run the smaller DistilBART in the worker, downloaded on first use.
const MAX_ABSTRACT_WORDS = 700;       // stays under BART's 1,024-token input limit

function resetAbstract(result) {
  const button = $("ai-button");
  $("ai-column").hidden = true;
  button.hidden = false;
  button.disabled = false;
  button.textContent = "✨ Write a condensed summary with AI";
  $("ai-hint").textContent = result.abstractive
    ? "Shows a summary written by BART-large-CNN, a pretrained abstractive model."
    : "Runs DistilBART, a smaller abstractive model, in your browser. The first time it downloads about 270 MB; after that it's cached.";
}

function showAbstract(text, model, scores) {
  $("ai-model").textContent = model;
  $("ai-text").textContent = text;
  const count = Math.min(state.length, state.result.sentences.length);
  $("ai-score").textContent = scores && state.result.rouge
    ? `ROUGE-1 against the human summary: ${scores.r1.toFixed(1)} (the selected sentences above: ${state.result.rouge.model[count].r1.toFixed(1)}).`
    : "";
  $("ai-button").hidden = true;
  $("ai-hint").textContent = "";
  $("ai-column").hidden = false;
}

async function writeAbstract() {
  const result = state.result;
  const button = $("ai-button");
  if (result.abstractive) {
    showAbstract(result.abstractive.text, result.abstractive.model, result.abstractive);
    return;
  }
  button.disabled = true;
  button.textContent = "Preparing…";
  try {
    const text = result.sentences.join(" ").split(/\s+/).slice(0, MAX_ABSTRACT_WORDS).join(" ");
    const summary = await ask("abstract", { text }, { onProgress: (percent) => {
      button.textContent = percent === null ? "Writing the summary…" : `Downloading the AI model ${percent}%`;
    } });
    if (state.result === result) showAbstract(summary, "DistilBART, running in your browser");
  } catch (error) {
    if (state.result !== result) return;
    button.disabled = false;
    button.textContent = "✨ Try again";
    $("ai-hint").textContent = `The AI summary couldn't be written (${error.message}).`;
  }
}

$("ai-button").addEventListener("click", writeAbstract);

// Upload drop zones: the real file input covers each zone, so clicking and dropping both work natively.
document.querySelectorAll(".drop").forEach((zone) => {
  const input = zone.querySelector("input");
  const label = zone.querySelector(".drop-file");
  const show = () => {
    const file = input.files[0];
    zone.classList.toggle("has-file", Boolean(file));
    const size = file && (file.size < 2 ** 20 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / 2 ** 20).toFixed(1)} MB`);
    label.textContent = file ? `${file.name} · ${size}` : zone.dataset.empty;
  };
  input.addEventListener("change", show);
  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, () => zone.classList.add("over")));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("over")));
});

function showNotes(notes) {
  $("notes").replaceChildren(...notes.map((note) => Object.assign(document.createElement("li"), { textContent: note })));
  $("notes").hidden = notes.length === 0;
}

// Flow ------------------------------------------------------------------------------------------------------------
async function loadModels(variant) {
  const config = await (modelsLoaded ??= ask("load"));
  await ask("loadVariant", { variant });
  return config;
}

async function readLimitedArticle(pdf, config, notes) {
  const sentences = await readArticle(pdf);
  if (sentences.length <= config.max_sentences) return sentences;
  notes.push(`The article has ${sentences.length} sentences; the model read the first ${config.max_sentences}.`);
  return sentences.slice(0, config.max_sentences);
}

async function runArticleOnly(pdf) {
  const notes = ["No video was uploaded, so this summary comes from the article text alone and has no key frames."];
  setSteps(ARTICLE_STEPS);
  setStep(0);
  const config = await loadModels("text");

  setStep(1);
  const sentences = await readLimitedArticle(pdf, config, notes);

  setStep(2);
  const output = await ask("summarize", { sentences, transcript: "", variant: "text" });
  showResult({ ...output, sentences, transcript: "", times: [], images: [], frameIndices: [],
               defaultLength: config.modes.text.summary_size });
  showNotes(notes);
  setStep(ARTICLE_STEPS.length);
}

async function runVideoOnly(videoFile) {
  const notes = [];
  setSteps(VIDEO_STEPS);
  setStep(0);
  const config = await loadModels("speech");
  await ask("loadVision");

  setStep(1);
  const { times, truncated } = await sampleFrames(videoFile, (detail) => setStep(1, detail));
  if (times.length === 0) throw new Error("The video has no frames to read.");
  if (truncated) notes.push("The video is over 10 minutes; only the first 10 were used.");

  setStep(2);
  const { text: transcript, segments: speech } = await transcribeVideo(videoFile, (detail) => setStep(2, detail), notes);
  const spoken = speechSentences(transcript).slice(0, config.max_sentences);
  const words = spoken.reduce((total, sentence) => total + sentence.split(/\s+/).length, 0);
  const sentences = words >= MIN_SPEECH_WORDS ? spoken : [];

  setStep(3);
  let sentenceLogits = [];
  if (sentences.length) {
    ({ sentenceLogits } = await ask("summarize", { sentences, transcript, variant: "speech" }));
    notes.push("No article was uploaded, so this summary comes only from what is said in the video. It is much less reliable than a summary made from an article, so check it against the video.");
  } else if (transcript) {
    notes.push("The speech in the video was too short to summarize, so only key frames are shown.");
  }
  const frameIndices = await ask("vsumm", { count: FRAMES_SHOWN });
  const images = [];
  for (const i of frameIndices) images.push({ time: times[i], src: await thumbnail($("player"), times[i]) });
  showResult({ sentenceLogits, frameLogits: [], sentences, transcript, speech, live: true, times, images, frameIndices,
               source: "speech", inputs: ["video", ...(transcript ? ["speech"] : [])],
               defaultLength: config.modes.speech.summary_size });
  showNotes(notes);
  setStep(VIDEO_STEPS.length);
}

async function runUpload(pdf, videoFile) {
  if (!pdf && !videoFile) throw new Error("Choose an article PDF, a video, or both.");
  if (!videoFile) return runArticleOnly(pdf);
  if (!pdf) return runVideoOnly(videoFile);

  const notes = [];
  setSteps(UPLOAD_STEPS);
  setStep(0);
  const config = await loadModels("full");
  await ask("loadVision");

  setStep(1);
  const sentences = await readLimitedArticle(pdf, config, notes);

  setStep(2);
  const { times, truncated } = await sampleFrames(videoFile, (detail) => setStep(2, detail));
  if (times.length === 0) throw new Error("The video has no frames to read.");
  if (truncated) notes.push("The video is over 10 minutes; only the first 10 were used.");

  setStep(3);
  const { text: transcript, segments: speech } = await transcribeVideo(videoFile, (detail) => setStep(3, detail), notes);

  setStep(4);
  const output = await ask("summarize", { sentences, transcript });
  const frameIndices = topIndices(output.frameLogits, FRAMES_SHOWN).sort((a, b) => a - b);
  const images = [];
  for (const i of frameIndices) images.push({ time: times[i], src: await thumbnail($("player"), times[i]) });
  showResult({ ...output, sentences, transcript, speech, live: true, times, images, frameIndices,
               inputs: ["article", "video", ...(output.usedTranscript ? ["speech"] : [])], defaultLength: config.summary_size });
  showNotes(notes);
  setStep(UPLOAD_STEPS.length);
}

async function runSample(sample) {
  setSteps(SAMPLE_STEPS);
  setStep(0);
  const config = await loadModels("full");
  setStep(1);
  const response = await fetch(`samples/${sample.number}.bin`);
  if (!response.ok) throw new Error(`Couldn't download the story's video features (${response.status}).`);
  const features = new Float32Array(await response.arrayBuffer());
  await ask("setFrames", { features }, { transfer: [features.buffer] });
  setStep(2);
  const output = await ask("summarize", { sentences: sample.sentences, transcript: sample.transcript });

  const byTime = new Map(sample.frames.map((frame) => [frame.time, frame.file]));
  const frameIndices = topIndices(output.frameLogits, FRAMES_SHOWN + 2)
    .filter((i) => byTime.has(sample.times[i])).slice(0, FRAMES_SHOWN).sort((a, b) => a - b);
  const images = frameIndices.map((i) => ({ time: sample.times[i], src: `samples/${byTime.get(sample.times[i])}` }));
  showResult({ ...output, title: sample.title, sentences: sample.sentences, transcript: sample.transcript,
               times: sample.times, images, frameIndices, human: sample.human, rouge: sample.rouge, abstractive: sample.abstractive,
               speech: sample.speech, inputs: ["article", "video", ...(output.usedTranscript ? ["speech"] : [])],
               previewFrames: sample.frames.map((frame) => ({ time: frame.time, src: `samples/${frame.file}` })),
               speechNote: "Times come from Whisper. For sample stories the model read the transcript supplied with the dataset.",
               defaultLength: config.summary_size });
  setStep(SAMPLE_STEPS.length);
}

async function guarded(button, work) {
  $("error").hidden = true;
  $("notes").hidden = true;
  $("results").hidden = true;
  button.disabled = true;
  document.body.dataset.state = "running";
  try {
    await work();
    document.body.dataset.state = "done";
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
    document.body.dataset.state = "failed";
  } finally {
    button.disabled = false;
  }
}

function setMode(mode) {
  document.querySelectorAll(".nav").forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  $("samples-panel").hidden = mode !== "samples";
  $("upload-panel").hidden = mode !== "upload";
  $("explain-panel").hidden = mode !== "explain";
  $("welcome").hidden = mode === "explain";
  $("results").hidden = true;
  $("error").hidden = true;
  $("notes").hidden = true;
  $("steps").hidden = true;
}

function describeSample(sample) {
  const seconds = sample.times.at(-1) ?? 0;
  $("sample-meta").textContent = `${sample.sentences.length} sentences · ${formatTime(seconds)} of video · ${sample.transcript.split(/\s+/).filter(Boolean).length} transcript words · Story ${sample.id}`;
}

async function start() {
  const library = await fetch("samples/samples.json").then((r) => r.json());
  state.samples = library.samples;
  $("sample").replaceChildren(...library.samples.map((sample, i) => Object.assign(document.createElement("option"), {
    value: i, textContent: `${i + 1}. ${sample.title}`,
  })));
  describeSample(library.samples[0]);
}

$("length").addEventListener("input", (event) => {
  state.length = Number(event.target.value);
  renderSummary();
});
$("sample").addEventListener("change", (event) => {
  describeSample(state.samples[event.target.value]);
  $("results").hidden = true;
  $("steps").hidden = true;
});
document.querySelectorAll(".nav").forEach((item) => item.addEventListener("click", () => setMode(item.dataset.mode)));
$("sample-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("generate"), () => runSample(state.samples[$("sample").value]));
});
$("form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("run"), () => runUpload($("pdf").files[0], $("video").files[0]));
});
start().catch((error) => {
  $("error").textContent = `Couldn't load the app (${error.message}).`;
  $("error").hidden = false;
});
