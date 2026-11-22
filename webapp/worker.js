// Runs ResNet-50, M2SM and Whisper off the main thread. Nothing leaves the browser except model downloads.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { vsummFrames } from "./vsumm.js";

const WHISPER_MODEL = "Xenova/whisper-tiny.en";
const ABSTRACT_MODEL = "Xenova/distilbart-cnn-6-6";
let abstracter = null;
const UNK = 1;

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency - 1)) : 1;

let config, vocabulary, resnet;
let frames = [];
const variants = {};

const session = (name) => ort.InferenceSession.create(`models/${name}`, { executionProviders: ["wasm"] });

async function fetchChecked(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Couldn't download ${path} (${response.status})`);
  return response;
}

async function load() {
  [config, vocabulary] = await Promise.all([
    fetchChecked("models/config.json").then((r) => r.json()),
    fetchChecked("models/vocabulary.json").then((r) => r.json()),
  ]);
  return config;
}

// "full" reads article and video, "text" the article alone, "speech" the video's speech and frames.
// Each trained its own word vectors, and only the one a run needs is downloaded.
async function loadVariant({ variant }) {
  const folder = variant === "full" ? "" : `${variant}/`;
  const names = variant === "full" ? ["summarizer_transcript.onnx", "summarizer_video.onnx"] : ["summarizer.onnx"];
  variants[variant] ??= await Promise.all([
    fetchChecked(`models/${folder}embeddings.bin`).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
    session(`${folder}word_encoder.onnx`),
    ...names.map((name) => session(folder + name)),
  ]).then(([embeddings, wordEncoder, ...summarizers]) => ({ embeddings, wordEncoder, summarizers }));
}

// Only uploads need ResNet-50: the samples come with their frame features.
async function loadVision() {
  resnet ??= await session("resnet.onnx");
}

async function encodeFrames(pixels, count) {
  const output = await resnet.run({ images: new ort.Tensor("float32", pixels, [count, 3, 224, 224]) });
  frames.push(output.features.data.slice());
}

// Same rule as tokenize() in m2sm_data.py.
function encode(text, limit) {
  const tokens = text.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]/g) ?? [];
  const ids = tokens.map((token) => (Object.hasOwn(vocabulary, token) ? vocabulary[token] : UNK)).slice(0, limit);
  return ids.length ? ids : [UNK];
}

function embed(ids, embeddings) {
  const size = config.embedding_dim;
  const out = new Float32Array(ids.length * size);
  ids.forEach((id, i) => out.set(embeddings.subarray(id * size, (id + 1) * size), i * size));
  return out;
}

function allFrames() {
  const all = new Float32Array(frames.reduce((total, part) => total + part.length, 0));
  frames.reduce((offset, part) => (all.set(part, offset), offset + part.length), 0);
  return all;
}

async function summarize(sentences, transcript, variant = "full") {
  const size = config.embedding_dim;
  const { embeddings, wordEncoder, summarizers } = variants[variant];
  const pooled = new Float32Array(sentences.length * 128);
  for (let i = 0; i < sentences.length; i++) {
    const ids = encode(sentences[i], config.max_sentence_words);
    const input = new ort.Tensor("float32", embed(ids, embeddings), [1, ids.length, size]);
    pooled.set((await wordEncoder.run({ embedded: input })).pooled.data, i * 128);
  }
  const inputs = { pooled: new ort.Tensor("float32", pooled, [sentences.length, 128]) };
  if (variant === "text") {
    const result = await summarizers[0].run(inputs);
    return { sentenceLogits: Array.from(result.sentence_logits.data), frameLogits: [], usedTranscript: false };
  }

  const all = allFrames();
  inputs.frames = new ort.Tensor("float32", all, [all.length / 2048, 2048]);
  if (variant === "speech") {
    const result = await summarizers[0].run(inputs);
    return { sentenceLogits: Array.from(result.sentence_logits.data), frameLogits: [], usedTranscript: false };
  }
  const useWords = Boolean(transcript.trim());
  if (useWords) {
    const ids = encode(transcript, config.max_transcript_words);
    inputs.words = new ort.Tensor("float32", embed(ids, embeddings), [ids.length, size]);
  }
  const result = await summarizers[useWords ? 0 : 1].run(inputs);
  return { sentenceLogits: Array.from(result.sentence_logits.data), frameLogits: Array.from(result.frame_logits.data),
           usedTranscript: useWords, features: all };
}

async function transcribe(audio, report) {
  const { pipeline } = await import("./vendor/transformers.min.js");
  const recognizer = await pipeline("automatic-speech-recognition", WHISPER_MODEL, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (event) => {
      if (event.status === "progress" && event.file?.endsWith(".onnx")) report(Math.round(event.progress));
    },
  });
  report(null);
  const output = await recognizer(audio, { chunk_length_s: 30, stride_length_s: 5, return_timestamps: true });
  await recognizer.dispose();
  // Each chunk is a stretch of speech with its [start, end] time in seconds.
  const segments = (output.chunks ?? []).map((chunk) => ({
    start: chunk.timestamp?.[0] ?? 0, end: chunk.timestamp?.[1] ?? chunk.timestamp?.[0] ?? 0, text: chunk.text.trim(),
  })).filter((segment) => segment.text);
  return { text: output.text.trim(), segments };
}

// Abstractive summary with DistilBART (8-bit), loaded once on first use. Settings follow the greedy decoding
// chosen for BART-large-CNN on validation stories.
async function abstract(text, report) {
  if (!abstracter) {
    const { pipeline } = await import("./vendor/transformers.min.js");
    abstracter = await pipeline("summarization", ABSTRACT_MODEL, {
      dtype: "q8",
      device: "wasm",
      progress_callback: (event) => {
        if (event.status === "progress" && event.file?.endsWith(".onnx")) report(Math.round(event.progress));
      },
    });
  }
  report(null);
  const [output] = await abstracter(text, { max_new_tokens: 142, min_length: 56, no_repeat_ngram_size: 3, num_beams: 1 });
  return output.summary_text.trim();
}

const handlers = {
  load,
  loadVariant,
  loadVision,
  reset: () => { frames = []; },
  setFrames: ({ features }) => { frames = [features]; },
  encodeFrames: ({ pixels, count }) => encodeFrames(pixels, count),
  summarize: ({ sentences, transcript, variant }) => summarize(sentences, transcript, variant),
  vsumm: ({ count }) => vsummFrames(allFrames(), count),
  transcribe: ({ audio }, id) => transcribe(audio, (percent) => self.postMessage({ id, progress: percent })),
  abstract: ({ text }, id) => abstract(text, (percent) => self.postMessage({ id, progress: percent })),
};

self.onmessage = async ({ data: { id, type, ...payload } }) => {
  try {
    self.postMessage({ id, result: await handlers[type](payload, id) });
  } catch (error) {
    self.postMessage({ id, error: error.message ?? String(error) });
  }
};
