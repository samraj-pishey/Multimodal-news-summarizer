const $ = (id) => document.getElementById(id);

const state = { data: null, story: null, length: 3 };

function topSentences(scores, count) {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .slice(0, count)
    .map((item) => item.index)
    .sort((a, b) => a - b);
}

function clearAndAppend(parent, nodes) {
  parent.replaceChildren(...nodes);
}

function renderSummary(picks) {
  clearAndAppend($("summary"), picks.map((index) => {
    const item = document.createElement("li");
    item.textContent = state.story.sentences[index];
    return item;
  }));
}

function scoreRow(label, row) {
  const tr = document.createElement("tr");
  [label, row.r1.toFixed(1), row.r2.toFixed(1), row.rl.toFixed(1)].forEach((text, i) => {
    const cell = document.createElement(i === 0 ? "th" : "td");
    if (i === 0) cell.scope = "row";
    cell.textContent = text;
    tr.append(cell);
  });
  return tr;
}

function renderScores() {
  const { rouge } = state.story;
  const n = state.length;
  const plural = n === 1 ? "sentence" : "sentences";
  clearAndAppend($("score-rows"), [
    scoreRow("This model", rouge.model[n]),
    scoreRow(`First ${n} ${plural} of the article`, rouge.lead[n]),
  ]);
}

function renderArticle(picks) {
  const chosen = new Set(picks);
  const nodes = [];
  state.story.sentences.forEach((text, index) => {
    const sentence = document.createElement(chosen.has(index) ? "mark" : "span");
    sentence.className = "sentence";
    sentence.textContent = text;
    const tag = document.createElement("span");
    tag.className = "score-tag";
    tag.textContent = `[${state.story.scores[index].toFixed(2)}]`;
    nodes.push(sentence, tag, document.createTextNode(" "));
  });
  clearAndAppend($("article"), nodes);
}

function renderFrames() {
  const { frames } = state.story;
  $("frames-block").hidden = frames.length === 0;
  clearAndAppend($("frames"), frames.map((frame) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = `data/frames/${frame.file}`;
    image.alt = `Video frame at ${frame.time} seconds`;
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = `${frame.time.toFixed(0)} s, score ${frame.score.toFixed(2)}`;
    figure.append(image, caption);
    return figure;
  }));
}

function render() {
  const picks = topSentences(state.story.scores, state.length);
  $("length").value = state.length;
  $("length-value").textContent = `${state.length} ${state.length === 1 ? "sentence" : "sentences"}`;
  renderSummary(picks);
  renderScores();
  renderArticle(picks);
}

function selectStory(index) {
  state.story = state.data.examples[index];
  document.querySelectorAll("#story-list button").forEach((button, i) => {
    button.setAttribute("aria-current", String(i === index));
  });
  $("reference").textContent = state.story.reference;
  $("transcript").textContent = state.story.transcript;
  $("transcript-block").hidden = !state.story.transcript;
  renderFrames();
  render();
}

function renderStoryList() {
  clearAndAppend($("story-list"), state.data.examples.map((story, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    const words = story.sentences[0].split(/\s+/);
    button.textContent = words.slice(0, 14).join(" ") + (words.length > 14 ? "…" : "");
    button.addEventListener("click", () => selectStory(index));
    item.append(button);
    return item;
  }));
}

function renderModelInfo() {
  const { run, learning_rate: rate, test } = state.data;
  const summary = test
    ? ` Test average over ${test.n} Daily Mail stories: ROUGE-1 ${test.r1.toFixed(2)}, ROUGE-2 ${test.r2.toFixed(2)}, ROUGE-L ${test.rl.toFixed(2)}.`
    : "";
  $("model-info").textContent = `Model: ${run}, learning rate ${rate}.${summary}`;
  if (!state.data.selects_frames) {
    $("notice").textContent = `Preview: this model (${run}) doesn't select video frames. The final demo will use the best video model.`;
    $("notice").hidden = false;
  }
}

async function start() {
  const response = await fetch("data/examples.json");
  if (!response.ok) throw new Error(`examples.json: ${response.status}`);
  state.data = await response.json();
  state.length = state.data.default_length;
  $("length").max = state.data.max_length;
  $("length").addEventListener("input", (event) => {
    state.length = Number(event.target.value);
    render();
  });
  renderModelInfo();
  renderStoryList();
  selectStory(0);
}

start().catch((error) => {
  $("notice").textContent = `Could not load the examples: ${error.message}`;
  $("notice").hidden = false;
});
