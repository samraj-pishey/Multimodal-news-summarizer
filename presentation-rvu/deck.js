// Minimal slide engine: → / space / click to step, ← to go back, N for speaker notes, F for full screen.
// Elements with class "f" are revealed one at a time; [data-w] bars, [data-x] dots and [data-span] intervals
// animate once they are visible.
const slides = [...document.querySelectorAll(".slide")];
const notesBox = document.querySelector(".notes");
const state = { i: 0, step: 0, notes: false };
let timelineTimer = null;

const builds = (slide) => [...slide.querySelectorAll(".f")];

function visible(node, slide) {
  const build = node.closest(".f");
  return slide.classList.contains("on") && (!build || build.classList.contains("shown"));
}

function animate(slide) {
  slide.querySelectorAll("[data-w]").forEach((node) => {
    node.style.width = visible(node, slide) ? `${node.dataset.w}%` : "0";
  });
  slide.querySelectorAll("[data-x]").forEach((node) => {
    node.style.left = visible(node, slide) ? `${node.dataset.x}%` : "0";
  });
  slide.querySelectorAll("[data-span]").forEach((node) => {
    const [left, width] = node.dataset.span.split(",").map(Number);
    const on = visible(node, slide);
    node.style.left = `${on ? left : 50}%`;
    node.style.width = `${on ? width : 0}%`;
  });
  slide.querySelectorAll(".draw").forEach((node) => node.classList.toggle("shown", visible(node, slide)));
}

function runTimeline(slide) {
  clearInterval(timelineTimer);
  const box = slide.querySelector(".timeline");
  if (!box || !slide.classList.contains("on")) return;
  const segments = [...box.querySelectorAll(".speech-seg")];
  const frames = [...box.querySelectorAll(".frame-cell")];
  let k = 0;
  const tick = () => {
    if (!visible(box, slide)) return;
    segments.forEach((s, j) => s.classList.toggle("lit", j === k));
    const [from, to] = segments[k].dataset.frames.split("-").map(Number);
    frames.forEach((f, j) => f.classList.toggle("lit", j >= from && j <= to));
    k = (k + 1) % segments.length;
  };
  tick();
  timelineTimer = setInterval(tick, 1600);
}

function render() {
  slides.forEach((slide, i) => {
    slide.classList.toggle("on", i === state.i);
    builds(slide).forEach((node, k) => {
      const shown = i < state.i || (i === state.i && k < state.step);
      node.classList.toggle("shown", shown);
    });
    animate(slide);
  });
  runTimeline(slides[state.i]);
  document.querySelector(".progress").style.width = `${(100 * (state.i + 1)) / slides.length}%`;
  document.querySelector(".counter").textContent = `${state.i + 1} / ${slides.length}`;
  const notes = slides[state.i].querySelector("aside[data-notes]");
  notesBox.hidden = !state.notes || !notes;
  if (notes) notesBox.innerHTML = notes.innerHTML;
  history.replaceState(null, "", `#${state.i + 1}`);
}

function next() {
  if (state.step < builds(slides[state.i]).length) state.step += 1;
  else if (state.i < slides.length - 1) { state.i += 1; state.step = 0; }
  render();
}

function prev() {
  if (state.step > 0) state.step -= 1;
  else if (state.i > 0) { state.i -= 1; state.step = builds(slides[state.i]).length; }
  render();
}

function go(i) {
  state.i = Math.max(0, Math.min(slides.length - 1, i));
  state.step = 0;
  render();
}

document.addEventListener("keydown", (event) => {
  if (event.target.closest?.("input, select, textarea")) return;
  const key = event.key;
  if (["ArrowRight", "ArrowDown", " ", "PageDown", "Enter"].includes(key)) { event.preventDefault(); next(); }
  else if (["ArrowLeft", "ArrowUp", "PageUp", "Backspace"].includes(key)) { event.preventDefault(); prev(); }
  else if (key === "Home") go(0);
  else if (key === "End") go(slides.length - 1);
  else if (key.toLowerCase() === "n") { state.notes = !state.notes; render(); }
  else if (key.toLowerCase() === "f") {
    if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.();
  }
});
document.querySelector(".deck").addEventListener("click", (event) => {
  if (event.target.closest("a, button, .notes")) return;
  if (event.clientX < window.innerWidth * 0.25) prev(); else next();
});

const start = Number(location.hash.slice(1));
state.i = Number.isInteger(start) && start >= 1 && start <= slides.length ? start - 1 : 0;
render();
