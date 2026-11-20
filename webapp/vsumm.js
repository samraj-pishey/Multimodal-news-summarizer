// VSUMM over ResNet features, as vsumm_choice() in m2sm_baselines.py: cluster the frames, keep the frame nearest
// each centre. Starts from k-means++ with a fixed seed, so the same video always gives the same frames.
const ITERATIONS = 25;

function normalized(features, size) {
  const out = new Float32Array(features.length);
  for (let f = 0; f < features.length / size; f++) {
    let norm = 0;
    for (let d = 0; d < size; d++) norm += features[f * size + d] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < size; d++) out[f * size + d] = features[f * size + d] / norm;
  }
  return out;
}

function distance(points, index, centres, centre, size) {
  let total = 0;
  for (let d = 0; d < size; d++) total += (points[index * size + d] - centres[centre * size + d]) ** 2;
  return total;
}

function seededRandom(seed) {
  let value = seed;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 2 ** 32;
  };
}

function nearestCentre(points, index, centres, count, size) {
  let best = 0;
  let bestDistance = Infinity;
  for (let c = 0; c < count; c++) {
    const value = distance(points, index, centres, c, size);
    if (value < bestDistance) [best, bestDistance] = [c, value];
  }
  return { centre: best, distance: bestDistance };
}

function startCentres(points, frames, k, size, random) {
  const centres = new Float32Array(k * size);
  centres.set(points.subarray(0, size));
  for (let c = 1; c < k; c++) {
    const weights = Array.from({ length: frames }, (_, f) => nearestCentre(points, f, centres, c, size).distance);
    let target = random() * weights.reduce((a, b) => a + b, 0);
    let pick = frames - 1;
    for (let f = 0; f < frames; f++) {
      target -= weights[f];
      if (target <= 0) { pick = f; break; }
    }
    centres.set(points.subarray(pick * size, (pick + 1) * size), c * size);
  }
  return centres;
}

// Frame indices in time order. Fewer than k come back if several centres share a nearest frame.
export function vsummFrames(features, k, size = 2048, seed = 42) {
  const frames = features.length / size;
  const count = Math.min(k, frames);
  if (count === 0) return [];
  const points = normalized(features, size);
  const centres = startCentres(points, frames, count, size, seededRandom(seed));
  const assigned = new Int32Array(frames).fill(-1);

  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    let changed = false;
    for (let f = 0; f < frames; f++) {
      const { centre } = nearestCentre(points, f, centres, count, size);
      if (centre !== assigned[f]) [assigned[f], changed] = [centre, true];
    }
    if (!changed) break;
    const totals = new Float64Array(count * size);
    const members = new Int32Array(count);
    for (let f = 0; f < frames; f++) {
      members[assigned[f]]++;
      for (let d = 0; d < size; d++) totals[assigned[f] * size + d] += points[f * size + d];
    }
    for (let c = 0; c < count; c++) {
      if (!members[c]) continue;
      for (let d = 0; d < size; d++) centres[c * size + d] = totals[c * size + d] / members[c];
    }
  }

  const chosen = new Set();
  for (let c = 0; c < count; c++) {
    let best = 0;
    let bestDistance = Infinity;
    for (let f = 0; f < frames; f++) {
      const value = distance(points, f, centres, c, size);
      if (value < bestDistance) [best, bestDistance] = [f, value];
    }
    chosen.add(best);
  }
  return [...chosen].sort((a, b) => a - b);
}
