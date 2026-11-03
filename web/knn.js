// Teach-by-example recognizer: nearest neighbours over image embeddings.
//
// Each taught photo becomes one example: (label, embedding). To recognize a new photo we find
// the k most similar examples (cosine similarity) and let them vote. A "background" label,
// taught by pointing at the room, gives the votes somewhere to go when no Pokémon is in view.

export const BACKGROUND = "background";

export function normalizeVec(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export class Learner {
  constructor() {
    this.examples = []; // { label, vec: Float32Array (unit length) }
  }

  add(label, embedding) {
    this.examples.push({ label, vec: normalizeVec(embedding) });
  }

  remove(label) {
    this.examples = this.examples.filter((e) => e.label !== label);
  }

  counts() {
    const c = {};
    for (const e of this.examples) c[e.label] = (c[e.label] || 0) + 1;
    return c;
  }

  labels() {
    return Object.keys(this.counts()).filter((l) => l !== BACKGROUND);
  }

  // Returns { label, confidence, similarity } or null with nothing taught.
  // Each label scores the mean of its 3 best similarities; confidence is a softmax across
  // labels, so a photo halfway between two Pokémon comes out near 50% instead of falsely sure.
  predict(embedding, per = 3, temperature = 0.04) {
    if (!this.examples.length) return null;
    const q = normalizeVec(embedding);
    const sims = new Map();
    for (const e of this.examples) (sims.get(e.label) || sims.set(e.label, []).get(e.label)).push(dot(q, e.vec));
    const scores = [];
    for (const [label, list] of sims) {
      list.sort((x, y) => y - x);
      const top = list.slice(0, per);
      scores.push({ label, score: top.reduce((s, x) => s + x, 0) / top.length, best: top[0] });
    }
    scores.sort((x, y) => y.score - x.score);
    const max = scores[0].score;
    const exps = scores.map((s) => Math.exp((s.score - max) / temperature));
    const sum = exps.reduce((s, x) => s + x, 0);
    return { label: scores[0].label, confidence: exps[0] / sum, similarity: scores[0].best };
  }

  // Compact storage: one Float32Array per label, examples concatenated.
  serialize() {
    const byLabel = {};
    for (const e of this.examples) (byLabel[e.label] ||= []).push(e.vec);
    const out = {};
    for (const [label, vecs] of Object.entries(byLabel)) {
      const dim = vecs[0].length;
      const flat = new Float32Array(vecs.length * dim);
      vecs.forEach((v, i) => flat.set(v, i * dim));
      out[label] = { dim, data: flat };
    }
    return out;
  }

  static restore(saved) {
    const l = new Learner();
    for (const [label, { dim, data }] of Object.entries(saved || {})) {
      const flat = data instanceof Float32Array ? data : new Float32Array(data);
      for (let i = 0; i + dim <= flat.length; i += dim) l.examples.push({ label, vec: flat.slice(i, i + dim) });
    }
    return l;
  }
}
