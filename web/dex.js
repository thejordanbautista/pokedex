// Dex logic shared by the phone app and its tests. Mirrors pokedex/dex.py so both apps
// read data/world.json the same way.

export function normalize(label) {
  return String(label).trim().toLowerCase().replace(/[_-]/g, " ").replace(/\s+/g, " ");
}

export class Dex {
  constructor(entries) {
    this.entries = [...entries].sort((a, b) => a.number - b.number);
    this.byLabel = new Map();
    for (const e of this.entries) {
      for (const l of [e.name, ...(e.labels || [])]) this.byLabel.set(normalize(l), e);
    }
  }

  get size() {
    return this.entries.length;
  }

  lookup(label) {
    // ImageNet names can carry synonyms: "tabby, tabby cat". Any of them may match.
    for (const part of String(label).split(",")) {
      const e = this.byLabel.get(normalize(part));
      if (e) return e;
    }
    return null;
  }

  // Map ImageNet index ranges (0-999) to the model's label names.
  // `labels` may start with "background" (1001 entries), as MobileNet's label file does.
  expandRanges(labels) {
    if (!labels || !labels.length) return 0;
    const offset = normalize(labels[0]) === "background" ? 1 : 0;
    let added = 0;
    for (const e of this.entries) {
      for (const [first, last] of e.imagenet_ranges || []) {
        for (let i = first + offset; i <= Math.min(last + offset, labels.length - 1); i++) {
          const key = normalize(labels[i]);
          if (!this.byLabel.has(key)) {
            this.byLabel.set(key, e);
            added++;
          }
        }
      }
    }
    return added;
  }

  // Pool confidence per entry across the top guesses: 0.30 golden retriever + 0.25 Labrador
  // + 0.15 kuvasz says "Dog" at 0.70 even though no single breed clears the threshold.
  best(predictions) {
    const totals = new Map();
    for (const [label, conf] of predictions) {
      const e = this.lookup(label);
      if (e) totals.set(e, (totals.get(e) || 0) + conf);
    }
    const top = predictions.length ? predictions[0] : null;
    if (!totals.size) return { entry: null, confidence: top ? top[1] : 0, top: top ? top[0] : null };
    let entry = null;
    let confidence = -1;
    for (const [e, c] of totals) if (c > confidence) [entry, confidence] = [e, c];
    return { entry, confidence: Math.min(confidence, 1), top: top[0] };
  }
}

export function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Top-k (label, probability) pairs from raw logits. `labels` may include "background" first.
export function topK(logits, labels, k = 10) {
  let values = Array.from(logits);
  let names = labels;
  if (values.length === 1001) values = values.slice(1); // drop background logit if present
  if (names.length === values.length + 1) names = names.slice(1); // drop background label
  const probs = softmax(values);
  return probs
    .map((p, i) => [names[i] ?? `class ${i}`, p])
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
}

export const TYPE_COLORS = {
  normal: "#A8A77A", fire: "#EE8130", water: "#6390F0", electric: "#F7D02C", grass: "#7AC74C",
  ice: "#96D9D6", fighting: "#C22E28", poison: "#A33EA1", ground: "#E2BF65", flying: "#A98FF3",
  psychic: "#F95587", bug: "#A6B91A", rock: "#B6A136", ghost: "#735797", dragon: "#6F35FC",
  dark: "#705746", steel: "#B7B7CE", fairy: "#D685AD",
};
