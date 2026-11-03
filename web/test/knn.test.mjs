import { test } from "node:test";
import assert from "node:assert/strict";
import { Learner, BACKGROUND } from "../knn.js";

// Fake embeddings: each "object" is a direction in 64-d space plus noise.
function rng(seed) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5;
}
const r = rng(7);
const base = (k) => Float32Array.from({ length: 64 }, (_, i) => (i % 8 === k ? 1 : 0));
const noisy = (v, amount = 0.35) => Float32Array.from(v, (x) => x + amount * r());

test("learns two plushies and the background", () => {
  const l = new Learner();
  for (let i = 0; i < 20; i++) {
    l.add("bulbasaur", noisy(base(0)));
    l.add("munchlax", noisy(base(1)));
    l.add(BACKGROUND, noisy(base(5)));
  }
  assert.deepEqual(l.labels().sort(), ["bulbasaur", "munchlax"]);
  assert.equal(l.predict(noisy(base(0))).label, "bulbasaur");
  assert.equal(l.predict(noisy(base(1))).label, "munchlax");
  assert.equal(l.predict(noisy(base(5))).label, BACKGROUND);
  assert.ok(l.predict(noisy(base(0))).confidence > 0.8);
});

test("ambiguous input gets low confidence", () => {
  const l = new Learner();
  for (let i = 0; i < 10; i++) {
    l.add("a", noisy(base(0), 0.05));
    l.add("b", noisy(base(1), 0.05));
  }
  const mixed = Float32Array.from(base(0), (x, i) => x + base(1)[i]);
  assert.ok(l.predict(mixed).confidence < 0.7);
});

test("serialize / restore round-trips and remove forgets", () => {
  const l = new Learner();
  for (let i = 0; i < 5; i++) {
    l.add("bulbasaur", noisy(base(0)));
    l.add("munchlax", noisy(base(1)));
  }
  const saved = l.serialize();
  // what IndexedDB hands back may be a plain array
  const plain = Object.fromEntries(Object.entries(saved).map(([k, v]) => [k, { dim: v.dim, data: Array.from(v.data) }]));
  const back = Learner.restore(plain);
  assert.deepEqual(back.counts(), { bulbasaur: 5, munchlax: 5 });
  assert.equal(back.predict(noisy(base(1))).label, "munchlax");
  back.remove("munchlax");
  assert.deepEqual(back.labels(), ["bulbasaur"]);
});

test("nothing taught -> null", () => {
  assert.equal(new Learner().predict(base(0)), null);
});
