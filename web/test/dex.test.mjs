import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Dex, topK, normalize } from "../dex.js";

const world = JSON.parse(readFileSync(new URL("../../data/world.json", import.meta.url)));

test("labels and ImageNet synonyms map to entries", () => {
  const dex = new Dex(world.entries);
  assert.equal(dex.lookup("golden retriever").name, "Dog");
  assert.equal(dex.lookup("tabby, tabby cat").name, "Cat");
  assert.equal(dex.lookup("Tiger_Cat").name, "Cat");
  assert.equal(dex.lookup("toaster"), null);
});

test("ranges cover every dog breed and respect a background label", () => {
  const dex = new Dex(world.entries);
  const labels = ["background", ...Array.from({ length: 1000 }, (_, i) => `class${i}`)];
  labels[1 + 151] = "Chihuahua";
  labels[1 + 268] = "Mexican hairless";
  assert.ok(dex.expandRanges(labels) > 100);
  assert.equal(dex.lookup("Mexican hairless").name, "Dog");
  assert.equal(dex.lookup("class150"), null);
  assert.equal(dex.lookup("class269"), null);
});

test("best() pools confidence across breeds", () => {
  const dex = new Dex(world.entries);
  const r = dex.best([["golden retriever", 0.3], ["tennis ball", 0.28], ["labrador retriever", 0.25], ["pug", 0.1]]);
  assert.equal(r.entry.name, "Dog");
  assert.ok(Math.abs(r.confidence - 0.65) < 1e-9);
  assert.equal(r.top, "golden retriever");
  const none = dex.best([["toaster", 0.9]]);
  assert.equal(none.entry, null);
  assert.equal(none.confidence, 0.9);
});

test("topK drops the background logit and label consistently", () => {
  const labels = ["background", "tench", "goldfish", "shark"];
  const logits = [9, 0, 5, 1]; // background would win if not dropped
  const top = topK(logits.concat(Array(997).fill(-20)), labels.concat(Array(997).fill("x")), 2);
  assert.equal(top[0][0], "goldfish");
  const top2 = topK([0, 5, 1], labels, 1); // 3 logits, 4 labels -> labels shifted
  assert.equal(top2[0][0], "goldfish");
});

test("normalize matches the Python version", () => {
  assert.equal(normalize("  Tiger_Cat-Big  "), "tiger cat big");
});
