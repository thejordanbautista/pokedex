import { Dex, topK, TYPE_COLORS } from "./dex.js";
import { Learner, BACKGROUND } from "./knn.js";
import { fetchEntry, mockEntry, speciesNames } from "./pokeapi.js";
import * as store from "./store.js";

const THRESHOLD = 0.6;
const MIN_SCAN_MS = 900; // keep the sweep on screen long enough to read as a scan
const TEACH_SHOTS = 20;
const TEACH_GAP_MS = 220;
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MOCK = params.has("mock");

const state = {
  mode: loadText("mode", "world"), // "world" | "pokemon"
  dex: null, // World dex from world.json
  net: null, // { classify(canvas), embed(canvas) }
  labels: [],
  learner: new Learner(), // Pokémon taught on this phone
  entries: {}, // key -> Pokémon entry (from PokéAPI)
  stream: null,
  facing: "environment",
  voice: loadPref("voice", true),
  seen: loadSeen(),
  busy: false,
  teachEntry: null, // entry being taught, or BACKGROUND
};

// ---------- small storage helpers (per device) ----------
function loadSeen() {
  try { return JSON.parse(localStorage.getItem("fielddex.seen") || "{}"); } catch { return {}; }
}
function saveSeen() {
  try { localStorage.setItem("fielddex.seen", JSON.stringify(state.seen)); } catch { /* private mode */ }
}
function loadPref(k, d) {
  try { const v = localStorage.getItem("fielddex." + k); return v === null ? d : v === "1"; } catch { return d; }
}
function savePref(k, v) {
  try { localStorage.setItem("fielddex." + k, v ? "1" : "0"); } catch { /* ignore */ }
}
function loadText(k, d) {
  try { return localStorage.getItem("fielddex." + k) || d; } catch { return d; }
}
function saveText(k, v) {
  try { localStorage.setItem("fielddex." + k, v); } catch { /* ignore */ }
}

async function saveLearned() {
  try {
    await store.set("learner", state.learner.serialize());
    await store.set("entries", state.entries);
  } catch (e) {
    setStatus("Couldn't save on this phone: " + (e.message || e));
  }
}
async function loadLearned() {
  try {
    state.learner = Learner.restore(await store.get("learner"));
    state.entries = (await store.get("entries")) || {};
  } catch { /* start empty */ }
}

// ---------- status ----------
function setStatus(text) { $("status").textContent = text || ""; }
const seenKey = (mode, name) => `${mode}:${name}`;
function updateSeen() {
  let names = [];
  if (state.mode === "world") names = state.dex ? state.dex.entries.map((e) => e.name) : [];
  else names = state.learner.labels().map((k) => state.entries[k]?.name || k);
  if (state.mode === "pokemon") {
    // Pokémon mode counts what this phone has been taught
    $("seenCount").textContent = names.length;
    $("seenOf").textContent = "taught";
  } else {
    $("seenCount").textContent = names.filter((n) => state.seen[seenKey(state.mode, n)] || state.seen[n]).length;
    $("seenOf").replaceChildren("of ", Object.assign(document.createElement("span"), { id: "seenTotal", textContent: names.length }), " seen");
  }
}
function updateScanButton() {
  // In Pokémon mode with nothing taught, the big button opens the teach screen instead.
  const needsTeach = state.mode === "pokemon" && !state.learner.labels().length;
  const ready = state.net && state.stream;
  $("scanBtn").disabled = !ready || state.busy;
  $("scanLabel").textContent = !state.net ? "Loading" : !state.stream ? "Camera off" : needsTeach ? "Teach" : "Scan";
}
function renderMode() {
  document.body.classList.toggle("mode-pokemon", state.mode === "pokemon");
  $("modeLabel").textContent = state.mode === "pokemon" ? "Pokémon" : "World";
  $("modeBtn").setAttribute("aria-label", state.mode === "pokemon" ? "Switch to World mode" : "Switch to Pokémon mode");
  updateSeen();
  updateScanButton();
  if (state.mode === "pokemon" && !state.learner.labels().length && state.stream && !state.busy) {
    setStatus("Pokémon mode learns your own figures and plushies. Tap Teach to start.");
  }
}

// ---------- data ----------
async function loadDex() {
  const res = await fetch("world.json");
  const data = await res.json();
  state.dex = new Dex(data.entries || data);
}
async function loadLabels() {
  try {
    const res = await fetch("labels.txt");
    if (!res.ok) return [];
    return (await res.text()).split("\n").map((l) => l.trim()).filter(Boolean);
  } catch { return []; }
}

// ---------- model ----------
async function loadNet() {
  if (MOCK) {
    // Offline test mode: World answers come from ?mock=a,b; embeddings are tiny thumbnails,
    // so teaching and recognizing still work on the fake camera's picture.
    const seq = (params.get("mock") || "golden retriever").split(",");
    let i = 0;
    return {
      classify: async () => [[seq[i++ % seq.length], 0.9]],
      embed: async (canvas) => {
        const t = document.createElement("canvas");
        t.width = t.height = 12;
        t.getContext("2d").drawImage(canvas, 0, 0, 12, 12);
        return Float32Array.from(t.getContext("2d").getImageData(0, 0, 12, 12).data, (v) => v / 255 - 0.5);
      },
    };
  }
  if (!window.mobilenet || !window.tf) throw new Error("The recognition library didn't load. Check your connection and reload.");
  const net = await window.mobilenet.load({ version: 2, alpha: 1.0 });
  return {
    async classify(canvas) {
      if (state.labels.length) {
        const logits = window.tf.tidy(() => net.infer(canvas, false));
        const values = await logits.data();
        logits.dispose();
        return topK(values, state.labels, 10);
      }
      const res = await net.classify(canvas, 10);
      return res.map((r) => [r.className, r.probability]);
    },
    async embed(canvas) {
      const t = window.tf.tidy(() => net.infer(canvas, true));
      const v = await t.data();
      t.dispose();
      return Float32Array.from(v);
    },
  };
}

// ---------- camera ----------
async function startCamera() {
  stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    showGlassMessage("This browser can't use the camera. Open the page in Safari or Chrome over https.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    const v = $("video");
    v.srcObject = state.stream;
    await v.play().catch(() => {});
    document.querySelector(".glass").classList.add("live");
  } catch (e) {
    state.stream = null;
    const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
    showGlassMessage(denied
      ? "Camera access is off. In Safari: aA menu, Website Settings, Camera, Allow. Then reload."
      : "No camera found. Try the other camera, or reload the page.");
  }
  renderMode();
}
function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}
function showGlassMessage(text) {
  document.querySelector(".glass").classList.remove("live");
  $("glassText").textContent = text;
}

// Center-square crop of the current video frame, at the model's input size.
function grabFrame(target = $("still")) {
  const v = $("video");
  const s = Math.min(v.videoWidth, v.videoHeight);
  if (!s) return null;
  target.width = target.height = 224;
  target.getContext("2d").drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, 224, 224);
  return target;
}

// ---------- scanning ----------
async function scan() {
  if (state.busy || !state.net || !state.stream) return;
  if (state.mode === "pokemon" && !state.learner.labels().length) return openTeach();
  closeSheets(false);
  const frame = grabFrame();
  if (!frame) { setStatus("The camera hasn't started yet. Try again in a second."); return; }
  state.busy = true;
  updateScanButton();
  const glass = document.querySelector(".glass");
  glass.classList.add("frozen", "scanning");
  document.body.classList.add("scanning");
  setStatus("");
  if (navigator.vibrate) navigator.vibrate(15);
  const started = performance.now();
  let result = null;
  try {
    result = state.mode === "pokemon" ? await recognizePokemon(frame) : await recognizeWorld(frame);
  } catch (e) {
    setStatus("Recognition failed: " + (e.message || e));
  }
  const wait = MIN_SCAN_MS - (performance.now() - started);
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  glass.classList.remove("scanning");
  document.body.classList.remove("scanning");
  state.busy = false;
  updateScanButton();
  if (!result) return;
  if (result.entry) {
    const key = seenKey(state.mode, result.entry.name);
    const isNew = !state.seen[key] && !state.seen[result.entry.name];
    state.seen[key] = state.seen[key] || Date.now();
    saveSeen();
    updateSeen();
    showEntry(result.entry, result.confidence, isNew);
  } else {
    showUnknown(result.message);
  }
}

async function recognizeWorld(frame) {
  const preds = await state.net.classify(frame);
  const { entry, confidence, top } = state.dex.best(preds);
  console.log("scan", preds.slice(0, 5));
  if (entry && confidence >= THRESHOLD) return { entry, confidence };
  const guess = top ? top.split(",")[0] : null;
  const plushie = guess && /teddy|toy|doll|plush/i.test(guess);
  return {
    message: guess
      ? `Closest guess was ${guess} at ${Math.round((preds[0]?.[1] || 0) * 100)}%, which isn't in the World dex. ` +
        (plushie ? "For Pokémon figures and plushies, switch to Pokémon mode and teach them."
          : "Fill the frame with the subject, add light, and scan again.")
      : "Nothing recognized. Fill the frame with the subject, add light, and scan again.",
  };
}

async function recognizePokemon(frame) {
  const emb = await state.net.embed(frame);
  const p = state.learner.predict(emb);
  console.log("scan", p);
  if (p && p.label !== BACKGROUND && p.confidence >= THRESHOLD) {
    return { entry: state.entries[p.label] || { name: p.label, number: 0 }, confidence: p.confidence };
  }
  const near = p && p.label !== BACKGROUND ? state.entries[p.label]?.name : null;
  const hasBg = !!state.learner.counts()[BACKGROUND];
  return {
    message: (near ? `It looked a little like ${near} (${Math.round(p.confidence * 100)}%), not enough to be sure. ` : "No Pokémon you've taught is in view. ") +
      (hasBg ? "Fill the frame with it and scan again, or teach it if it's new."
        : "Teaching the background helps the scanner tell \"no Pokémon\" apart from a match."),
    teach: true,
  };
}

// ---------- teaching ----------
async function openTeach(prefill) {
  state.teachEntry = null;
  $("teachPick").hidden = true;
  $("teachGoBtn").disabled = true;
  $("teachMsg").textContent = "";
  $("teachInput").value = prefill || "";
  openSheet("teachSheet", "teachInput");
  if (!MOCK && !$("speciesList").options.length) {
    speciesNames().then((names) => {
      $("speciesList").replaceChildren(...names.map((n) => Object.assign(document.createElement("option"), { value: n })));
    }).catch(() => {});
  }
  if (prefill) findPokemon();
}

async function findPokemon(e) {
  e?.preventDefault();
  const q = $("teachInput").value.trim();
  if (!q) return;
  $("teachMsg").textContent = "Looking it up…";
  $("teachGoBtn").disabled = true;
  try {
    const entry = MOCK ? mockEntry(q) : await fetchEntry(q);
    state.teachEntry = entry;
    $("teachNo").textContent = `No. ${String(entry.number).padStart(4, "0")}`;
    $("teachName").textContent = entry.name;
    $("teachArt").src = entry.image || "";
    $("teachPick").hidden = false;
    const have = state.learner.counts()[entry.key] || 0;
    $("teachMsg").textContent = have ? `Already taught with ${have} photos. More photos make it more reliable.` : "";
    $("teachGoBtn").disabled = false;
    $("teachInput").blur(); // drop the keyboard so the button is visible
    $("teachGoBtn").scrollIntoView({ block: "nearest", behavior: "smooth" });
    $("teachGoBtn").focus({ preventScroll: true });
  } catch (err) {
    state.teachEntry = null;
    $("teachPick").hidden = true;
    $("teachMsg").textContent = err.message === "not_found"
      ? `No Pokémon called "${q}". Check the spelling or use its number.`
      : "Couldn't reach PokéAPI. Check your connection and try again.";
  }
}

async function capture(label, entry) {
  if (!state.net || !state.stream) { setStatus("Turn on the camera first."); return; }
  closeSheets(false);
  state.busy = true;
  updateScanButton();
  const glass = document.querySelector(".glass");
  glass.classList.add("capturing");
  const count = document.createElement("div");
  count.className = "count";
  glass.append(count);
  const scratch = document.createElement("canvas");
  const what = entry ? entry.name : "the room";
  try {
    for (let i = 0; i < TEACH_SHOTS; i++) {
      count.textContent = String(TEACH_SHOTS - i);
      setStatus(entry ? `Hold ${what} in the frame and turn it slowly.` : "Point at the room, the table and your hands. Keep any Pokémon out of view.");
      const frame = grabFrame(scratch);
      if (frame) state.learner.add(label, await state.net.embed(frame));
      if (navigator.vibrate) navigator.vibrate(8);
      await new Promise((r) => setTimeout(r, TEACH_GAP_MS));
    }
    if (entry) state.entries[label] = entry;
    await saveLearned();
  } finally {
    count.remove();
    glass.classList.remove("capturing");
    state.busy = false;
    setStatus("");
    renderMode();
  }
  showLearned(entry);
}

function showLearned(entry) {
  const hasBg = !!state.learner.counts()[BACKGROUND];
  const n = state.learner.labels().length;
  if (entry) {
    $("learnedTitle").textContent = `${entry.name} learned`;
    $("learnedText").textContent = !hasBg
      ? "Next, teach the background (20 photos of the room with no Pokémon in view) so random things aren't mistaken for it."
      : n < 2 ? "Scan it to try it out, or teach another Pokémon." : `The scanner knows ${n} Pokémon. Scan to try it out.`;
    $("learnedNextBtn").textContent = hasBg ? "Teach another" : "Teach the background";
    $("learnedNextBtn").dataset.next = hasBg ? "teach" : "background";
  } else {
    $("learnedTitle").textContent = "Background learned";
    $("learnedText").textContent = n ? "The scanner can now tell when no Pokémon is in view." : "Now teach your first Pokémon.";
    $("learnedNextBtn").textContent = n ? "Scan now" : "Teach a Pokémon";
    $("learnedNextBtn").dataset.next = n ? "scan" : "teach";
  }
  openSheet("learnedSheet");
}

async function forgetCurrent() {
  const key = $("entrySheet").dataset.key;
  if (!key) return;
  const name = state.entries[key]?.name || key;
  if (!confirm(`Forget ${name}? Its photos are removed from this phone.`)) return;
  state.learner.remove(key);
  delete state.entries[key];
  await saveLearned();
  closeSheets();
  renderMode();
  setStatus(`${name} forgotten.`);
}

// ---------- sheets ----------
let lastFocus = null;
const SHEETS = ["entrySheet", "unknownSheet", "dexSheet", "teachSheet", "learnedSheet"];
function openSheet(id, focusId) {
  closeSheets(false);
  lastFocus = document.activeElement;
  $("scrim").hidden = false;
  $(id).hidden = false;
  ($(focusId) || $(id).querySelector(".primary"))?.focus({ preventScroll: true });
}
function closeSheets(restore = true) {
  for (const id of SHEETS) $(id).hidden = true;
  $("scrim").hidden = true;
  document.querySelector(".glass").classList.remove("frozen");
  window.speechSynthesis?.cancel();
  if (restore && lastFocus) lastFocus.focus({ preventScroll: true });
}

const fmt = (n, unit) => (n == null ? "Unknown" : `${n} ${unit}`);

function showEntry(e, confidence, isNew) {
  const pokemon = state.mode === "pokemon";
  $("entrySheet").dataset.key = pokemon ? e.key || "" : "";
  $("entryArt").hidden = !e.image;
  $("entrySheet").classList.toggle("has-art", !!e.image);
  if (e.image) $("entryArt").src = e.image;
  $("entryNo").textContent = `No. ${String(e.number).padStart(pokemon ? 4 : 3, "0")}`;
  $("entryNew").hidden = !isNew;
  $("entryName").textContent = e.name;
  fitName($("entryName"));
  $("entryCat").textContent = e.category || "";
  $("entryTypes").replaceChildren(...(e.types || []).map((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    li.style.background = TYPE_COLORS[t.toLowerCase()] || "#6b7480";
    return li;
  }));
  $("entryHt").textContent = fmt(e.height_m, "m");
  $("entryWt").textContent = fmt(e.weight_kg, "kg");
  $("entryMatch").textContent = confidence == null ? "Logged" : `${Math.round(confidence * 100)}%`;
  $("entryDesc").textContent = e.description || "";
  const chain = e.evolutions && e.evolutions.length > 1 ? e.evolutions : [];
  $("entryEvo").hidden = !chain.length;
  $("entryEvo").replaceChildren(...chain.map((name) => {
    const li = document.createElement("li");
    li.textContent = name;
    if (name.toLowerCase() === e.name.toLowerCase()) li.setAttribute("aria-current", "true");
    return li;
  }));
  $("entryManage").hidden = !pokemon || confidence != null; // manage from the dex, not mid-scan
  openSheet("entrySheet");
  if (confidence != null) speak(`${e.name}. ${e.category ? e.category + "." : ""} ${e.description || ""}`);
}

// Shrink a long name until every word fits its line, instead of splitting a word.
function fitName(el) {
  el.style.fontSize = "";
  requestAnimationFrame(() => {
    let size = parseFloat(getComputedStyle(el).fontSize);
    while (size > 24 && el.scrollWidth > el.clientWidth + 1) {
      size -= 2;
      el.style.fontSize = size + "px";
    }
  });
}

function showUnknown(message) {
  $("unknownText").textContent = message;
  $("unknownTeachBtn").hidden = state.mode !== "pokemon";
  $("unknownPokeBtn").hidden = state.mode !== "world";
  openSheet("unknownSheet");
}

function showDex() {
  const pokemon = state.mode === "pokemon";
  const grid = $("dexGrid");
  $("dexTitle").textContent = pokemon ? "Your Pokémon" : "World dex";
  $("dexTeachBtn").hidden = !pokemon;
  const keys = state.learner.labels().sort((a, b) => (state.entries[a]?.number || 0) - (state.entries[b]?.number || 0));
  $("dexNote").hidden = !pokemon;
  $("dexNote").textContent = keys.length
    ? `${keys.length} taught on this phone.${state.learner.counts()[BACKGROUND] ? "" : " Teach the background too, so empty scans don't match."}`
    : "Nothing taught yet. Teach the scanner your figures, plushies or cards, one Pokémon at a time.";
  const items = pokemon
    ? keys.map((k) => ({ entry: state.entries[k] || { name: k, number: 0 }, seen: true, photos: state.learner.counts()[k] }))
    : state.dex.entries.map((e) => ({ entry: e, seen: !!(state.seen[seenKey("world", e.name)] || state.seen[e.name]) }));
  grid.replaceChildren(...items.map(({ entry: e, seen, photos }) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.disabled = !seen;
    if (e.image) {
      const img = document.createElement("img");
      img.src = e.image;
      img.alt = "";
      b.append(img);
    } else {
      const no = document.createElement("span");
      no.className = "no";
      no.textContent = String(e.number).padStart(3, "0");
      b.append(no);
    }
    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = seen ? e.name : "Not seen";
    b.append(nm);
    b.setAttribute("aria-label", seen ? `${e.name}${photos ? `, ${photos} photos` : ""}` : `Number ${e.number}, not seen yet`);
    b.addEventListener("click", () => showEntry(e, null, false));
    li.append(b);
    return li;
  }));
  openSheet("dexSheet");
}

// ---------- voice ----------
function speak(text) {
  if (!state.voice || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  u.pitch = 0.8;
  window.speechSynthesis.speak(u);
}
function renderVoice() {
  $("speakBtn").textContent = state.voice ? "Voice on" : "Voice off";
  $("speakBtn").setAttribute("aria-pressed", String(state.voice));
}

function setMode(mode, teach = false) {
  state.mode = mode;
  saveText("mode", mode);
  renderMode();
  if (teach) return openTeach();
  setStatus(mode === "pokemon"
    ? (state.learner.labels().length ? "Pokémon mode: recognizes the Pokémon you've taught." : "Pokémon mode: tap Teach to show it your first Pokémon.")
    : "World mode: recognizes animals and everyday objects.");
  clearTimeout(setMode.t);
  setMode.t = setTimeout(() => setStatus(""), 3500);
}

// ---------- wiring ----------
function wire() {
  $("startBtn").addEventListener("click", startCamera);
  $("scanBtn").addEventListener("click", scan);
  $("againBtn").addEventListener("click", scan);
  $("unknownAgainBtn").addEventListener("click", scan);
  $("unknownTeachBtn").addEventListener("click", () => openTeach());
  $("unknownPokeBtn").addEventListener("click", () => setMode("pokemon", true));
  $("dexCloseBtn").addEventListener("click", () => closeSheets());
  $("dexTeachBtn").addEventListener("click", () => openTeach());
  $("scrim").addEventListener("click", () => closeSheets());
  $("seenBtn").addEventListener("click", showDex);
  $("flipBtn").addEventListener("click", () => {
    state.facing = state.facing === "environment" ? "user" : "environment";
    startCamera();
  });
  $("modeBtn").addEventListener("click", () => setMode(state.mode === "world" ? "pokemon" : "world"));
  $("speakBtn").addEventListener("click", () => {
    state.voice = !state.voice;
    savePref("voice", state.voice);
    renderVoice();
    if (!state.voice) window.speechSynthesis?.cancel();
  });
  $("teachForm").addEventListener("submit", findPokemon);
  $("teachInput").addEventListener("change", () => findPokemon());
  $("teachGoBtn").addEventListener("click", () => state.teachEntry && capture(state.teachEntry.key, state.teachEntry));
  $("teachBgBtn").addEventListener("click", () => capture(BACKGROUND, null));
  $("teachMoreBtn").addEventListener("click", () => {
    const key = $("entrySheet").dataset.key;
    if (key && state.entries[key]) capture(key, state.entries[key]);
  });
  $("forgetBtn").addEventListener("click", forgetCurrent);
  $("learnedDoneBtn").addEventListener("click", () => closeSheets());
  $("learnedNextBtn").addEventListener("click", () => {
    const next = $("learnedNextBtn").dataset.next;
    if (next === "background") capture(BACKGROUND, null);
    else if (next === "teach") openTeach();
    else scan();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSheets();
    if ((e.key === " " || e.key === "Enter") && document.activeElement === document.body) { e.preventDefault(); scan(); }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopCamera();
    else if (document.querySelector(".glass").classList.contains("live")) startCamera();
  });
}

async function main() {
  wire();
  $("version").textContent = `Version ${document.documentElement.dataset.version || "dev"}`;
  renderVoice();
  if (params.get("mode") === "pokemon" || params.get("mode") === "world") state.mode = params.get("mode");
  renderMode();
  try {
    await Promise.all([loadDex(), loadLearned()]);
  } catch {
    setStatus("Couldn't load the dex data. Reload the page.");
    return;
  }
  renderMode();
  startCamera();
  setStatus("Loading the scanner…");
  try {
    const [net, labels] = await Promise.all([loadNet(), loadLabels()]);
    state.labels = labels;
    state.dex.expandRanges(labels);
    state.net = net;
    setStatus("");
  } catch (e) {
    setStatus(e.message || "The scanner failed to load. Reload the page.");
  }
  renderMode();
}

main();
