import { Dex, topK, TYPE_COLORS } from "./dex.js";

const THRESHOLD = 0.6;
const MIN_SCAN_MS = 900; // keep the sweep on screen long enough to read as a scan
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const state = {
  dex: null,
  model: null, // { classify(canvas) -> [[label, prob], ...] }
  labels: [],
  stream: null,
  facing: "environment",
  voice: loadPref("voice", true),
  seen: loadSeen(),
  busy: false,
};

// ---------- storage (per device; the dex progress lives on the phone) ----------
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

// ---------- status line ----------
function setStatus(text) { $("status").textContent = text || ""; }
function updateSeen() {
  const n = state.dex ? state.dex.entries.filter((e) => state.seen[e.name]).length : 0;
  $("seenCount").textContent = n;
  $("seenTotal").textContent = state.dex ? state.dex.size : 0;
}
function updateScanButton() {
  const ready = state.model && state.stream;
  $("scanBtn").disabled = !ready || state.busy;
  $("scanLabel").textContent = ready ? "Scan" : !state.model ? "Loading" : "Camera off";
}

// ---------- data ----------
async function loadDex() {
  const res = await fetch("world.json");
  const data = await res.json();
  state.dex = new Dex(data.entries || data);
  updateSeen();
}

async function loadLabels() {
  try {
    const res = await fetch("labels.txt");
    if (!res.ok) return [];
    return (await res.text()).split("\n").map((l) => l.trim()).filter(Boolean);
  } catch { return []; }
}

// ---------- model ----------
async function loadModel() {
  const mock = params.get("mock");
  if (mock !== null) {
    // Test mode: returns the given labels in turn, e.g. ?mock=golden%20retriever,toaster
    const seq = (mock || "golden retriever").split(",");
    let i = 0;
    return { classify: async () => [[seq[i++ % seq.length], 0.9]] };
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
  updateScanButton();
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
function grabFrame() {
  const v = $("video");
  const c = $("still");
  const s = Math.min(v.videoWidth, v.videoHeight);
  if (!s) return null;
  const side = 224;
  c.width = side; c.height = side;
  const ctx = c.getContext("2d");
  ctx.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, side, side);
  return c;
}

// ---------- scanning ----------
async function scan() {
  if (state.busy || !state.model || !state.stream) return;
  closeSheets();
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
  let preds = [];
  try {
    preds = await state.model.classify(frame);
  } catch (e) {
    setStatus("Recognition failed: " + (e.message || e));
  }
  const wait = MIN_SCAN_MS - (performance.now() - started);
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  glass.classList.remove("scanning");
  document.body.classList.remove("scanning");
  state.busy = false;
  updateScanButton();

  const { entry, confidence, top } = state.dex.best(preds);
  console.log("scan", preds.slice(0, 5));
  if (entry && confidence >= THRESHOLD) {
    const isNew = !state.seen[entry.name];
    state.seen[entry.name] = state.seen[entry.name] || Date.now();
    saveSeen();
    updateSeen();
    showEntry(entry, confidence, isNew);
  } else {
    showUnknown(top, preds[0]?.[1]);
  }
}

// ---------- sheets ----------
let lastFocus = null;
function openSheet(id) {
  closeSheets(false);
  lastFocus = document.activeElement;
  $("scrim").hidden = false;
  $(id).hidden = false;
  $(id).querySelector(".primary")?.focus({ preventScroll: true });
}
function closeSheets(restore = true) {
  for (const id of ["entrySheet", "unknownSheet", "dexSheet"]) $(id).hidden = true;
  $("scrim").hidden = true;
  document.querySelector(".glass").classList.remove("frozen");
  window.speechSynthesis?.cancel();
  if (restore && lastFocus) lastFocus.focus({ preventScroll: true });
}

function fmt(n, unit) { return n == null ? "Unknown" : `${n} ${unit}`; }

function showEntry(e, confidence, isNew) {
  $("entryNo").textContent = `No. ${String(e.number).padStart(3, "0")}`;
  $("entryNew").hidden = !isNew;
  $("entryName").textContent = e.name;
  $("entryCat").textContent = e.category || "";
  const types = $("entryTypes");
  types.replaceChildren(...(e.types || []).map((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    li.style.background = TYPE_COLORS[t.toLowerCase()] || "#6b7480";
    return li;
  }));
  $("entryHt").textContent = fmt(e.height_m, "m");
  $("entryWt").textContent = fmt(e.weight_kg, "kg");
  $("entryMatch").textContent = confidence == null ? "Logged" : `${Math.round(confidence * 100)}%`;
  $("entryDesc").textContent = e.description || "";
  const evo = $("entryEvo");
  const chain = e.evolutions && e.evolutions.length > 1 ? e.evolutions : [];
  evo.hidden = !chain.length;
  evo.replaceChildren(...chain.map((name) => {
    const li = document.createElement("li");
    li.textContent = name;
    if (name.toLowerCase() === e.name.toLowerCase()) li.setAttribute("aria-current", "true");
    return li;
  }));
  openSheet("entrySheet");
  speak(`${e.name}. ${e.category ? e.category + "." : ""} ${e.description || ""}`);
}

function showUnknown(top, conf) {
  $("unknownText").textContent = top
    ? `Closest guess was ${top.split(",")[0]} at ${Math.round((conf || 0) * 100)}%, which isn't in this dex. Fill the frame with the subject, add light, and scan again.`
    : "Nothing recognized. Fill the frame with the subject, add light, and scan again.";
  openSheet("unknownSheet");
}

function showDex() {
  const grid = $("dexGrid");
  grid.replaceChildren(...state.dex.entries.map((e) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    const seen = !!state.seen[e.name];
    b.disabled = !seen;
    b.innerHTML = `<span class="no"></span><span class="nm"></span>`;
    b.querySelector(".no").textContent = String(e.number).padStart(3, "0");
    b.querySelector(".nm").textContent = seen ? e.name : "Not seen";
    b.setAttribute("aria-label", seen ? `${e.name}, number ${e.number}` : `Number ${e.number}, not seen yet`);
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

// ---------- wiring ----------
function wire() {
  $("startBtn").addEventListener("click", startCamera);
  $("scanBtn").addEventListener("click", scan);
  $("againBtn").addEventListener("click", scan);
  $("unknownAgainBtn").addEventListener("click", scan);
  $("dexCloseBtn").addEventListener("click", () => closeSheets());
  $("scrim").addEventListener("click", () => closeSheets());
  $("seenBtn").addEventListener("click", showDex);
  $("flipBtn").addEventListener("click", () => {
    state.facing = state.facing === "environment" ? "user" : "environment";
    startCamera();
  });
  $("modeBtn").addEventListener("click", () => setStatus("Pokémon mode arrives once its model is trained. World mode is on."));
  $("speakBtn").addEventListener("click", () => {
    state.voice = !state.voice;
    savePref("voice", state.voice);
    renderVoice();
    if (!state.voice) window.speechSynthesis?.cancel();
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
  renderVoice();
  updateScanButton();
  try {
    await loadDex();
  } catch {
    setStatus("Couldn't load the dex data. Reload the page.");
    return;
  }
  // Camera first (prompts once), model in parallel: the model is ~14 MB on first visit.
  startCamera();
  setStatus("Loading the scanner…");
  try {
    const [model, labels] = await Promise.all([loadModel(), loadLabels()]);
    state.labels = labels;
    state.dex.expandRanges(labels);
    state.model = model;
    setStatus("");
  } catch (e) {
    setStatus(e.message || "The scanner failed to load. Reload the page.");
  }
  updateScanButton();
}

main();
