// Browser test of Pokémon mode: teach a Pokémon from the (fake) camera, scan it, then teach
// the background from the same picture and check the scanner refuses to guess.
//
//   node web/test/teach.e2e.mjs <site-url> <video.y4m> [--mock] [--shots dir]
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)("playwright");
const [url, video, ...rest] = process.argv.slice(2);
const mock = rest.includes("--mock");
const shotsAt = rest.indexOf("--shots");
const shots = shotsAt >= 0 ? rest[shotsAt + 1] : null;
if (shots) mkdirSync(shots, { recursive: true });

const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${video}`],
});
const context = await browser.newContext({
  viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, permissions: ["camera"],
});
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("dialog", (d) => d.accept());
const shot = async (name) => shots && page.screenshot({ path: `${shots}/teach_${name}.png` });

async function scanResult() {
  await page.click("#scanBtn");
  const sheet = await page.waitForSelector("#entrySheet:not([hidden]), #unknownSheet:not([hidden])", { timeout: 30000 });
  await page.waitForTimeout(400);
  return (await sheet.getAttribute("id")) === "entrySheet" ? (await page.textContent("#entryName")).trim() : "unknown";
}

let failed = null;
const check = (cond, msg) => { if (!cond && !failed) failed = msg; };
try {
  await page.goto(url + (mock ? "?mock&mode=pokemon" : "?mode=pokemon"));
  await page.waitForSelector(".glass.live", { timeout: 20000 });
  await page.waitForFunction(() => document.getElementById("scanLabel").textContent === "Teach", null, { timeout: 120000 });
  await shot("1_empty");

  await page.click("#scanBtn"); // with nothing taught, the big button opens the teach screen
  await page.waitForSelector("#teachSheet:not([hidden])", { timeout: 5000 });
  await page.fill("#teachInput", "bulbasaur");
  await page.press("#teachInput", "Enter");
  await page.waitForSelector("#teachGoBtn:not([disabled])", { timeout: 30000 });
  check((await page.textContent("#teachName")).trim() === "Bulbasaur", "PokéAPI lookup didn't return Bulbasaur");
  await shot("2_pick");
  await page.click("#teachGoBtn");
  await page.waitForTimeout(1500);
  await shot("3_capturing");
  await page.waitForSelector("#learnedSheet:not([hidden])", { timeout: 60000 });
  check((await page.textContent("#learnedTitle")).includes("Bulbasaur learned"), "no 'learned' confirmation");
  await shot("4_learned");
  await page.click("#learnedDoneBtn");

  const first = await scanResult();
  console.log("RESULT after teaching Bulbasaur:", first);
  check(first === "Bulbasaur", `expected Bulbasaur, got ${first}`);
  await shot("5_result");
  await page.keyboard.press("Escape");

  // Background taught from the very same picture: now it's a coin toss, which must read as "no match".
  await page.click("#seenBtn");
  await page.click("#dexTeachBtn");
  await page.click("#teachBgBtn");
  await page.waitForSelector("#learnedSheet:not([hidden])", { timeout: 60000 });
  await page.click("#learnedDoneBtn");
  const second = await scanResult();
  console.log("RESULT after teaching an identical background:", second);
  check(second === "unknown", `expected no match when Bulbasaur and background look identical, got ${second}`);
  await page.keyboard.press("Escape");

  // Persists across reloads (IndexedDB)
  await page.reload();
  await page.waitForFunction(() => document.getElementById("seenCount").textContent === "1" && document.getElementById("seenOf").textContent === "taught", null, { timeout: 120000 });
  await page.click("#seenBtn");
  await page.waitForTimeout(500);
  await shot("6_dex");
  const tiles = await page.$$eval("#dexGrid .nm", (els) => els.map((e) => e.textContent));
  console.log("Dex after reload:", tiles);
  check(tiles.includes("Bulbasaur"), "taught Pokémon not kept after reload");
} catch (e) {
  failed = e.message;
  await shot("error").catch(() => {});
}
for (const l of logs.filter((l) => !/favicon|ERR_TUNNEL|404 \(File not found\)/.test(l))) console.log(l);
await browser.close();
if (failed) { console.error("FAIL: " + failed); process.exit(1); }
console.log("OK");
