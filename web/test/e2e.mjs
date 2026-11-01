// Browser test: a fake camera plays a still image, the page scans it, and we check the result.
//
//   node web/test/e2e.mjs <site-url> <video.y4m|mjpeg> <expect> [--mock] [--shots dir]
//
// --mock skips the real model (for offline runs); without it the real MobileNet is used.
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";

// require() honours NODE_PATH, so a globally installed Playwright works too.
const { chromium } = createRequire(import.meta.url)("playwright");

const [url, video, expect, ...rest] = process.argv.slice(2);
const mock = rest.includes("--mock");
const shotsAt = rest.indexOf("--shots");
const shots = shotsAt >= 0 ? rest[shotsAt + 1] : null;
if (shots) mkdirSync(shots, { recursive: true });

const browser = await chromium.launch({
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${video}`,
  ],
});
const context = await browser.newContext({
  viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
  permissions: ["camera"],
});
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

const q = mock ? `?mock=${encodeURIComponent(expect === "unknown" ? "toaster" : "golden retriever")}` : "";
let failed = null;
try {
  await page.goto(url + q, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".glass.live", { timeout: 20000 });
  await page.waitForFunction(() => !document.getElementById("scanBtn").disabled, null, { timeout: 120000 });
  await page.waitForTimeout(600);
  if (shots) await page.screenshot({ path: `${shots}/phone_1_ready.png` });

  await page.click("#scanBtn");
  await page.waitForTimeout(350);
  if (shots) await page.screenshot({ path: `${shots}/phone_2_scanning.png` });

  const sheet = await page.waitForSelector("#entrySheet:not([hidden]), #unknownSheet:not([hidden])", { timeout: 30000 });
  await page.waitForTimeout(500);
  if (shots) await page.screenshot({ path: `${shots}/phone_3_result.png` });
  const id = await sheet.getAttribute("id");
  const name = id === "entrySheet" ? await page.textContent("#entryName") : "unknown";
  const detail = id === "entrySheet" ? await page.textContent("#entryMatch") : await page.textContent("#unknownText");
  console.log(`RESULT: ${name} (${detail})`);
  if (name !== expect) failed = `expected ${expect}, got ${name}`;

  if (id === "entrySheet") {
    await page.click("#againBtn"); // second scan of the same subject: not new any more
    await page.waitForSelector("#entrySheet:not([hidden])", { timeout: 30000 });
    if (await page.isVisible("#entryNew")) failed = failed || "second scan still marked as new";
    await page.keyboard.press("Escape");
    await page.waitForTimeout(300);
    await page.click("#seenBtn");
    await page.waitForTimeout(500);
    if (shots) await page.screenshot({ path: `${shots}/phone_4_dex.png` });
    const seenCount = await page.textContent("#seenCount");
    if (seenCount !== "1") failed = failed || `seen count ${seenCount}, expected 1`;
  }
} catch (e) {
  failed = e.message;
  if (shots) await page.screenshot({ path: `${shots}/phone_error.png` }).catch(() => {});
}
for (const l of logs.filter((l) => !l.includes("favicon"))) console.log(l);
await browser.close();
if (failed) {
  console.error("FAIL: " + failed);
  process.exit(1);
}
console.log("OK");
