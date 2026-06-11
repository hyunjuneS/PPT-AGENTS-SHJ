/**
 * html2pptx_cli.js
 * HTML 슬라이드 폴더 → PPTX 변환 CLI (Playwright 스크린샷 + PptxGenJS 조립)
 *
 * Usage:
 *   node html2pptx_cli.js --html_dir <dir> --output <file.pptx> [--layout 16:9]
 *
 * Env:
 *   PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH  로컬 Chromium 실행파일 경로
 */

"use strict";

const fs = require("node:fs");
const path = require("node:path");

// offline node_modules 우선 탐색
const localNodeModules = path.join(__dirname, "node_modules");
if (fs.existsSync(localNodeModules)) {
  module.paths.unshift(localNodeModules);
}

const minimist = require("minimist");
const pptxgen  = require("pptxgenjs");
const { chromium } = require("playwright");

// ---------------------------------------------------------------------------
// 레이아웃 정의
// ---------------------------------------------------------------------------

const LAYOUTS = {
  "16:9": { pptxLayout: "LAYOUT_WIDE", width: 13.33, height: 7.5,  vpW: 1280, vpH: 720  },
  "4:3":  { pptxLayout: "LAYOUT_4x3",  width: 10,    height: 7.5,  vpW: 960,  vpH: 720  },
  "A4":   { pptxLayout: "A4",          width: 8.27,  height: 11.69, vpW: 794,  vpH: 1123 },
  "A3":   { pptxLayout: "A3",          width: 11.69, height: 16.54, vpW: 1122, vpH: 1587 },
};

// ---------------------------------------------------------------------------
// 메인
// ---------------------------------------------------------------------------

async function run() {
  const args = minimist(process.argv.slice(2));
  const layoutKey = args.layout || "16:9";
  const outputFile = args.output;
  const htmlDir    = args.html_dir || args["html-dir"];

  if (!htmlDir || !outputFile) {
    console.error(
      "Usage: node html2pptx_cli.js --html_dir <dir> --output <file.pptx> [--layout 16:9|4:3|A4|A3]"
    );
    process.exit(1);
  }

  if (!fs.existsSync(htmlDir)) {
    console.error(`html_dir not found: ${htmlDir}`);
    process.exit(1);
  }

  const layout = LAYOUTS[layoutKey];
  if (!layout) {
    console.error(`Unsupported layout: ${layoutKey}. Use one of: ${Object.keys(LAYOUTS).join(", ")}`);
    process.exit(1);
  }

  // slide_NN.html 파일 목록 (정렬)
  const htmlFiles = fs.readdirSync(htmlDir)
    .filter(f => /^slide_\d+\.html$/.test(f))
    .sort()
    .map(f => path.resolve(htmlDir, f));

  if (!htmlFiles.length) {
    console.error(`No slide_NN.html files found in: ${htmlDir}`);
    process.exit(1);
  }

  console.log(`[html2pptx] ${htmlFiles.length} slides, layout=${layoutKey}`);

  // ---------------------------------------------------------------------------
  // Playwright 브라우저 실행
  // ---------------------------------------------------------------------------

  const launchOptions = {
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--hide-scrollbars",
    ],
  };

  // 환경변수로 로컬 Chromium 바이너리 지정
  const execPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  if (execPath) {
    launchOptions.executablePath = execPath;
    console.log(`[html2pptx] Using local Chromium: ${execPath}`);
  }

  const browser = await chromium.launch(launchOptions);
  const page    = await browser.newPage({
    viewport: { width: layout.vpW, height: layout.vpH },
  });

  // ---------------------------------------------------------------------------
  // PptxGenJS 초기화
  // ---------------------------------------------------------------------------

  const pptx = new pptxgen();
  pptx.author  = "DeepPresenter";
  pptx.title   = "DeepPresenter Presentation";
  pptx.company = "DeepPresenter";

  if (layoutKey === "A4" || layoutKey === "A3") {
    pptx.defineLayout({ name: layoutKey, width: layout.width, height: layout.height });
  }
  pptx.layout = layout.pptxLayout;

  // ---------------------------------------------------------------------------
  // 슬라이드별 스크린샷 → PPTX 슬라이드 추가
  // ---------------------------------------------------------------------------

  for (let i = 0; i < htmlFiles.length; i++) {
    const htmlFile = htmlFiles[i];
    console.log(`[html2pptx] Rendering (${i + 1}/${htmlFiles.length}): ${path.basename(htmlFile)}`);

    await page.goto(`file://${htmlFile}`, { waitUntil: "networkidle", timeout: 60000 });

    const pngBuffer = await page.screenshot({
      type: "png",
      clip: { x: 0, y: 0, width: layout.vpW, height: layout.vpH },
    });

    const slide = pptx.addSlide();
    slide.addImage({
      data: "data:image/png;base64," + pngBuffer.toString("base64"),
      x: 0,
      y: 0,
      w: layout.width,
      h: layout.height,
    });
  }

  await browser.close();

  // ---------------------------------------------------------------------------
  // PPTX 저장
  // ---------------------------------------------------------------------------

  const outPath = path.resolve(outputFile);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await pptx.writeFile({ fileName: outPath });
  console.log(`[html2pptx] Saved: ${outPath}`);
}

run().catch(err => {
  console.error(err?.stack || err?.message || String(err));
  process.exit(1);
});
