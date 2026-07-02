'use strict';
/**
 * screenshot.js - render an HTML slide to JPEG using Playwright
 * Usage: node screenshot.js --html <path> --output <path> [--width 1280] [--height 720]
 */
const { chromium } = require('playwright');
const path = require('path');
const args = require('minimist')(process.argv.slice(2));

(async () => {
  const htmlFile = path.resolve(args.html);
  const outputFile = path.resolve(args.output);
  const width  = parseInt(args.width  || '1280', 10);
  const height = parseInt(args.height || '720',  10);

  const launchOptions = {
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--lang=ko-KR,ko,en-US,en',
      '--font-render-hinting=none',
    ],
  };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    launchOptions.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const browser = await chromium.launch(launchOptions);

  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width, height });
    await page.goto(`file://${htmlFile}`, { waitUntil: 'networkidle' });

    // Measure real content overflow before any cosmetic DOM tweaks below.
    // scrollWidth/scrollHeight reflect the true laid-out content size even
    // when body has `overflow:hidden` (which visually clips it, so it would
    // otherwise be invisible in the screenshot below).
    const dims = await page.evaluate(() => {
      const body = document.body;
      const style = window.getComputedStyle(body);
      return {
        width: parseFloat(style.width),
        height: parseFloat(style.height),
        scrollWidth: body.scrollWidth,
        scrollHeight: body.scrollHeight,
      };
    });

    // Visualize data-chart-type placeholders for this screenshot only.
    // The real element is an empty div only consumed by html2pptx.js at PPTX
    // export time, so it renders as nothing here -- without this, the VLM
    // sees blank space and may place other content on top of the chart's
    // reserved area, causing real overlap once exported. This styling is
    // applied only to this page's live DOM and is never written back to the
    // source HTML file.
    await page.evaluate(() => {
      document.querySelectorAll('[data-chart-type]').forEach((el) => {
        const type = el.getAttribute('data-chart-type') || 'chart';
        el.style.backgroundColor = '#E8ECF3';
        el.style.border = '2px dashed #1F3864';
        el.style.boxSizing = 'border-box';
        el.style.display = 'flex';
        el.style.alignItems = 'center';
        el.style.justifyContent = 'center';
        el.style.color = '#1F3864';
        el.style.fontSize = '20px';
        el.style.fontWeight = 'bold';
        el.textContent = `[CHART: ${type}]`;
      });
    });

    // Append Noto Sans CJK KR as a last-resort fallback on every element so Korean
    // glyphs render even when the slide specifies a Latin-only font (e.g. Arial).
    await page.evaluate(() => {
      const faceStyle = document.createElement('style');
      faceStyle.textContent =
        "@font-face{font-family:'NotoKR';" +
        "src:local('Noto Sans CJK KR'),local('NotoSansCJKkr-Regular'),local('NotoSansCJK-Regular');" +
        "unicode-range:U+AC00-D7A3,U+1100-11FF,U+3130-318F;}";
      document.head.appendChild(faceStyle);

      document.querySelectorAll('*').forEach(el => {
        const ff = window.getComputedStyle(el).fontFamily;
        if (!ff.includes('NotoKR') && !ff.includes('Noto Sans CJK')) {
          el.style.fontFamily = `${ff}, 'NotoKR'`;
        }
      });
    });
    await page.screenshot({ path: outputFile, type: 'jpeg', quality: 85, fullPage: false });
    console.log(`Screenshot saved: ${outputFile}`);
    console.log(`DIMS:${JSON.stringify(dims)}`);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e.message); process.exit(1); });
