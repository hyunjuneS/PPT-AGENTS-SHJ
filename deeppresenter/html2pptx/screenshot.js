'use strict';
/**
 * screenshot.js - render an HTML slide to JPEG using Playwright
 * Usage: node screenshot.js --html <path> --output <path> [--width 1280] [--height 720]
 */
const { chromium } = require('playwright-core');
const path = require('path');
const args = require('minimist')(process.argv.slice(2));

(async () => {
  const htmlFile = path.resolve(args.html);
  const outputFile = path.resolve(args.output);
  const width  = parseInt(args.width  || '1280', 10);
  const height = parseInt(args.height || '720',  10);

  const browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width, height });
    await page.goto(`file://${htmlFile}`, { waitUntil: 'networkidle' });
    await page.screenshot({ path: outputFile, type: 'jpeg', quality: 85, fullPage: false });
    console.log(`Screenshot saved: ${outputFile}`);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e.message); process.exit(1); });
