'use strict';
/**
 * validate.js - DOM-based slide validation using Playwright
 * Detects overflow and overlap issues that cause text to overlap in PPTX.
 *
 * Usage: node validate.js --html <path> [--width 1280] [--height 720]
 * Output: JSON array of { type, element, detail } to stdout.
 */
const { chromium } = require('playwright-core');
const path = require('path');
const args = require('minimist')(process.argv.slice(2));

(async () => {
  const htmlFile = path.resolve(args.html);
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

    const issues = await page.evaluate(() => {
      const found = [];

      // ── 1. Overflow detection ────────────────────────────────────────────────
      // overflow:hidden 컨테이너에서 실제 내용물(scrollHeight)이
      // 가시 높이(clientHeight)를 초과하면 PPT에서 텍스트가 튀어나와 겹침 발생.
      document.querySelectorAll('*').forEach(el => {
        const s = window.getComputedStyle(el);
        const isHidden = s.overflow === 'hidden' || s.overflowY === 'hidden';
        if (!isHidden) return;

        const scrollH = el.scrollHeight;
        const clientH = el.clientHeight;
        // 2px 이상 차이 시 오류 (렌더링 소수점 오차 무시)
        if (scrollH <= clientH + 2) return;

        const tag = el.tagName.toLowerCase();
        const cls = el.className && typeof el.className === 'string'
          ? el.className.trim().split(/\s+/)[0]
          : '';
        const label = cls ? `${tag}.${cls}` : tag;

        found.push({
          type: 'overflow',
          element: label,
          detail: `content height ${scrollH}px exceeds container ${clientH}px — text will overflow in PPT (add ${scrollH - clientH}px to height)`,
        });
      });

      // ── 2. Overlap detection ─────────────────────────────────────────────────
      // position:absolute/fixed 요소들의 bounding box 교차 여부 확인.
      // 2px 이상 겹치면 PPT에서도 겹침.
      const positioned = [];
      document.querySelectorAll('*').forEach(el => {
        const s = window.getComputedStyle(el);
        if (s.position !== 'absolute' && s.position !== 'fixed') return;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return;
        const tag = el.tagName.toLowerCase();
        const cls = el.className && typeof el.className === 'string'
          ? el.className.trim().split(/\s+/)[0]
          : '';
        positioned.push({ label: cls ? `${tag}.${cls}` : tag, r });
      });

      for (let i = 0; i < positioned.length; i++) {
        for (let j = i + 1; j < positioned.length; j++) {
          const a = positioned[i].r;
          const b = positioned[j].r;
          const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          if (ox > 2 && oy > 2) {
            found.push({
              type: 'overlap',
              element: `${positioned[i].label} & ${positioned[j].label}`,
              detail: `overlap ${Math.round(ox)}×${Math.round(oy)}px at (${Math.round(Math.max(a.left, b.left))}, ${Math.round(Math.max(a.top, b.top))})`,
            });
          }
        }
      }

      return found;
    });

    console.log(JSON.stringify(issues));
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e.message); process.exit(1); });
