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
      const MAX_ISSUES = 8;

      // 1. Overflow detection
      // overflow:hidden 컨테이너에서 scrollHeight > clientHeight 이면
      // PPT에서 잘렸던 텍스트가 아래로 튀어나와 겹침 발생.
      document.querySelectorAll('*').forEach(el => {
        if (found.length >= MAX_ISSUES) return;
        const s = window.getComputedStyle(el);
        if (s.overflow !== 'hidden' && s.overflowY !== 'hidden') return;

        const scrollH = el.scrollHeight;
        const clientH = el.clientHeight;
        if (scrollH <= clientH + 2) return;  // 2px 이내 오차 무시

        const tag = el.tagName.toLowerCase();
        const cls = el.className && typeof el.className === 'string'
          ? el.className.trim().split(/\s+/)[0] : '';
        found.push({
          type: 'overflow',
          element: cls ? `${tag}.${cls}` : tag,
          detail: `content ${scrollH}px > container ${clientH}px — add ${scrollH - clientH}px to height`,
        });
      });

      // 2. Overlap detection
      // position:absolute 요소들 중 부모-자식 관계가 아닌 것끼리 bounding box 교차 확인.
      const positioned = [];
      document.querySelectorAll('*').forEach(el => {
        const s = window.getComputedStyle(el);
        if (s.position !== 'absolute' && s.position !== 'fixed') return;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return;
        const tag = el.tagName.toLowerCase();
        const cls = el.className && typeof el.className === 'string'
          ? el.className.trim().split(/\s+/)[0] : '';
        positioned.push({ el, label: cls ? `${tag}.${cls}` : tag, r });
      });

      for (let i = 0; i < positioned.length && found.length < MAX_ISSUES; i++) {
        for (let j = i + 1; j < positioned.length && found.length < MAX_ISSUES; j++) {
          const A = positioned[i];
          const B = positioned[j];
          // 부모-자식 관계는 의도적 중첩이므로 skip
          if (A.el.contains(B.el) || B.el.contains(A.el)) continue;

          const a = A.r, b = B.r;
          const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          if (ox > 2 && oy > 2) {
            found.push({
              type: 'overlap',
              element: `${A.label} & ${B.label}`,
              detail: `overlap ${Math.round(ox)}x${Math.round(oy)}px at (${Math.round(Math.max(a.left, b.left))}, ${Math.round(Math.max(a.top, b.top))})`,
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
