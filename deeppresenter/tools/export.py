"""HTML slides → PPTX via Node.js (Playwright screenshot + PptxGenJS)."""

import asyncio
import html as html_escape
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "html2pptx"
_CLI_JS = _SCRIPT_DIR / "html2pptx_cli.js"

# _DEFAULT_CHROMIUM = Path(
#     "/mnt/c/Users/X0160146/Desktop/26/playwright/chromium-1223/chrome-linux64/chrome"
# )


def _get_chromium_executable() -> str | None:
    """Return Chromium executable path from env var, or None to use Playwright's installed Chromium."""
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # if _DEFAULT_CHROMIUM.exists():
    #     return str(_DEFAULT_CHROMIUM)
    return None


async def html_slides_to_pptx(
    slides_dir: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    soft: bool = True,
) -> str:
    """
    Convert slide_*.html files in slides_dir to a PPTX file using Node.js.
    Returns the output path.
    """
    if not _CLI_JS.exists():
        raise FileNotFoundError(f"html2pptx_cli.js not found at {_CLI_JS}")

    slides_path = Path(slides_dir)
    html_files = sorted(slides_path.glob("slide_*.html"))
    if not html_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    env = os.environ.copy()
    chromium_exe = _get_chromium_executable()
    if chromium_exe:
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = chromium_exe

    cmd = [
        "node", str(_CLI_JS),
        "--html_dir", str(slides_path.resolve()),
        "--output",   str(Path(output_path).resolve()),
        "--layout",   aspect_ratio,
    ]
    if soft:
        cmd.append("--soft")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_SCRIPT_DIR),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("html2pptx Node.js process timed out (5min)")

    log = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"html2pptx failed (exit {proc.returncode}):\n{log}")

    return output_path


_CHART_RENDER_MARKER = "__ppt_chart_render__"

# 순수 JS + inline SVG로 data-chart-type 요소를 실제로 그린다 (외부 라이브러리/CDN 없음).
# html2pptx.js는 data-chart-type 엘리먼트를 attribute + getBoundingClientRect()로만 읽고
# markProcessed()로 그 자손 전체를 스캔 대상에서 제외하므로(html2pptx.js:1580-1601,1224-1226),
# 여기서 그 안에 넣는 <svg>는 PPTX 변환 결과에 전혀 영향을 주지 않는다 — PPTX는 여전히
# native, 편집 가능한 차트 오브젝트로 그대로 생성된다. 브라우저로 볼 때만 보이는 시각화다.
_CHART_RENDER_SCRIPT = f"""<script>/* {_CHART_RENDER_MARKER} */(function() {{
  function svgEl(tag, attrs) {{
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }}
  function defaultColor(i) {{
    var p = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'];
    return p[i % p.length];
  }}
  function toHex(c, i) {{
    if (!c) return defaultColor(i);
    return c[0] === '#' ? c : ('#' + c);
  }}
  function renderBar(svg, W, H, labels, series, colors, barDir) {{
    var m = {{ top: 20, right: 20, bottom: 40, left: 46 }};
    var iw = W - m.left - m.right, ih = H - m.top - m.bottom;
    var maxVal = 0;
    series.forEach(function(s) {{ (s.values || []).forEach(function(v) {{ if (v > maxVal) maxVal = v; }}); }});
    if (maxVal <= 0) maxVal = 1;
    var n = labels.length || 1, sc = series.length || 1;
    if (barDir === 'bar') {{
      var groupH = ih / n, barH = (groupH * 0.7) / sc;
      labels.forEach(function(label, i) {{
        series.forEach(function(s, si) {{
          var val = (s.values || [])[i] || 0;
          var w = (val / maxVal) * iw;
          var y = m.top + i * groupH + groupH * 0.15 + si * barH;
          svg.appendChild(svgEl('rect', {{ x: m.left, y: y, width: Math.max(w, 0), height: barH * 0.9, fill: toHex(colors[si], si) }}));
        }});
        var t = svgEl('text', {{ x: m.left - 6, y: m.top + i * groupH + groupH / 2, 'text-anchor': 'end', 'dominant-baseline': 'middle', 'font-size': 11 }});
        t.textContent = label; svg.appendChild(t);
      }});
    }} else {{
      var groupW = iw / n, barW = (groupW * 0.7) / sc;
      labels.forEach(function(label, i) {{
        series.forEach(function(s, si) {{
          var val = (s.values || [])[i] || 0;
          var h = (val / maxVal) * ih;
          var x = m.left + i * groupW + groupW * 0.15 + si * barW;
          var y = m.top + ih - h;
          svg.appendChild(svgEl('rect', {{ x: x, y: y, width: barW * 0.9, height: Math.max(h, 0), fill: toHex(colors[si], si) }}));
        }});
        var t = svgEl('text', {{ x: m.left + i * groupW + groupW / 2, y: m.top + ih + 16, 'text-anchor': 'middle', 'font-size': 11 }});
        t.textContent = label; svg.appendChild(t);
      }});
    }}
    svg.appendChild(svgEl('line', {{ x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: '#999' }}));
  }}
  function renderLineArea(svg, W, H, labels, series, colors, filled) {{
    var m = {{ top: 20, right: 20, bottom: 40, left: 46 }};
    var iw = W - m.left - m.right, ih = H - m.top - m.bottom;
    var maxVal = 0;
    series.forEach(function(s) {{ (s.values || []).forEach(function(v) {{ if (v > maxVal) maxVal = v; }}); }});
    if (maxVal <= 0) maxVal = 1;
    var n = labels.length || 1;
    var stepX = n > 1 ? iw / (n - 1) : iw;
    series.forEach(function(s, si) {{
      var pts = (s.values || []).map(function(v, i) {{ return [m.left + i * stepX, m.top + ih - (v / maxVal) * ih]; }});
      var color = toHex(colors[si], si);
      var d = pts.map(function(p, i) {{ return (i === 0 ? 'M' : 'L') + p[0] + ',' + p[1]; }}).join(' ');
      if (filled && pts.length) {{
        var last = pts[pts.length - 1], first = pts[0];
        var areaD = d + ' L' + last[0] + ',' + (m.top + ih) + ' L' + first[0] + ',' + (m.top + ih) + ' Z';
        svg.appendChild(svgEl('path', {{ d: areaD, fill: color, opacity: 0.3, stroke: 'none' }}));
      }}
      svg.appendChild(svgEl('path', {{ d: d, fill: 'none', stroke: color, 'stroke-width': 2 }}));
      pts.forEach(function(p) {{ svg.appendChild(svgEl('circle', {{ cx: p[0], cy: p[1], r: 3, fill: color }})); }});
    }});
    labels.forEach(function(label, i) {{
      var t = svgEl('text', {{ x: m.left + i * stepX, y: m.top + ih + 16, 'text-anchor': 'middle', 'font-size': 11 }});
      t.textContent = label; svg.appendChild(t);
    }});
    svg.appendChild(svgEl('line', {{ x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: '#999' }}));
  }}
  function renderPie(svg, W, H, labels, series, colors, isDoughnut) {{
    var values = (series[0] && series[0].values) || [];
    var total = values.reduce(function(a, b) {{ return a + b; }}, 0) || 1;
    var legendW = labels.length ? 130 : 0;
    var cx = (W - legendW) / 2, cy = H / 2, r = Math.min(W - legendW, H) / 2 - 16;
    var innerR = isDoughnut ? r * 0.55 : 0;
    var angle = -Math.PI / 2;
    values.forEach(function(v, i) {{
      var frac = v / total, next = angle + frac * Math.PI * 2;
      var color = toHex(colors[i], i);
      var x1 = cx + r * Math.cos(angle), y1 = cy + r * Math.sin(angle);
      var x2 = cx + r * Math.cos(next), y2 = cy + r * Math.sin(next);
      var large = (next - angle) > Math.PI ? 1 : 0, d;
      if (isDoughnut) {{
        var ix1 = cx + innerR * Math.cos(angle), iy1 = cy + innerR * Math.sin(angle);
        var ix2 = cx + innerR * Math.cos(next), iy2 = cy + innerR * Math.sin(next);
        d = 'M' + ix1 + ',' + iy1 + ' L' + x1 + ',' + y1 + ' A' + r + ',' + r + ' 0 ' + large + ' 1 ' + x2 + ',' + y2 +
            ' L' + ix2 + ',' + iy2 + ' A' + innerR + ',' + innerR + ' 0 ' + large + ' 0 ' + ix1 + ',' + iy1 + ' Z';
      }} else {{
        d = 'M' + cx + ',' + cy + ' L' + x1 + ',' + y1 + ' A' + r + ',' + r + ' 0 ' + large + ' 1 ' + x2 + ',' + y2 + ' Z';
      }}
      svg.appendChild(svgEl('path', {{ d: d, fill: color }}));
      angle = next;
    }});
    if (legendW) {{
      var lx = W - legendW + 10;
      labels.forEach(function(label, i) {{
        var ly = H / 2 - (labels.length * 18) / 2 + i * 18;
        svg.appendChild(svgEl('rect', {{ x: lx, y: ly, width: 10, height: 10, fill: toHex(colors[i], i) }}));
        var t = svgEl('text', {{ x: lx + 14, y: ly + 9, 'font-size': 11 }});
        t.textContent = label; svg.appendChild(t);
      }});
    }}
  }}
  function buildChartSVG(type, labels, series, colors, barDir) {{
    var W = 600, H = 400;
    var svg = svgEl('svg', {{
      viewBox: '0 0 ' + W + ' ' + H, width: '100%', height: '100%',
      preserveAspectRatio: 'xMidYMid meet', style: 'display:block;font-family:sans-serif;'
    }});
    if (type === 'pie' || type === 'doughnut') renderPie(svg, W, H, labels, series, colors, type === 'doughnut');
    else if (type === 'line' || type === 'area') renderLineArea(svg, W, H, labels, series, colors, type === 'area');
    else renderBar(svg, W, H, labels, series, colors, barDir);
    return svg;
  }}
  function renderCharts() {{
    document.querySelectorAll('[data-chart-type]').forEach(function(el) {{
      if (el.__chartRendered) return;
      el.__chartRendered = true;
      try {{
        var type = el.getAttribute('data-chart-type');
        var labels = JSON.parse(el.getAttribute('data-chart-labels') || '[]');
        var series = JSON.parse(el.getAttribute('data-chart-series') || '[]');
        var colors = JSON.parse(el.getAttribute('data-chart-colors') || '[]');
        var barDir = el.getAttribute('data-chart-bardir') || 'col';
        el.appendChild(buildChartSVG(type, labels, series, colors, barDir));
      }} catch (e) {{ /* leave the div empty if malformed */ }}
    }});
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', renderCharts);
  else renderCharts();
}})();</script>"""


def inject_chart_rendering(slides_dir: str) -> int:
    """slides_dir의 slide_*.html 중 data-chart-type 요소가 있는 파일에 순수 JS+SVG 차트
    렌더링 스크립트를 </body> 직전에 삽입해, PPTX 변환 없이도(개별 슬라이드 html, 이걸 그대로
    embed하는 combined.html 둘 다) 브라우저에서 실제 차트가 보이게 한다.

    html2pptx.js는 data-chart-type 엘리먼트를 attribute + getBoundingClientRect()만으로 읽고
    그 자손 전체를 스캔에서 제외하므로, 이 스크립트가 렌더링해 넣는 <svg>는 PPTX 변환 결과에
    전혀 영향을 주지 않는다 — PPTX는 여전히 native, 편집 가능한 차트로 그대로 생성된다.

    이미 삽입된 파일은 마커로 감지해 다시 건드리지 않는다. 반환값은 실제로 수정한 파일 수.
    """
    slides_path = Path(slides_dir)
    modified = 0
    for slide_file in sorted(slides_path.glob("slide_*.html")):
        content = slide_file.read_text(encoding="utf-8")
        if "data-chart-type" not in content or _CHART_RENDER_MARKER in content:
            continue
        if "</body>" in content:
            content = content.replace("</body>", f"{_CHART_RENDER_SCRIPT}</body>", 1)
        else:
            content += _CHART_RENDER_SCRIPT
        slide_file.write_text(content, encoding="utf-8")
        modified += 1
    return modified


def combine_html_slides(
    slides_dir: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
) -> str:
    """slides_dir의 slide_*.html N개를 세로로 스크롤 가능한 하나의 HTML로 합쳐
    output_path에 저장하고 그 경로를 반환한다.

    각 슬라이드는 원본 문서를 그대로 <iframe srcdoc="...">로 통째로 embed한다 — 슬라이드마다
    독립된 문서로 렌더링되므로 슬라이드 간 CSS 선택자/id 충돌이 없고, 슬라이드 자신의
    <link rel="stylesheet" href="global.css">(있다면)도 combined.html과 같은 폴더를 기준으로
    상대경로가 정상적으로 풀려서 그대로 적용된다. global.css를 이 함수가 별도로 다시 주입하지
    않는다 — 예전엔 안전하게 하려고 매 슬라이드에 global.css 전체를 <style>로 재주입했었는데,
    그러면 슬라이드 자신의 스타일보다 문서상 나중에 들어가버려서(예: global.css의
    `.slide{background:#fff}` 같은 축약 속성이 슬라이드 자체의 `.slide{background-image:...}`를
    뒤엎어버림) 오히려 슬라이드 고유 스타일을 깨뜨리는 버그가 됐다.

    로컬 이미지 참조(배경 이미지 url(), <img src>)도 원본 그대로 상대경로를 유지한다 —
    srcdoc 안의 상대경로는 이 함수가 만든 output_path(=combined.html)의 위치를 기준으로
    풀리므로, combined.html과 그 이미지 파일들이 항상 같은 폴더에 함께 있어야 렌더링된다
    (호출부가 이미지들도 combined.html과 같이 묶어서 배포/업로드해야 함).
    """
    slides_path = Path(slides_dir)
    slide_files = sorted(slides_path.glob("slide_*.html"))
    if not slide_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    # 브라우저 기본 UA 스타일은 <body>에 margin(보통 8px)을 넣는데, 슬라이드 자체 CSS가 이걸
    # 0으로 재설정하지 않은 경우 실제 렌더링 크기가 iframe의 고정 뷰포트(width x height)보다
    # 살짝 커져서 그 iframe 안에서만 스크롤이 생긴다. 원본 slide_NN.html은 건드리지 않고
    # 합쳐진 문서에서만 명시적으로 0으로 강제해 이 여유분을 없앤다. !important라서 슬라이드
    # 자신의 스타일과 순서/충돌 걱정 없이 항상 이긴다.
    style_tag = "<style>html,body{margin:0 !important;padding:0 !important;}</style>"

    frames = []
    for slide_file in slide_files:
        content = slide_file.read_text(encoding="utf-8")
        if "</head>" in content:
            content = content.replace("</head>", f"{style_tag}</head>", 1)
        else:
            content = style_tag + content
        escaped = html_escape.escape(content, quote=True)
        frames.append(
            f'<iframe class="combined-slide" scrolling="no" srcdoc="{escaped}" '
            f'style="width:{width}px;height:{height}px;border:0;display:block;overflow:hidden;'
            f'margin:0 auto 24px auto;box-shadow:0 2px 10px rgba(0,0,0,.15);"></iframe>'
        )

    combined_html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>Combined Slides</title>\n"
        "<style>html,body{margin:0;padding:0;background:#e5e5e5;}</style>\n"
        "</head>\n<body>\n" + "\n".join(frames) + "\n</body>\n</html>\n"
    )
    Path(output_path).write_text(combined_html, encoding="utf-8")
    return output_path
