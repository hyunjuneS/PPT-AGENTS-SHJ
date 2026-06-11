"""HTML slide → PPTX conversion using Playwright + python-pptx."""

import asyncio
import io
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu


# Slide dimensions (16:9 at 96 dpi → EMU)
# 1280 x 720 px  @  96 dpi  → 12192000 x 6858000 EMU  (914400 EMU = 1 inch)
_PX_TO_EMU = 914400 / 96  # 9525 EMU per pixel


def _px_emu(px: int) -> int:
    return int(px * _PX_TO_EMU)


SLIDE_SIZES = {
    "16:9": (1280, 720),
    "4:3":  (960,  720),
    "A4":   (794,  1123),
}


async def render_slide_png(html_path: str, width: int, height: int) -> bytes:
    """Render a single HTML slide to PNG bytes via Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(f"file://{html_path}", wait_until="networkidle")
        png_bytes = await page.screenshot(
            full_page=False,
            clip={"x": 0, "y": 0, "width": width, "height": height},
            type="png",
        )
        await browser.close()
    return png_bytes


async def html_slides_to_pptx(
    slides_dir: str,
    output_path: str,
    aspect_ratio: str = "16:9",
) -> str:
    """
    Convert all slide_*.html files in slides_dir to a single PPTX file.
    Returns the path of the created PPTX.
    """
    slides_dir_path = Path(slides_dir)
    html_files = sorted(slides_dir_path.glob("slide_*.html"))
    if not html_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    w_px, h_px = SLIDE_SIZES.get(aspect_ratio, SLIDE_SIZES["16:9"])

    prs = Presentation()
    prs.slide_width  = Emu(_px_emu(w_px))
    prs.slide_height = Emu(_px_emu(h_px))

    blank_layout = prs.slide_layouts[6]  # completely blank layout

    for html_file in html_files:
        png_bytes = await render_slide_png(str(html_file.resolve()), w_px, h_px)

        slide = prs.slides.add_slide(blank_layout)
        img_stream = io.BytesIO(png_bytes)
        slide.shapes.add_picture(
            img_stream,
            left=Emu(0),
            top=Emu(0),
            width=prs.slide_width,
            height=prs.slide_height,
        )

    prs.save(output_path)
    return output_path
