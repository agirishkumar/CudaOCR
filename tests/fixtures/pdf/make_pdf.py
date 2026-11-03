#!/usr/bin/env python3
"""Write two_pages.pdf: a 2-page "scanned" PDF made from the synthetic fixtures
page_sans_clean.png and line_pangram_sans.png, embedded at 300 dpi.

Pillow embeds the gray pages as JPEG (as many scanners do), so the PDF tests
compare against the source PNGs with a tolerance; regenerate with
    python3 tests/fixtures/pdf/make_pdf.py
"""

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synth"
PAGES = ["page_sans_clean", "line_pangram_sans"]

images = [Image.open(SYNTH / f"{p}.png").convert("L") for p in PAGES]
images[0].save(HERE / "two_pages.pdf", save_all=True, append_images=images[1:], resolution=300.0)
(HERE / "two_pages.gt.txt").write_text(
    "".join((SYNTH / f"{p}.gt.txt").read_text() for p in PAGES))
print(f"wrote {HERE / 'two_pages.pdf'}")
