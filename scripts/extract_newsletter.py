#!/usr/bin/env python3
"""
Extract a newsletter PDF's text and images for the /deadline-brief skill.

PDFs are untrusted input. Run this with the interpreter in isolated mode and
write into a fresh, empty directory outside the repo, so nothing in the PDF's
neighbourhood can shadow a module:

    .venv/bin/python -I scripts/extract_newsletter.py <pdf> <empty out dir>

Writes text.txt (page-delimited) and every image at least 400px wide as PNG —
the Pinnacle fixture table usually arrives as an image, not text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf  # requirements-local.txt


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        sys.exit(f"{out} is not empty — extract into a fresh directory")
    doc = pymupdf.open(src)
    with open(out / "text.txt", "w") as fh:
        for i, page in enumerate(doc, start=1):
            fh.write(f"\n===== PAGE {i} =====\n{page.get_text()}")
    n = 0
    for i, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            pix = pymupdf.Pixmap(doc, img[0])
            if pix.n - pix.alpha >= 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            if pix.width >= 400:
                n += 1
                pix.save(out / f"p{i:02d}_img{n:02d}_{pix.width}x{pix.height}.png")
    print(f"pages={len(doc)} images={n} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
