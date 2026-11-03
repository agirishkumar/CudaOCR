#!/usr/bin/env python3
"""Prepare a seeded UNLV/ISRI sample for cudaocr evaluation (called by fetch_unlv.sh).

For each sampled page <id> (e.g. 8500_001) writes:
  page/<id>.png      the 300 dpi G4 TIFF as 8-bit gray PNG (stb cannot read TIFF)
  page/<id>.gt.txt   the UNLV ground truth (concatenation of its zones' text)
  zones/<id>__zNN.png one crop per zone of <id>.uzn (10 px margin), in zone order
  zones/<id>.zones   the crop stems in order, one per line
Sampling is stratified: an equal share from each set, seeded, sorted by id.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

MARGIN = 10


def pages_of(raw: Path, set_name: str) -> list[tuple[str, Path]]:
    """(page id, directory) from the set's `pages` index ("8500_001.3B 0")."""
    out = []
    for line in (raw / set_name / "pages").read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            out.append((parts[0].removesuffix(".3B"), raw / set_name / parts[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_set = max(1, args.sample // len(args.sets))
    chosen = []
    for s in args.sets:
        pages = [p for p in pages_of(args.raw, s)
                 if (p[1] / f"{p[0]}.3B.tif").exists() and (p[1] / f"{p[0]}.3B.uzn").exists()]
        chosen += rng.sample(pages, min(per_set, len(pages)))
    chosen.sort()

    for sub in ("page", "zones"):
        shutil.rmtree(args.out / sub, ignore_errors=True)
        (args.out / sub).mkdir(parents=True)
    zones_total = 0
    for pid, d in chosen:
        img = Image.open(d / f"{pid}.3B.tif").convert("L")
        img.save(args.out / "page" / f"{pid}.png")
        shutil.copy(d / f"{pid}.3B.txt", args.out / "page" / f"{pid}.gt.txt")
        stems = []
        for k, line in enumerate((d / f"{pid}.3B.uzn").read_text(errors="replace").split("\n")):
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, w, h = map(int, parts[:4])
            box = (max(0, x - MARGIN), max(0, y - MARGIN),
                   min(img.width, x + w + MARGIN), min(img.height, y + h + MARGIN))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            stem = f"{pid}__z{k:02d}"
            img.crop(box).save(args.out / "zones" / f"{stem}.png")
            stems.append(stem)
        (args.out / "zones" / f"{pid}.zones").write_text("\n".join(stems) + "\n")
        zones_total += len(stems)
    (args.out / "sample.txt").write_text(
        f"{len(chosen)} pages ({zones_total} zones), seed {args.seed}\n")
    print(f"prepared {len(chosen)} pages ({zones_total} zones) from {', '.join(args.sets)} "
          f"into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
