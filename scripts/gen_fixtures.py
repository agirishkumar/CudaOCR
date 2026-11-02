#!/usr/bin/env python3
"""Generate synthetic OCR fixtures with exact ground truth.

Renders seeded, word-wrapped English-like text at 300 DPI with system fonts and
writes, per fixture:
    <name>.png      8-bit grayscale image
    <name>.gt.txt   the exact text, one line per rendered text line
plus manifest.tsv (name, font, size_pt, variant, angle_deg, lines).

Output is byte-deterministic for a given --seed and font files (Pillow's BASIC
layout engine is forced so raqm/HarfBuzz versions don't matter). Text uses only
printable ASCII so every character is in an English recognizer's dictionary.

Usage:
    scripts/gen_fixtures.py --out tests/fixtures/synth          # committed CI set
    scripts/gen_fixtures.py --out /tmp/synth --set large --seed 7
    scripts/gen_fixtures.py --check tests/fixtures/synth        # regenerate + compare
"""

from __future__ import annotations

import argparse
import filecmp
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

DPI = 300
FONT_DIR = Path("/usr/share/fonts/truetype")
FONTS = {
    "sans": "dejavu/DejaVuSans.ttf",
    "serif": "liberation/LiberationSerif-Regular.ttf",
    "mono": "liberation/LiberationMono-Regular.ttf",
    "sans2": "liberation/LiberationSans-Regular.ttf",
    "serif2": "dejavu/DejaVuSerif.ttf",
}

# Common English words (no source text is reproduced; sentences are random).
WORDS = """
the of and to in is was for on are with as be at by this have from or one had
not but what all were when we there can an your which their said if do will
each about how up out them then she many some so these would other into has
more her two like him see time could no make than first been its who now people
my made over did down only way find use may water long little very after words
called just where most know get through back much before go good new write our
used me man too any day same right look think also around another came come work
three word must because does part even place well such here take why things help
put years different away again off went old number great tell men say small
every found still between name should home big give air line set own under read
last never us left end along while might next sound below saw something thought
both few those always looked show large often together asked house world going
want school important until form food keep children feet land side without boy
once animals life enough took sometimes four head above kind began almost live
page got earth need far hand high year mother light parts country father let
night following picture being study second eyes soon times story boys since white
days ever paper hard near sentence better best across during today others however
sure means knew try told young miles sun ways thing whole hear example heard
several change answer room sea against top turned learn point city play toward
five using himself usually money seen car morning river table system data report
""".split()

PANGRAMS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs, 1234567890.",
    "Sphinx of black quartz, judge my vow: (A-Z) & [0-9]!",
]


@dataclass(frozen=True)
class Spec:
    name: str
    font: str
    size_pt: float
    variant: str  # clean | skew | noise | inverted
    angle: float = 0.0  # degrees, counter-clockwise (skew variant)
    lines: int = 8  # text lines for pages; ignored for single-line fixtures
    text: str | None = None  # fixed single-line text instead of random paragraphs


def ci_set() -> list[Spec]:
    """Small committed set (keep the directory under ~2 MB)."""
    return [
        Spec("page_sans_clean", "sans", 11, "clean"),
        Spec("page_serif_clean", "serif", 11, "clean"),
        Spec("page_mono_clean", "mono", 10, "clean"),
        Spec("page_sans_skew_p3", "sans", 11, "skew", angle=3.0),
        Spec("page_serif_skew_m4", "serif", 11, "skew", angle=-4.0),
        Spec("page_sans_noise", "sans", 12, "noise", lines=6),
        Spec("page_serif_inverted", "serif", 11, "inverted"),
        Spec("line_pangram_sans", "sans", 12, "clean", text=PANGRAMS[0]),
        Spec("line_digits_serif", "serif", 12, "clean", text=PANGRAMS[1]),
        Spec("line_punct_mono", "mono", 12, "clean", text=PANGRAMS[2]),
    ]


def large_set() -> list[Spec]:
    """Bigger on-demand set for accuracy runs (not committed)."""
    specs = []
    for font in FONTS:
        for size in (10, 12):
            specs.append(Spec(f"page_{font}_{size}_clean", font, size, "clean", lines=20))
            specs.append(Spec(f"page_{font}_{size}_skew_p2", font, size, "skew", 2.0, 20))
            specs.append(Spec(f"page_{font}_{size}_skew_m5", font, size, "skew", -5.0, 20))
            specs.append(Spec(f"page_{font}_{size}_noise", font, size, "noise", lines=20))
            specs.append(Spec(f"page_{font}_{size}_inverted", font, size, "inverted", lines=20))
    return specs


def random_sentence(rng: random.Random) -> str:
    n = rng.randint(5, 12)
    words = [rng.choice(WORDS) for _ in range(n)]
    for i in range(len(words)):
        r = rng.random()
        if r < 0.06:
            words[i] = str(rng.randint(0, 9999))
        elif r < 0.10:
            words[i] = words[i].upper()
    words[0] = words[0][0].upper() + words[0][1:]
    if rng.random() < 0.25 and n > 4:
        k = rng.randint(1, n - 2)
        words[k] += rng.choice([",", ";", ":"])
    return " ".join(words) + rng.choice([".", ".", ".", "?", "!"])


def wrap(rng: random.Random, font: ImageFont.FreeTypeFont, width: int, nlines: int) -> list[str]:
    """Greedy word wrap of random sentences into exactly `nlines` lines."""
    lines: list[str] = []
    cur = ""
    while len(lines) < nlines:
        for word in random_sentence(rng).split():
            cand = f"{cur} {word}" if cur else word
            if font.getlength(cand) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = word
                if len(lines) == nlines:
                    break
    return lines


def render(spec: Spec, seed: int, font_dir: Path) -> tuple[Image.Image, list[str]]:
    rng = random.Random(f"{seed}:{spec.name}")  # independent stream per fixture
    px = round(spec.size_pt * DPI / 72)
    font = ImageFont.truetype(str(font_dir / FONTS[spec.font]), px,
                              layout_engine=ImageFont.Layout.BASIC)
    margin = px
    if spec.text is not None:
        lines = [spec.text]
        width = int(font.getlength(spec.text)) + 2 * margin
    else:
        width = 1240  # ~4.1 in at 300 DPI: half a page keeps PNGs small
        lines = wrap(rng, font, width - 2 * margin, spec.lines)
    ascent, descent = font.getmetrics()
    pitch = round(1.5 * px)
    height = 2 * margin + (len(lines) - 1) * pitch + ascent + descent

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((margin, margin + i * pitch), text, font=font, fill=0)

    if spec.variant == "skew":
        img = img.rotate(spec.angle, resample=Image.Resampling.BICUBIC, expand=True,
                         fillcolor=255)
    elif spec.variant == "noise":
        nrng = np.random.default_rng(rng.getrandbits(64))
        a = np.asarray(img, dtype=np.float32) + nrng.normal(0.0, 25.0, (height, width))
        img = Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8), "L")
    elif spec.variant == "inverted":
        img = ImageOps.invert(img)
    elif spec.variant != "clean":
        raise ValueError(f"unknown variant {spec.variant}")
    return img, lines


def generate(out: Path, specs: list[Spec], seed: int, font_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = ["name\tfont\tsize_pt\tvariant\tangle_deg\tlines"]
    for spec in specs:
        img, lines = render(spec, seed, font_dir)
        img.save(out / f"{spec.name}.png", optimize=False)  # no time/text chunks
        (out / f"{spec.name}.gt.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        rows.append(f"{spec.name}\t{spec.font}\t{spec.size_pt:g}\t{spec.variant}\t"
                    f"{spec.angle:g}\t{len(lines)}")
    (out / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")


def check(committed: Path, specs: list[Spec], seed: int, font_dir: Path) -> int:
    """Regenerate into a temp dir and byte-compare with `committed`."""
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp), specs, seed, font_dir)
        names = sorted(p.name for p in Path(tmp).iterdir())
        _, mismatch, errors = filecmp.cmpfiles(committed, tmp, names, shallow=False)
        extra = sorted({p.name for p in committed.iterdir()} - set(names) - {"README.md"})
    for name in mismatch + errors + extra:
        print(f"differs: {name}", file=sys.stderr)
    if mismatch or errors or extra:
        return 1
    print(f"ok: {len(names)} files in {committed} match seed {seed}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="output directory")
    ap.add_argument("--check", type=Path, help="verify a committed directory regenerates exactly")
    ap.add_argument("--set", choices=["ci", "large"], default="ci")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--font-dir", type=Path, default=FONT_DIR)
    args = ap.parse_args()

    missing = [f for f in FONTS.values() if not (args.font_dir / f).exists()]
    if missing:
        print(f"SKIP: fonts not found under {args.font_dir}: {missing}", file=sys.stderr)
        return 77  # ctest SKIP_RETURN_CODE
    specs = ci_set() if args.set == "ci" else large_set()
    if args.check:
        return check(args.check, specs, args.seed, args.font_dir)
    if not args.out:
        ap.error("--out or --check is required")
    generate(args.out, specs, args.seed, args.font_dir)
    print(f"wrote {len(specs)} fixtures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
