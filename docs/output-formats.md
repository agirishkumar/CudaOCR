# Output formats

`cudaocr` writes one of these per page (`--format`, or inferred from `-o`'s extension).

## `txt`

Recognized line texts in reading order, one per line, each ending in `\n`; lines with no
recognized text are skipped. UTF-8.

## `json` — schema `cudaocr.page/1`

```json
{
  "schema": "cudaocr.page/1",
  "width": 1252, "height": 159,          // input image size (px)
  "skewDeg": 0,                          // detected skew; boxes are in the deskewed frame
  "inverted": false,                     // page was light-on-dark and got normalized
  "text": "The quick brown fox ...\n",    // same as the txt output
  "metrics": {"otsu.threshold": 127},    // component-reported scalars (may be empty)
  "timingsMs": {"median": 0.24, "onnx": 346.3, ...},  // per stage, pipeline order
  "lines": [
    {"bbox": [50, 59, 1197, 106], "conf": 0.989, "text": "The quick ...", "words": [
      {"bbox": [59, 59, 136, 106], "conf": 0.99998, "text": "The"},
      ...
    ]}
  ]
}
```

| Field | Meaning |
|---|---|
| `bbox` | `[x0, y0, x1, y1]`, **inclusive** pixel coordinates in the **deskewed** page (same size as the input; rotate by `skewDeg` to map back). `null` when unknown. |
| `conf` | Mean probability of the emitted CTC symbols, in [0, 1]; 0 when nothing was recognized. |
| line `text` | Recognized text of the line (empty with `--no-recognize`). |
| `words` | With recognition: words split at spaces; boxes from CTC timesteps (half-step padded; on clean pages they match glyph-gap boxes with mean IoU ≈ 0.93). Detection only: the detector's gap-based word boxes, text empty, plus `glyphs`. |
| `glyphs` | Optional, detection only: per-glyph boxes from column projections (debug quality). |

Strings are JSON-escaped (`\"`, `\\`, `\n`, `\t`, `\r`, `\u00XX` for other control
characters); other UTF-8 passes through. Numbers use the shortest round-trip form. Layout
is deterministic (one word per line) so outputs diff cleanly. Breaking changes bump the
schema suffix.

## `hocr`

[hOCR 1.2](http://kba.github.io/hocr-spec/1.2/) XHTML (well-formed XML, UTF-8), e.g. for
searchable-PDF tools:

```html
<div class="ocr_page" id="page_1" title="bbox 0 0 1252 159">
 <span class="ocr_line" id="line_1_1" title="bbox 50 59 1198 107; x_wconf 99">
  <span class="ocrx_word" id="word_1_1_1" title="bbox 59 59 137 107; x_wconf 100">The</span>
  ...
```

- `bbox` uses **exclusive** right/bottom (`x1 + 1`, `y1 + 1` of the JSON's inclusive box),
  the convention Tesseract writes; coordinates are in the deskewed frame like the JSON.
- `x_wconf` = round(conf × 100).
- Words with unknown geometry omit `bbox`; a recognized line without words carries its text
  directly. Text is XML-escaped (`& < > " '`).
