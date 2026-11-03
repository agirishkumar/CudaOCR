# Pipeline config (TOML)

`cudaocr --config page.toml img.png` loads a pipeline config; command-line flags given
explicitly are applied on top (flags you don't pass never override the file). Start from
the effective config:

```bash
cudaocr --print-config > my.toml          # full current config (defaults + any flags)
cudaocr --config my.toml --print-config   # prints the same file back (round trip)
```

## Format

Each pipeline slot is a component: a `name` registered in `ComponentRegistry`
([architecture.md](architecture.md) §6.3 lists them with their parameters) plus that component's
parameters. Keys you leave out keep their default.

```toml
toGray       = { name = "grayscale" }                      # colour -> gray (3-channel input only)
preprocess   = [ { name = "median", k = 3 }, { name = "polarity" } ]   # shared by both paths
binarizePrep = []                                          # binarizer-only stages
binarizer    = { name = "adaptive", blockSize = 31, C = 40 }
deskewer     = { name = "hough", maxAngle = 15, step = 0.25 }          # or: deskewer = false
detector     = { name = "projection", minLineHeight = 3 }
recognizer   = { name = "onnx", ep = "cuda", maxBatch = 16 }           # or: recognizer = false
decoder      = { name = "ctc-greedy" }
```

- Slots: `toGray`, `preprocess` (array), `binarizePrep` (array), `binarizer`, `deskewer`
  (optional), `detector`, `recognizer` (optional), `decoder`. `[binarizer]`-style tables work
  too.
- Parameter values are numbers, strings or booleans; components validate them when the
  pipeline is created (e.g. `median: k must be 3 or 5, got 4`).
- Recognizer (`onnx`) parameters: `model`, `dict` (default: `$CUDAOCR_MODEL_DIR`, the install's
  `share/cudaocr/models`, or the source tree's `models/`), `ep` (`cuda`/`cpu`), `maxBatch`
  (16), `padFrac` (line-crop padding, 0.1), `maxWidth` (6144), `tf32` (false).
- PDF rendering resolution is a command-line setting (`--pdf-dpi`), not part of the config.
- Errors name the location: `my.toml:3:1: unknown key 'binariser' (known: ...)`.

## Flags vs file

| Flag | Effect on the loaded config |
|---|---|
| `--median N` | sets `k` of the `median` stage in `preprocess` (inserts it first if absent; `0` removes it) |
| `--equalize` | inserts `equalize` first in `preprocess` if absent |
| `--otsu` | `binarizer = { name = "otsu" }` (conflicts with `--block-size`/`--C`) |
| `--block-size N`, `--C N` | set on an `adaptive` binarizer (switches to `adaptive` if needed) |
| `--no-deskew`, `--no-recognize` | `deskewer = false`, `recognizer = false` |
| `--ep`, `--model-dir`, `--tf32` | set on the recognizer (warns if recognition is off) |

Without `--config` the base is the built-in default, which includes the `onnx` recognizer
when the build has ONNX Runtime.
