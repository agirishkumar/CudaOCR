# CudaOCR

A GPU-first OCR engine for printed documents, written in CUDA/C++. Image preprocessing,
deskew, line detection and line cropping run as hand-written CUDA kernels — each verified
against a CPU reference implementation by the test suite — and text recognition runs the
PP-OCRv5 English model through ONNX Runtime on the GPU. One command turns scans, images or
scanned PDFs into text, JSON or hOCR.

```bash
./build/cudaocr scan.pdf -o out/            # out/scan-p001.txt, out/scan-p002.txt, ...
```

**Scope:** printed, mostly single-column documents (scans, screenshots, PDFs of scans).
Not handwriting, scene text, or layout understanding (tables, forms). Goal: a functional,
modular, well-tested OCR whose parts can be swapped or optimized — not a state-of-the-art
competitor. Details: [docs/product.md](docs/product.md) · design:
[docs/architecture.md](docs/architecture.md) · decisions: [docs/decisions.md](docs/decisions.md) ·
open work: [docs/backlog.md](docs/backlog.md).

## Features

- **Inputs:** PNG, JPEG, BMP, TGA, GIF, PNM, **PDF** (each page rendered at 300 dpi, poppler),
  any number of files and directories in one run.
- **Outputs:** plain text; **JSON** with line/word boxes and confidences
  ([schema](docs/output-formats.md)); **hOCR** for searchable-PDF tools.
- **GPU pipeline:** grayscale → median denoise → polarity normalization (light-on-dark pages)
  → adaptive (or Otsu) binarization → Hough skew detection + rotation → projection-profile
  line detection → line crop/resize kernel → PP-OCRv5 recognizer (CUDA or CPU) → CTC decoding
  → word boxes from CTC timesteps.
- **Configurable and modular:** every stage is a named component; a TOML file
  ([docs/config.md](docs/config.md)) or flags choose components and parameters.
- **Batch mode:** one model load for all pages (~6× faster than one process per page).
- **Measured:** CER/WER harness with synthetic, real and UNLV/ISRI test sets and accuracy
  ratchets in the test suite.

## Pipeline

```
image / PDF page ─► grayscale ─► median 3×3 ─► polarity ─┬─► adaptive threshold ─► Hough deskew ─► line detection
                    (all custom CUDA kernels)            │                          (angle)       (row/column profiles)
                                                         └─► deskewed gray ◄──────────┘                 │ line boxes
                                                               │                                        ▼
                     txt / JSON / hOCR ◄─ words + boxes ◄─ CTC decode ◄─ PP-OCRv5 (ONNX Runtime) ◄─ crop+resize to 48 px
```

## Requirements

- NVIDIA GPU, sm_75 or newer (default build target sm_86), CUDA Toolkit 13.x, CMake ≥ 3.24,
  a C++20 compiler (developed with g++ 15, CUDA 13.4, RTX 3060 Laptop).
- **cuDNN 9 for CUDA 13** for GPU recognition: `sudo apt install cudnn9-cuda-13`.
- **PDF input (optional):** `sudo apt install libpoppler-cpp-dev` before configuring. Without
  it everything else works and PDFs report "built without PDF support".
- Fetched automatically at configure time: ONNX Runtime 1.31.1 (CUDA 13 build, 233 MB),
  GoogleTest, toml++ — all SHA-256 pinned. Image I/O (stb) is vendored.
- Python 3 + Pillow only for the developer scripts (fixture generation, UNLV preparation).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/fetch_models.sh                  # recognition model (7.8 MB) into models/
```

Options: `-DCUDAOCR_WITH_ORT=OFF` (no recognizer, detection only), `-DCUDAOCR_ORT_ROOT=<dir>`
(reuse an extracted ONNX Runtime, e.g. `build/_deps/onnxruntime-src`),
`-DCUDAOCR_WITH_PDF=OFF`, `-DCMAKE_CUDA_ARCHITECTURES=<sm>`.

Install a self-contained copy (binaries, the ONNX Runtime libraries and the fetched model):

```bash
cmake --install build --prefix ~/.local
```

## Usage

```bash
./build/cudaocr page.png                          # text on stdout
./build/cudaocr page.png -o page.json             # lines + words with boxes and confidences
./build/cudaocr page.png -o page.hocr             # hOCR
./build/cudaocr scan.pdf -o out/                  # one file per PDF page
./build/cudaocr scans/ more.png -o out/ --format json   # many inputs, one model load
./build/cudaocr scans/ > all.txt                  # pages separated by form feeds
./build/cudaocr page.png -v --benchmark --dump-stages stages/   # summary, timings, overlays
./build/cudaocr page.png --no-recognize -o boxes.json           # detection only, no model
```

`--dump-stages` writes the gray, binary and deskewed images plus `04_overlay.png` with the
detected lines (blue) and words (red). Exit codes (worst page wins): 0 ok, 1 bad input or
config, 2 GPU/runtime error, 3 a page without text. All flags: `./build/cudaocr --help`.

**Configuration** — print the effective config, edit it, load it (explicit flags still
override the file):

```bash
./build/cudaocr --print-config > my.toml
./build/cudaocr --config my.toml page.png
```

**Tuning tips:** `--C 50` helps stained or blotchy scans (higher loses faint print);
`--pdf-dpi 400` for very small print in PDFs; `--ep cpu` runs without a GPU recognizer
(about 8× slower).

## Results

Accuracy from `scripts/accuracy_report.sh --large` with default settings
(per-file tables: [benchmarks/ACCURACY.md](benchmarks/ACCURACY.md)); lower is better:

| Test set | Pages | CER | WER |
|---|---:|---:|---:|
| Synthetic pages (clean, skewed, noisy, inverted; generated) | 50 | 0.21 % | 1.00 % |
| Clean typeset book page (`resources/sherlock.png`) | 1 | 0.94 % | 4.23 % |
| Real scans in `resources/` (book page, coffee-stained, crumpled) | 3 | 4.75 % | 9.03 % |
| UNLV/ISRI business letters + DOE reports, 300 dpi — zoned\* | 40 | 6.84 % | 15.10 % |
| UNLV/ISRI, same pages — full page | 40 | 19.64 % | 29.94 % |

\*Zoned: each ground-truth text zone OCRed separately (the ISRI protocol); full page adds our
own layout and reading order, which assumes one text column — that gap is the main open
work. UNLV data: `scripts/fetch_unlv.sh` (43 MB, not redistributable).

Speed (RTX 3060 Laptop, model warm, dense 33-line page `sherlock.png`): GPU preprocessing,
deskew and line detection **≈ 3 ms**; recognition **≈ 370 ms** (the bottleneck; first page
per process adds ≈ 0.5 s of model/cuDNN start-up). Custom kernels vs NVIDIA NPP on a
2500×3500 image (`./build/bench`, CUDA 13.4):

| Kernel | Custom | NPP | |
|---|---:|---:|---|
| grayscale | 0.137 ms | 0.112 ms | 1.2× NPP |
| box mean 31×31 | 2.44 ms | 19.5 ms | **8× faster** |
| Gaussian 5×5 | 0.76 ms | 0.29 ms | 2.6× NPP |
| median 5×5 | 19.2 ms | 0.72 ms | 27× NPP (the pipeline uses 3×3: 0.2 ms) |

## Known limitations

- **Multi-column pages, tables, headers and figures:** projection-profile layout reads one
  column top to bottom (UNLV full page 19.6 % vs 6.8 % zoned).
- **Heavy stains and speckle noise:** the coffee-stained sample scores ≈ 30 % CER; isolated
  specks can be read as characters.
- **Throughput:** ≈ 0.3–0.4 s per dense page, dominated by the recognizer.
- English/Latin script only (the model's 436-character dictionary).

## Development

```bash
ctest --test-dir build --output-on-failure      # ~200 tests (model-dependent ones skip if absent)
./scripts/verify.sh                             # gate before every PR: build, tests, memcheck, format
./scripts/accuracy_report.sh [--large]          # regenerate benchmarks/ACCURACY.md
./scripts/fetch_unlv.sh                         # optional real-scan set for the report and ratchet
./build/cudaocr-eval <pred_dir> <gt_dir>        # CER/WER of <stem>.txt vs <stem>.gt.txt
./build/bench                                   # custom kernels vs NPP
```

Every GPU kernel lands with a CPU oracle and a comparison test (byte-exact for integer
kernels), every fixed bug with a named regression test, and accuracy changes are guarded by
CER ratchets — see [docs/architecture.md](docs/architecture.md) §8.

| Path | Contents |
|---|---|
| `include/cudaocr/`, `src/` | library: `core/` (RAII GPU memory, streams, status, PDF), `stages/` (CUDA kernels), `components/` (pluggable stages, recognizer, formatters), `pipeline`, `config`, `eval/` |
| `apps/` | `cudaocr` CLI, `cudaocr-eval` |
| `tests/` | unit/integration tests, CPU oracles, fixtures (synthetic pages, real-scan ground truth, PDF) |
| `scripts/` | `verify.sh`, `fetch_models.sh`, `fetch_unlv.sh`, `gen_fixtures.py`, `accuracy_report.sh`, `onnx_info.py` |
| `docs/` | [product](docs/product.md), [architecture](docs/architecture.md), [decisions](docs/decisions.md), [backlog](docs/backlog.md), [config](docs/config.md), [output formats](docs/output-formats.md) |
| `resources/`, `models/` | sample pages with [sources and licences](resources/README.md); [model card](models/README.md) |

## License

MIT — see [LICENSE](LICENSE). The PP-OCRv5 model weights (fetched, not bundled) are
Apache-2.0; poppler (optional PDF input) is GPL-2.0-or-later.
