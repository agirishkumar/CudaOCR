# CudaOCR

A GPU-first OCR engine written from scratch in CUDA/C++. Every pipeline stage —
grayscale, denoising, binarization, deskew, and segmentation — runs as a
hand-written CUDA kernel, each verified byte-for-byte (or within a documented
tolerance) against a CPU oracle by the test suite. Libraries (NPP) appear only
as benchmark baselines.

See [product.md](product.md) for the vision/roadmap, [architecture.md](architecture.md)
for design decisions, and [BACKLOG.md](BACKLOG.md) for the task queue.

## Pipeline

```
image ─► grayscale ─► median denoise ─► polarity normalize ─► adaptive/Otsu
      binarization ─► Hough skew detect + GPU rotate ─► line/word/char
      segmentation ─► JSON + overlay output        (recognition: roadmap M3/M4)
```

## Requirements

- CUDA Toolkit 12.4+ (developed on 13.1), GPU sm_75+ (default arch: 86)
- CMake >= 3.24, a C++20 host compiler (g++-13)
- No other system dependencies: image I/O is vendored (stb), GoogleTest is
  fetched at configure time.

## Build & test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # 59 tests, all GPU kernels vs CPU oracles
./scripts/verify.sh                          # full gate: build + tests + memcheck
```

## Run

```bash
./build/cudaocr resources/sample.png --benchmark -o out.json --dump-stages stages/
```

`--dump-stages` writes the intermediate images plus an overlay with detected
lines (blue), words (red), and characters (green). See `./build/cudaocr --help`
for thresholding/deskew options.

Example on `resources/sample.png` (699x921 scan, RTX 3060 Laptop):
35 lines, 348 words, 903 characters in **4.2 ms** of GPU pipeline time.

## Benchmarks (`./build/bench`, 2500x3500, RTX 3060 Laptop, vs NPP)

| kernel | custom | NPP | ratio |
|---|---|---|---|
| grayscale | 0.137 ms | 0.113 ms | 1.22x |
| gaussian 5x5 | 0.792 ms | 0.288 ms | 2.75x |
| median 5x5 | 19.6 ms | 0.715 ms | 27x (optimization task filed) |
| box mean 31x31 | 2.45 ms | 19.0 ms | **0.13x — 7.8x faster than NPP** |

## License

MIT — see [LICENSE](LICENSE).
