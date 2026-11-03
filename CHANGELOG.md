# Changelog

## 0.1.0 — 2026-11-02

First release: end-to-end GPU OCR for printed documents.

- **Pipeline:** custom CUDA kernels for grayscale, median/Gaussian denoising, polarity
  normalization, adaptive/Otsu binarization, Hough deskew + rotation, projection-profile line
  and word detection and line cropping; PP-OCRv5 English recognition via ONNX Runtime 1.31.1
  (CUDA or CPU); CTC decoding with word boxes from timesteps.
- **Input:** PNG/JPEG/BMP/TGA/GIF/PNM, scanned PDFs (poppler), directories, any number of
  pages per run with one model load.
- **Output:** text, JSON (`cudaocr.page/1`, boxes + confidences), hOCR.
- **Configuration:** TOML pipeline config with `--print-config`; every stage is a named,
  replaceable component.
- **Accuracy (CER):** 0.21 % on 50 synthetic pages, 0.94 % on a clean book page, 4.75 % on
  real sample scans, 6.84 % on UNLV/ISRI pages zoned (19.64 % full page — multi-column layout
  is the main known gap).
- **Packages:** `.deb` for Ubuntu 24.04 and 26.04 (sm_75–sm_120, ONNX Runtime and model
  bundled, static CUDA runtime), Dockerfile (CUDA 13.4 + cuDNN 9).
