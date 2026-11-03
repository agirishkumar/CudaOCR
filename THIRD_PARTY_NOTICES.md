# Third-party notices

CudaOCR itself is MIT-licensed (see `LICENSE`). Binary distributions (the `.deb`, the Docker
image, `cmake --install`) include or link the following:

| Component | Use | Licence | Shipped as |
|---|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.31.1 | recognizer inference | MIT (© Microsoft); its own third-party notices in `onnxruntime/ThirdPartyNotices.txt` | bundled shared libraries (`lib/cudaocr/`) |
| PP-OCRv5 English recognition model (`en_PP-OCRv5_mobile_rec`), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); ONNX export by [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx) | recognition weights + dictionary | Apache-2.0 | `share/cudaocr/models/` (model card: `README.md` there) |
| [stb_image / stb_image_write](https://github.com/nothings/stb) | image decode/encode | public domain or MIT | compiled in |
| [toml++](https://github.com/marzer/tomlplusplus) 3.4.0 | config files | MIT (© Mark Gillard) | compiled in (header-only) |
| NVIDIA CUDA runtime (`cudart_static`) | GPU runtime | NVIDIA CUDA Toolkit EULA (redistributable component) | statically linked |
| [poppler](https://poppler.freedesktop.org/) (`libpoppler-cpp`) | PDF page rendering | GPL-2.0-or-later | **not bundled**: system package, linked dynamically; binaries built with PDF support are therefore distributed under terms compatible with the GPL |

Not shipped: GoogleTest (tests only), NVIDIA NPP (benchmarks only), cuDNN and the CUDA math
libraries (installed from NVIDIA's repository by the user).
