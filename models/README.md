# Model card — line recognizer (M3)

Files in this directory are **downloaded, not committed** (see `.gitignore`):

```bash
scripts/fetch_models.sh            # fetches + SHA-256-verifies into models/
scripts/onnx_info.py models/rec.onnx --markdown   # regenerate the op table below
```

| File | Size | SHA-256 |
|---|---:|---|
| `rec.onnx` | 7,830,888 B | `4e16deb22c4da6468bdca539b2cd3c8687825538b67109177c47d359ab994cd7` |
| `dict.txt` | 1,416 B | `e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6` |

## Provenance

- Model: **en_PP-OCRv5_mobile_rec** (PaddleOCR PP-OCRv5, English), trained by PaddlePaddle.
- ONNX export: Hugging Face [`monkt/paddleocr-onnx`](https://huggingface.co/monkt/paddleocr-onnx),
  `languages/english/`, pinned to revision `7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65`.
- License: **Apache-2.0** (repo metadata; PaddleOCR weights are Apache-2.0).
- The repo's `config.json` says input height 32 — **wrong for this file**; the ONNX graph
  declares 48 and the tests below confirm 48.

## I/O contract (asserted by `tests/unit/rec_model_test.cpp`)

| | Name | Shape | Type |
|---|---|---|---|
| input | `x` | `[N, 3, 48, W]` (N, W dynamic) | float32 |
| output | `fetch_name_0` | `[N, T, 438]`, **T = ⌊W / 8⌋** | float32 |

**Preprocessing** (what the M3.6 line-crop kernel must produce):
1. Crop the line **tightly** to its text (the detector's line box + a few px). With generous
   margins the glyphs shrink below what the model reads — a crop with one-font-size margins
   decoded as empty.
2. Bilinear-resize to height **48**, keeping the aspect ratio (W = round(w · 48 / h), ≥ 8).
3. Replicate gray to **3 channels** (the model was trained on color; channel order is
   irrelevant for gray input).
4. Normalize: `x = (v / 255 − 0.5) / 0.5` → [−1, 1] (white paper = +1).
5. *(Our choice, not a model fact)* batched lines are right-padded to a common W with +1
   (white); to be validated against unpadded inference in M3.8.

**Output** is already **softmax-normalized** (each timestep sums to 1 within 1e-4), so the
recognizer emits probabilities as the `LineRecognizer` contract requires.

**Classes (438)** — CTC layout used by PaddleOCR:

| index | symbol |
|---|---|
| 0 | CTC blank |
| 1 … 436 | `dict.txt` lines 1 … 436, in order (digits, ASCII letters, punctuation, Latin accents, symbols) |
| 437 | space `" "` (not in `dict.txt`; appended by PaddleOCR's `use_space_char`) |

**Verified end to end:** with exactly this preprocessing on the CPU and greedy CTC decoding,
all three M3.1 single-line fixtures (sans, serif, mono; digits, brackets, `&`, punctuation)
are read **exactly**.

## Graph (sizes the M5 custom-kernel work)

- IR 7, opset 14; **1,941,810 parameters** (all stored as `Constant` nodes, no initializers).
- 746 nodes, 25 op types (table below). Read from the op mix, not from source: a conv
  backbone (Conv, BatchNormalization, HardSwish/HardSigmoid, Sigmoid, pooling), attention-style
  blocks (MatMul, Softmax, Transpose/Reshape) and LayerNorm decomposed into
  ReduceMean/Sub/Pow/Sqrt/Div. For M5 this means: implicit-GEMM conv, GEMM, softmax,
  layernorm and a handful of elementwise kernels; 342 `Constant` nodes are just weights.

| op | count |
|---|---:|
| Constant | 342 |
| Add | 112 |
| Mul | 72 |
| Reshape | 50 |
| Conv | 38 |
| HardSwish | 28 |
| MatMul | 13 |
| ReduceMean | 10 |
| Squeeze | 9 |
| Transpose | 9 |
| Slice | 8 |
| Sigmoid | 7 |
| BatchNormalization | 6 |
| Unsqueeze | 5 |
| Sub | 5 |
| Pow | 5 |
| Sqrt | 5 |
| Div | 5 |
| Shape | 4 |
| Concat | 3 |
| Softmax | 3 |
| GlobalAveragePool | 2 |
| Relu | 2 |
| HardSigmoid | 2 |
| AveragePool | 1 |
