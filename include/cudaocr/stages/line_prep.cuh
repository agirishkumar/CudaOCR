#pragma once

// Line crop + resize + normalize for CTC line recognizers (models/README.md):
// each line box is padded, cropped from the deskewed gray page, bilinear-resized
// to a fixed height keeping aspect, normalized as x = (v/255 - mean) / scale,
// replicated to `channels`, and right-padded with paper white to the batch width.

#include <span>
#include <vector>

#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/status.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

struct LinePrepParams {
    int height = 48;       // model input height
    float padFrac = 0.1f;  // padding on every side, as a fraction of the box height
                           // (0.10 measured best: see docs/decisions.md #23)
    int minWidth = 8;      // model needs >= 8 px (one CTC timestep per 8 px)
    // Cap on the resized width (0 = none); wider lines are squeezed. Real full-width
    // text lines resize to ~3000-4400 px; thin rules detected as lines reached
    // ~30000 px and made the recognizer's attention run out of GPU memory (M4.6).
    int maxWidth = 6144;
    int channels = 3;   // gray replicated into this many planes
    float mean = 0.5f;  // x = (v / 255 - mean) / scale
    float scale = 0.5f;
};

// Where to sample each line from and how wide its resized image is. Host-only;
// shared by the kernel and the CPU oracle so both use the same geometry.
struct LineCrop {
    int x0, y0;  // crop origin in page pixels (may be negative: padding off-page)
    int w, h;    // crop size (>= 1)
    int outW;    // resized width = clamp(round(w * height / h), minWidth, maxWidth)
};

// InvalidArgument for empty/inverted boxes, boxes entirely off the page, or bad params.
Result<std::vector<LineCrop>> planLineCrops(std::span<const BBox> boxes, int pageW, int pageH,
                                            const LinePrepParams& p);

// Batch tensor, NCHW float32: data[((n * channels + c) * height + y) * width + x].
struct LineBatch {
    DeviceBuffer<float> data;
    int count = 0, channels = 0, height = 0, width = 0;  // width = max outW
    std::vector<LineCrop> crops;                         // per line, in input order
};

// gray: 1-channel deskewed page. Pixels sampled outside the page read as 255
// (paper); columns x >= crops[n].outW are filled with the normalized white value.
// Empty `boxes` -> an empty batch (count 0, no allocation).
Result<LineBatch> prepareLines(const DeviceImage& gray, std::span<const BBox> boxes,
                               const LinePrepParams& p, cudaStream_t stream = nullptr);

}  // namespace cudaocr
