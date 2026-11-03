#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/stages/line_prep.cuh"

#include <algorithm>
#include <cmath>
#include <string>

namespace cudaocr {
namespace {

__device__ __forceinline__ float pagePixel(const std::uint8_t* __restrict__ img, size_t pitch,
                                           int w, int h, int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h) return 255.0f;  // off-page padding is paper
    return img[static_cast<size_t>(y) * pitch + x];
}

// One thread per output (x, y) of line blockIdx.z, writing all channels.
// Source coordinates use double: lines can be ~1200 px wide, where float's
// ~1e-4 px resolution times a sharp edge's gradient would exceed the 1e-4
// oracle tolerance. Blending stays in float.
__global__ void linePrepKernel(const std::uint8_t* __restrict__ img, size_t pitch, int pageW,
                               int pageH, const LineCrop* __restrict__ crops,
                               float* __restrict__ out, int channels, int H, int W, float mean,
                               float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    if (x >= W || y >= H) return;
    const LineCrop c = crops[n];

    float v = (1.0f - mean) / scale;  // white
    if (x < c.outW) {
        const double sy = fmin(fmax((y + 0.5) * c.h / H - 0.5, 0.0), c.h - 1.0);
        const double sx = fmin(fmax((x + 0.5) * c.w / c.outW - 0.5, 0.0), c.w - 1.0);
        const int y0 = static_cast<int>(sy), x0 = static_cast<int>(sx);
        const int y1 = min(y0 + 1, c.h - 1), x1 = min(x0 + 1, c.w - 1);
        const float fy = static_cast<float>(sy - y0), fx = static_cast<float>(sx - x0);
        const int px0 = c.x0 + x0, px1 = c.x0 + x1, py0 = c.y0 + y0, py1 = c.y0 + y1;
        const float top = (1 - fx) * pagePixel(img, pitch, pageW, pageH, px0, py0) +
                          fx * pagePixel(img, pitch, pageW, pageH, px1, py0);
        const float bot = (1 - fx) * pagePixel(img, pitch, pageW, pageH, px0, py1) +
                          fx * pagePixel(img, pitch, pageW, pageH, px1, py1);
        v = ((1 - fy) * top + fy * bot) / 255.0f;
        v = (v - mean) / scale;
    }
    const size_t plane = static_cast<size_t>(H) * W;
    float* base = out + static_cast<size_t>(n) * channels * plane + static_cast<size_t>(y) * W + x;
    for (int ch = 0; ch < channels; ++ch) base[ch * plane] = v;
}

Status invalid(const std::string& msg) {
    return Status::Error(ErrorCode::InvalidArgument, "prepareLines: " + msg);
}

}  // namespace

Result<std::vector<LineCrop>> planLineCrops(std::span<const BBox> boxes, int pageW, int pageH,
                                            const LinePrepParams& p) {
    if (pageW <= 0 || pageH <= 0) return invalid("empty page");
    if (p.height <= 0 || p.minWidth <= 0 || p.channels <= 0 || p.padFrac < 0 || p.scale == 0 ||
        (p.maxWidth != 0 && p.maxWidth < p.minWidth))
        return invalid("bad parameters");
    std::vector<LineCrop> crops;
    crops.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        const BBox& b = boxes[i];
        if (b.empty()) return invalid("box " + std::to_string(i) + " is empty");
        if (b.x1 < 0 || b.y1 < 0 || b.x0 >= pageW || b.y0 >= pageH)
            return invalid("box " + std::to_string(i) + " is outside the page");
        const int pad = static_cast<int>(std::lround(p.padFrac * b.height()));
        LineCrop c;
        c.x0 = b.x0 - pad;
        c.y0 = b.y0 - pad;
        c.w = b.width() + 2 * pad;
        c.h = b.height() + 2 * pad;
        c.outW = std::max(p.minWidth,
                          static_cast<int>(std::lround(static_cast<double>(c.w) * p.height / c.h)));
        if (p.maxWidth > 0) c.outW = std::min(c.outW, p.maxWidth);
        crops.push_back(c);
    }
    return crops;
}

Result<LineBatch> prepareLines(const DeviceImage& gray, std::span<const BBox> boxes,
                               const LinePrepParams& p, cudaStream_t stream) {
    if (gray.channels() != 1) return invalid("gray page must be 1-channel");
    auto plan = planLineCrops(boxes, gray.width(), gray.height(), p);
    if (!plan.ok()) return plan.status();

    LineBatch batch;
    batch.crops = plan.take();
    batch.count = static_cast<int>(batch.crops.size());
    batch.channels = p.channels;
    batch.height = p.height;
    if (batch.count == 0) return batch;
    for (const auto& c : batch.crops) batch.width = std::max(batch.width, c.outW);

    auto d_crops = DeviceBuffer<LineCrop>::create(batch.crops.size());
    if (!d_crops.ok()) return d_crops.status();
    Status s = d_crops.value().upload(batch.crops, stream);
    if (!s.ok()) return s;
    auto data = DeviceBuffer<float>::create(static_cast<size_t>(batch.count) * p.channels *
                                            p.height * batch.width);
    if (!data.ok()) return data.status();
    batch.data = data.take();

    const dim3 block(32, 8);
    const dim3 grid((batch.width + block.x - 1) / block.x, (p.height + block.y - 1) / block.y,
                    batch.count);
    linePrepKernel<<<grid, block, 0, stream>>>(
        gray.data(), gray.pitch(), gray.width(), gray.height(), d_crops.value().data(),
        batch.data.data(), p.channels, p.height, batch.width, p.mean, p.scale);
    CUDAOCR_CHECK_LAUNCH("linePrepKernel");
    // d_crops is freed on return; make sure the kernel finished reading it.
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    return batch;
}

}  // namespace cudaocr
