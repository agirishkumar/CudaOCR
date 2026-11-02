#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/segment.cuh"

#include <algorithm>

namespace cudaocr {
namespace {

__global__ void rowProfileKernel(const std::uint8_t* __restrict__ img, size_t pitch, int w, int h,
                                 int* __restrict__ rows) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (img[static_cast<size_t>(y) * pitch + x] < 128) atomicAdd(&rows[y], 1);
}

__global__ void colProfileKernel(const std::uint8_t* __restrict__ img, size_t pitch, int w, int y0,
                                 int y1, int* __restrict__ cols) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y > y1) return;
    if (img[static_cast<size_t>(y) * pitch + x] < 128) atomicAdd(&cols[x], 1);
}

}  // namespace

Result<std::vector<int>> rowProfile(const DeviceImage& bin, cudaStream_t stream) {
    if (bin.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "rowProfile: 1-channel only");
    auto d_rows = DeviceBuffer<int>::create(bin.height());
    if (!d_rows.ok()) return d_rows.status();
    Status s = d_rows.value().zero();
    if (!s.ok()) return s;
    const dim3 block(32, 8);
    const dim3 grid((bin.width() + block.x - 1) / block.x, (bin.height() + block.y - 1) / block.y);
    rowProfileKernel<<<grid, block, 0, stream>>>(bin.data(), bin.pitch(), bin.width(), bin.height(),
                                                 d_rows.value().data());
    CUDAOCR_CHECK_R(cudaGetLastError());
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    std::vector<int> rows;
    s = d_rows.value().download(rows);
    if (!s.ok()) return s;
    return rows;
}

Result<std::vector<int>> colProfile(const DeviceImage& bin, int y0, int y1, cudaStream_t stream) {
    if (bin.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, "colProfile: 1-channel only");
    if (y0 < 0 || y1 >= bin.height() || y0 > y1)
        return Status::Error(ErrorCode::InvalidArgument, "colProfile: bad row band");
    auto d_cols = DeviceBuffer<int>::create(bin.width());
    if (!d_cols.ok()) return d_cols.status();
    Status s = d_cols.value().zero();
    if (!s.ok()) return s;
    const int bandH = y1 - y0 + 1;
    const dim3 block(32, 8);
    const dim3 grid((bin.width() + block.x - 1) / block.x, (bandH + block.y - 1) / block.y);
    colProfileKernel<<<grid, block, 0, stream>>>(bin.data(), bin.pitch(), bin.width(), y0, y1,
                                                 d_cols.value().data());
    CUDAOCR_CHECK_R(cudaGetLastError());
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    std::vector<int> cols;
    s = d_cols.value().download(cols);
    if (!s.ok()) return s;
    return cols;
}

std::vector<TextLine> detectLines(const std::vector<int>& rowCounts, int width, int minHeight) {
    const int thr = std::max(2, width / 200);
    std::vector<TextLine> lines;
    int start = -1;
    const int h = static_cast<int>(rowCounts.size());
    for (int y = 0; y <= h; ++y) {
        const bool ink = (y < h) && rowCounts[y] >= thr;
        if (ink && start < 0) start = y;
        if (!ink && start >= 0) {
            if (y - start >= minHeight) lines.push_back(TextLine{start, y - 1, {}});
            start = -1;
        }
    }
    return lines;
}

Status segmentLine(const DeviceImage& bin, TextLine& line, cudaStream_t stream) {
    auto colsRes = colProfile(bin, line.y0, line.y1, stream);
    if (!colsRes.ok()) return colsRes.status();
    const auto& cols = colsRes.value();
    const int w = static_cast<int>(cols.size());
    const int lineHeight = line.y1 - line.y0 + 1;
    const int wordGap = std::max(2, lineHeight / 3);

    line.words.clear();
    // Ink runs separated by >= wordGap blank columns form words; >= 1 blank column
    // inside a word separates characters.
    int runStart = -1, lastInk = -1, wordStart = -1;
    auto flushWord = [&](int end) {
        if (wordStart < 0) return;
        WordBox word{wordStart, line.y0, end, line.y1, {}};
        int cs = -1;
        for (int x = wordStart; x <= end + 1; ++x) {
            const bool ink = (x <= end) && cols[x] > 0;
            if (ink && cs < 0) cs = x;
            if (!ink && cs >= 0) {
                word.chars.push_back(CharBox{cs, line.y0, x - 1, line.y1});
                cs = -1;
            }
        }
        line.words.push_back(std::move(word));
        wordStart = -1;
    };
    for (int x = 0; x <= w; ++x) {
        const bool ink = (x < w) && cols[x] > 0;
        if (ink) {
            if (wordStart < 0) {
                wordStart = x;
            } else if (lastInk >= 0 && x - lastInk - 1 >= wordGap) {
                flushWord(lastInk);
                wordStart = x;
            }
            lastInk = x;
            if (runStart < 0) runStart = x;
        } else {
            runStart = -1;
        }
        if (x == w && wordStart >= 0) flushWord(lastInk);
    }
    return Status::Good();
}

}  // namespace cudaocr
