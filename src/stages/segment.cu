#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/segment.cuh"

#include <algorithm>
#include <limits>

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
    Status s = d_rows.value().zero(stream);
    if (!s.ok()) return s;
    const dim3 block(32, 8);
    const dim3 grid((bin.width() + block.x - 1) / block.x, (bin.height() + block.y - 1) / block.y);
    rowProfileKernel<<<grid, block, 0, stream>>>(bin.data(), bin.pitch(), bin.width(), bin.height(),
                                                 d_rows.value().data());
    CUDAOCR_CHECK_R(cudaGetLastError());
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    std::vector<int> rows;
    s = d_rows.value().download(rows, stream);
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
    Status s = d_cols.value().zero(stream);
    if (!s.ok()) return s;
    const int bandH = y1 - y0 + 1;
    const dim3 block(32, 8);
    const dim3 grid((bin.width() + block.x - 1) / block.x, (bandH + block.y - 1) / block.y);
    colProfileKernel<<<grid, block, 0, stream>>>(bin.data(), bin.pitch(), bin.width(), y0, y1,
                                                 d_cols.value().data());
    CUDAOCR_CHECK_R(cudaGetLastError());
    CUDAOCR_CHECK_R(cudaStreamSynchronize(stream));
    std::vector<int> cols;
    s = d_cols.value().download(cols, stream);
    if (!s.ok()) return s;
    return cols;
}

std::vector<TextLine> detectLines(const std::vector<int>& rowCounts, int width, int minHeight) {
    const int thr = std::max(2, width / 200);
    std::vector<TextLine> runs;
    int start = -1;
    const int h = static_cast<int>(rowCounts.size());
    for (int y = 0; y <= h; ++y) {
        const bool ink = (y < h) && rowCounts[y] >= thr;
        if (ink && start < 0) start = y;
        if (!ink && start >= 0) {
            runs.push_back(TextLine{start, y - 1, {}});
            start = -1;
        }
    }
    auto height = [](const TextLine& l) { return l.y1 - l.y0 + 1; };

    // Typical line height: median of the runs that would count as lines.
    std::vector<int> heights;
    for (const auto& r : runs)
        if (height(r) >= minHeight) heights.push_back(height(r));
    if (heights.empty()) return {};
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    const int typical = heights[heights.size() / 2];

    // Slivers (< 1/3 of the typical height) within 1/3 line of a neighbour are
    // descender bottoms or i/j dots split off by the ink threshold (measured:
    // 5-6 px against 35-45 px lines, 5 px gap): merge them into the nearer
    // neighbour (ties: the line above, the descender case). 1/3, not 1/2, so a
    // genuine short line next to noise blobs is not swallowed.
    constexpr int kFar = std::numeric_limits<int>::max();
    std::vector<TextLine> lines;
    for (size_t i = 0; i < runs.size(); ++i) {
        const TextLine& r = runs[i];
        if (3 * height(r) < typical) {
            const int gapPrev = lines.empty() ? kFar : r.y0 - lines.back().y1 - 1;
            const int gapNext = i + 1 < runs.size() ? runs[i + 1].y0 - r.y1 - 1 : kFar;
            if (3 * std::min(gapPrev, gapNext) <= typical) {
                if (gapPrev <= gapNext)
                    lines.back().y1 = r.y1;
                else
                    runs[i + 1].y0 = r.y0;
                continue;
            }
        }
        lines.push_back(r);
    }
    std::erase_if(lines, [&](const TextLine& l) { return height(l) < minHeight; });
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
