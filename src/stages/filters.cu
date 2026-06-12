#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/stages/filters.cuh"

namespace cudaocr {
namespace {

constexpr int kMaxGaussTaps = 64;
__constant__ int c_gaussWeights[kMaxGaussTaps];

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void rowSumKernel(const std::uint8_t* __restrict__ in, size_t pitch,
                             std::uint32_t* __restrict__ tmp, int w, int h, int r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const std::uint8_t* row = in + static_cast<size_t>(y) * pitch;
    std::uint32_t s = 0;
    for (int dx = -r; dx <= r; ++dx) s += row[clampi(x + dx, 0, w - 1)];
    tmp[static_cast<size_t>(y) * w + x] = s;
}

__global__ void colMeanKernel(const std::uint32_t* __restrict__ tmp,
                              std::uint8_t* __restrict__ out, size_t outPitch, int w, int h, int r,
                              std::uint32_t n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    std::uint32_t s = 0;
    for (int dy = -r; dy <= r; ++dy) s += tmp[static_cast<size_t>(clampi(y + dy, 0, h - 1)) * w + x];
    out[static_cast<size_t>(y) * outPitch + x] = static_cast<std::uint8_t>((s + n / 2) / n);
}

template <int K>
__global__ void medianKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                             std::uint8_t* __restrict__ out, size_t outPitch, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    constexpr int N = K * K;
    constexpr int R = K / 2;
    std::uint8_t v[N];
    int n = 0;
    for (int dy = -R; dy <= R; ++dy) {
        const std::uint8_t* row = in + static_cast<size_t>(clampi(y + dy, 0, h - 1)) * inPitch;
        for (int dx = -R; dx <= R; ++dx) v[n++] = row[clampi(x + dx, 0, w - 1)];
    }
    // insertion sort; N <= 25
    for (int i = 1; i < N; ++i) {
        const std::uint8_t key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }
    out[static_cast<size_t>(y) * outPitch + x] = v[N / 2];
}

// One separable Gaussian pass; HORIZONTAL selects the axis. Accumulates
// weight*pixel in u32 (max 65536*255 < 2^32) and rounds back to 8u.
template <bool HORIZONTAL>
__global__ void gaussPassKernel(const std::uint8_t* __restrict__ in, size_t inPitch,
                                std::uint8_t* __restrict__ out, size_t outPitch, int w, int h,
                                int r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    std::uint32_t acc = 0;
    for (int d = -r; d <= r; ++d) {
        const int xx = HORIZONTAL ? clampi(x + d, 0, w - 1) : x;
        const int yy = HORIZONTAL ? y : clampi(y + d, 0, h - 1);
        acc += static_cast<std::uint32_t>(c_gaussWeights[d + r]) *
               in[static_cast<size_t>(yy) * inPitch + xx];
    }
    out[static_cast<size_t>(y) * outPitch + x] = static_cast<std::uint8_t>((acc + 32768u) >> 16);
}

Status validate1ch(const DeviceImage& in, const DeviceImage& out, const char* who) {
    if (in.channels() != 1 || out.channels() != 1)
        return Status::Error(ErrorCode::InvalidArgument, std::string(who) + ": 1-channel only");
    if (in.width() != out.width() || in.height() != out.height())
        return Status::Error(ErrorCode::InvalidArgument, std::string(who) + ": shape mismatch");
    return Status::Good();
}

dim3 gridFor(const DeviceImage& img, dim3 block) {
    return dim3((img.width() + block.x - 1) / block.x, (img.height() + block.y - 1) / block.y);
}

}  // namespace

Status boxMean(const DeviceImage& in, DeviceImage& outMean, int blockSize, cudaStream_t stream) {
    Status s = validate1ch(in, outMean, "boxMean");
    if (!s.ok()) return s;
    if (blockSize < 1 || blockSize % 2 == 0)
        return Status::Error(ErrorCode::InvalidArgument, "boxMean: blockSize must be odd >= 1");
    const int r = blockSize / 2;
    auto tmp = DeviceBuffer<std::uint32_t>::create(static_cast<size_t>(in.width()) * in.height());
    if (!tmp.ok()) return tmp.status();

    const dim3 block(32, 8);
    const dim3 grid = gridFor(in, block);
    rowSumKernel<<<grid, block, 0, stream>>>(in.data(), in.pitch(), tmp.value().data(), in.width(),
                                             in.height(), r);
    CUDAOCR_CHECK_LAUNCH("rowSumKernel");
    const auto n = static_cast<std::uint32_t>(blockSize) * blockSize;
    colMeanKernel<<<grid, block, 0, stream>>>(tmp.value().data(), outMean.data(), outMean.pitch(),
                                              in.width(), in.height(), r, n);
    CUDAOCR_CHECK_LAUNCH("colMeanKernel");
    // tmp is freed on return; the kernels using it run on `stream`, so block until done.
    CUDAOCR_CHECK(cudaStreamSynchronize(stream));
    return Status::Good();
}

Status medianFilter(const DeviceImage& in, DeviceImage& out, int k, cudaStream_t stream) {
    Status s = validate1ch(in, out, "medianFilter");
    if (!s.ok()) return s;
    const dim3 block(32, 8);
    const dim3 grid = gridFor(in, block);
    if (k == 3) {
        medianKernel<3><<<grid, block, 0, stream>>>(in.data(), in.pitch(), out.data(), out.pitch(),
                                                    in.width(), in.height());
    } else if (k == 5) {
        medianKernel<5><<<grid, block, 0, stream>>>(in.data(), in.pitch(), out.data(), out.pitch(),
                                                    in.width(), in.height());
    } else {
        return Status::Error(ErrorCode::InvalidArgument, "medianFilter: k must be 3 or 5");
    }
    CUDAOCR_CHECK_LAUNCH("medianKernel");
    return Status::Good();
}

Status gaussianBlur(const DeviceImage& in, DeviceImage& out, int ksize, double sigma,
                    cudaStream_t stream) {
    Status s = validate1ch(in, out, "gaussianBlur");
    if (!s.ok()) return s;
    if (ksize < 3 || ksize % 2 == 0 || ksize > kMaxGaussTaps)
        return Status::Error(ErrorCode::InvalidArgument, "gaussianBlur: bad ksize");
    const std::vector<int> w = gaussianFixedWeights(ksize, sigma);
    CUDAOCR_CHECK(cudaMemcpyToSymbolAsync(c_gaussWeights, w.data(), w.size() * sizeof(int), 0,
                                          cudaMemcpyHostToDevice, stream));
    auto tmp = DeviceImage::create(in.width(), in.height(), 1);
    if (!tmp.ok()) return tmp.status();

    const dim3 block(32, 8);
    const dim3 grid = gridFor(in, block);
    const int r = ksize / 2;
    gaussPassKernel<true><<<grid, block, 0, stream>>>(in.data(), in.pitch(), tmp.value().data(),
                                                      tmp.value().pitch(), in.width(), in.height(),
                                                      r);
    CUDAOCR_CHECK_LAUNCH("gaussPassKernel<horizontal>");
    gaussPassKernel<false><<<grid, block, 0, stream>>>(tmp.value().data(), tmp.value().pitch(),
                                                       out.data(), out.pitch(), in.width(),
                                                       in.height(), r);
    CUDAOCR_CHECK_LAUNCH("gaussPassKernel<vertical>");
    CUDAOCR_CHECK(cudaStreamSynchronize(stream));  // tmp freed on return
    return Status::Good();
}

}  // namespace cudaocr
