// Custom kernels vs NPP (_Ctx) on a synthetic 2500x3500 page. Prints a table.
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>

#include <cstdio>
#include <functional>

#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/filters.cuh"
#include "cudaocr/stages/grayscale.cuh"

using namespace cudaocr;

namespace {

float timeMs(cudaStream_t stream, const std::function<void()>& fn, int iters = 20) {
    fn();  // warm-up
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

HostImage randomHost(int w, int h, int ch) {
    HostImage img = HostImage::create(w, h, ch);
    std::uint32_t state = 12345;
    for (auto& p : img.data) {
        state = state * 1664525u + 1013904223u;
        p = static_cast<std::uint8_t>(state >> 24);
    }
    return img;
}

}  // namespace

int main() {
    const int W = 2500, H = 3500;
    auto rgbRes = DeviceImage::fromHost(randomHost(W, H, 3));
    auto grayRes = DeviceImage::fromHost(randomHost(W, H, 1));
    auto out1 = DeviceImage::create(W, H, 1);
    auto out2 = DeviceImage::create(W, H, 1);
    if (!rgbRes.ok() || !grayRes.ok() || !out1.ok() || !out2.ok()) {
        std::fprintf(stderr, "alloc failed\n");
        return 1;
    }
    DeviceImage& rgb = rgbRes.value();
    DeviceImage& gray = grayRes.value();

    // CUDA 13 has no nppGetStreamContext; the application fills the struct.
    NppStreamContext ctx{};
    ctx.hStream = nullptr;  // same (default) stream as the custom kernels
    cudaGetDevice(&ctx.nCudaDeviceId);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId);
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                           cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                           cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId);
    const NppiSize roi{W, H};

    std::printf("%-28s %10s %10s %8s\n", "kernel (2500x3500)", "custom ms", "npp ms", "ratio");
    auto report = [](const char* name, float custom, float npp) {
        std::printf("%-28s %10.3f %10.3f %7.2fx\n", name, custom, npp, custom / npp);
    };

    {
        const Npp32f coeffs[3] = {77.0f / 256, 150.0f / 256, 29.0f / 256};
        const float c = timeMs(nullptr, [&] { (void)grayscale(rgb, out1.value()); });
        const float n = timeMs(nullptr, [&] {
            nppiColorToGray_8u_C3C1R_Ctx(rgb.data(), static_cast<int>(rgb.pitch()),
                                         out2.value().data(),
                                         static_cast<int>(out2.value().pitch()), roi, coeffs, ctx);
        });
        report("grayscale", c, n);
    }
    {
        const float c = timeMs(nullptr, [&] { (void)gaussianBlur(gray, out1.value(), 5, 1.1); });
        const float n = timeMs(nullptr, [&] {
            nppiFilterGaussBorder_8u_C1R_Ctx(gray.data(), static_cast<int>(gray.pitch()), roi,
                                             {0, 0}, out2.value().data(),
                                             static_cast<int>(out2.value().pitch()), roi,
                                             NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, ctx);
        });
        report("gaussian 5x5", c, n);
    }
    {
        // NPP median needs a scratch buffer.
        Npp32u bufSize = 0;
        nppiFilterMedianGetBufferSize_8u_C1R_Ctx(roi, {5, 5}, &bufSize, ctx);
        auto scratch = DeviceBuffer<std::uint8_t>::create(bufSize ? bufSize : 1);
        if (!scratch.ok()) return 1;
        const NppiSize inner{W - 8, H - 8};
        const float c = timeMs(nullptr, [&] { (void)medianFilter(gray, out1.value(), 5); });
        const float n = timeMs(nullptr, [&] {
            nppiFilterMedian_8u_C1R_Ctx(gray.data() + 4 * gray.pitch() + 4,
                                        static_cast<int>(gray.pitch()),
                                        out2.value().data() + 4 * out2.value().pitch() + 4,
                                        static_cast<int>(out2.value().pitch()), inner, {5, 5},
                                        {2, 2}, scratch.value().data(), ctx);
        });
        report("median 5x5", c, n);
    }
    {
        const float c = timeMs(nullptr, [&] { (void)boxMean(gray, out1.value(), 31); });
        const NppiSize inner{W - 32, H - 32};
        const float n = timeMs(nullptr, [&] {
            nppiFilterBox_8u_C1R_Ctx(
                gray.data() + 16 * gray.pitch() + 16, static_cast<int>(gray.pitch()),
                out2.value().data() + 16 * out2.value().pitch() + 16,
                static_cast<int>(out2.value().pitch()), inner, {31, 31}, {15, 15}, ctx);
        });
        report("box mean 31x31", c, n);
    }
    cudaDeviceSynchronize();
    return 0;
}
