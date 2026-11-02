#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_info.hpp"

namespace cudaocr {
namespace {
__global__ void archProbeKernel() {}
}  // namespace

Result<KernelArchInfo> queryKernelArch() {
    int dev = 0;
    CUDAOCR_CHECK_R(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDAOCR_CHECK_R(cudaGetDeviceProperties(&prop, dev));
    cudaFuncAttributes attr{};
    CUDAOCR_CHECK_R(cudaFuncGetAttributes(&attr, archProbeKernel));
    return KernelArchInfo{prop.major * 10 + prop.minor, attr.binaryVersion, attr.ptxVersion};
}

}  // namespace cudaocr
