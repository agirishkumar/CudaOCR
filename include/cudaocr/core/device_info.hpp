#pragma once

#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Versions are encoded major*10 + minor (sm_86 -> 86), as in cudaFuncAttributes.
struct KernelArchInfo {
    int deviceArch;     // compute capability of the current device
    int binaryVersion;  // SASS the library's kernels execute as
    int ptxVersion;     // PTX embedded alongside (0 if none)
};

// Probes a kernel compiled into the library. If the build lacks native SASS for
// the device, kernels only run via PTX JIT — which fails outright when the
// toolkit is newer than the driver (product.md §4.3 #1).
Result<KernelArchInfo> queryKernelArch();

}  // namespace cudaocr
