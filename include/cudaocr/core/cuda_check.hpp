#pragma once

#include <cuda_runtime_api.h>

#include <string>

#include "cudaocr/core/status.hpp"

namespace cudaocr::detail {
inline Status cudaErrorToStatus(cudaError_t err, const char* expr, const char* file, int line) {
    return Status::Error(ErrorCode::CudaError, std::string(expr) +
                                                   " failed: " + cudaGetErrorString(err) + " (" +
                                                   file + ":" + std::to_string(line) + ")");
}
}  // namespace cudaocr::detail

// Evaluates a CUDA runtime call; on failure returns a Status from the enclosing function.
#define CUDAOCR_CHECK(expr)                                                                       \
    do {                                                                                          \
        cudaError_t cudaocr_err_ = (expr);                                                        \
        if (cudaocr_err_ != cudaSuccess) {                                                        \
            return ::cudaocr::detail::cudaErrorToStatus(cudaocr_err_, #expr, __FILE__, __LINE__); \
        }                                                                                         \
    } while (0)

// Same, for functions returning Result<T>.
#define CUDAOCR_CHECK_R(expr) CUDAOCR_CHECK(expr)

// After a kernel launch: surface launch errors with the kernel's name attached.
#define CUDAOCR_CHECK_LAUNCH(kernelName)                                                    \
    do {                                                                                    \
        cudaError_t cudaocr_err_ = cudaGetLastError();                                      \
        if (cudaocr_err_ != cudaSuccess) {                                                  \
            return ::cudaocr::detail::cudaErrorToStatus(cudaocr_err_, kernelName, __FILE__, \
                                                        __LINE__);                          \
        }                                                                                   \
    } while (0)
