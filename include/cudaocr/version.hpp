#pragma once

#include <string>

namespace cudaocr {

// "0.1.0" (CMake project version).
const char* version();
// Multi-line build summary for --version: version, optional features
// (recognizer backend, PDF support) and the CUDA architectures compiled in.
std::string buildInfo(const char* program);

}  // namespace cudaocr
