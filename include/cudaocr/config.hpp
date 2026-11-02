#pragma once

// PipelineConfig <-> TOML (docs/config.md). Host-only; toml++ stays in the .cpp.
//
//   preprocess = [ { name = "median", k = 3 }, { name = "polarity" } ]
//   binarizer  = { name = "adaptive", C = 40 }
//   deskewer   = false                       # disable an optional slot
//   recognizer = { name = "onnx", ep = "cuda" }
//
// Keys not present keep the base config's value. Errors are InvalidArgument with
// "<source>:<line>:<col>: ..." locations.

#include <string>
#include <string_view>

#include "cudaocr/core/status.hpp"
#include "cudaocr/pipeline.hpp"

namespace cudaocr {

Result<PipelineConfig> parsePipelineConfig(std::string_view toml,
                                           std::string_view sourceName = "<config>",
                                           const PipelineConfig& base = {});
// IoError if the file cannot be read.
Result<PipelineConfig> loadPipelineConfig(const std::string& path, const PipelineConfig& base = {});

// Canonical, complete TOML for `cfg`; parsePipelineConfig(toToml(c)) == c.
std::string toToml(const PipelineConfig& cfg);

}  // namespace cudaocr
