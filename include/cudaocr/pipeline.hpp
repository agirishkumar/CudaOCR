#pragma once

// Component-based OCR pipeline. The config names each component; the registry
// turns names into instances once, at create(). Host-only header.

#include <memory>
#include <optional>
#include <vector>

#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/status.hpp"
#include "cudaocr/registry.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

// Defaults reproduce the original hard-wired pipeline: median 3x3, polarity
// normalization, adaptive threshold (31, 10), Hough deskew (±15°, 0.25° step),
// projection-profile detection (min line height 3).
struct PipelineConfig {
    ComponentSpec toGray{"grayscale", {}};  // runs only for 3-channel input
    std::vector<ComponentSpec> preprocess{{"median", Params().set("k", 3)}, {"polarity", {}}};
    ComponentSpec binarizer{"adaptive", {}};
    std::optional<ComponentSpec> deskewer = ComponentSpec{"hough", {}};  // nullopt = skip
    ComponentSpec detector{"projection", {}};
};

// Intermediate images for --dump-stages / debugging.
struct StageImages {
    HostImage gray;          // after preprocessing (polarity-normalized)
    HostImage binary;        // after binarization
    HostImage deskewed;      // binary after deskew (== binary when deskew is off)
    HostImage deskewedGray;  // gray after deskew, same angle; what the recognizer reads
};

class Pipeline {
  public:
    // Resolves and validates every component; errors name the config slot,
    // e.g. "preprocess[1] median: ...".
    static Result<Pipeline> create(
        const PipelineConfig& cfg,
        const ComponentRegistry& registry = ComponentRegistry::withBuiltins());

    // Reusable across pages; device buffers are kept and only reallocated when
    // the page size changes.
    Result<PageResult> run(const HostImage& input, StageImages* keep = nullptr);

    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    ~Pipeline();

  private:
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudaocr
