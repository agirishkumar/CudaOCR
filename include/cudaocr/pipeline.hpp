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

// Flow: input -> toGray -> preprocess -> G (the gray page the recognizer reads)
//       G -> binarizePrep -> binarizer -> deskew -> detector
// binarizePrep stages feed only the binarizer and never touch G. Measured in
// M3.11: moving the median there (recognizer reads unfiltered gray) made
// sherlock.png 0.89% -> 2.98% CER, so by default the median stays shared.
// Defaults: median 3x3 + polarity normalization (shared), no binarizePrep,
// adaptive threshold (31, 40; decision #25), Hough deskew (±15°, 0.25° step),
// projection-profile detection (min line height 3). Recognition is opt-in
// (`recognizer`, e.g. {"onnx", ...}): without it lines carry geometry only.
struct PipelineConfig {
    ComponentSpec toGray{"grayscale", {}};  // runs only for 3-channel input
    std::vector<ComponentSpec> preprocess{{"median", Params().set("k", 3)}, {"polarity", {}}};
    std::vector<ComponentSpec> binarizePrep;
    ComponentSpec binarizer{"adaptive", {}};
    std::optional<ComponentSpec> deskewer = ComponentSpec{"hough", {}};  // nullopt = skip
    ComponentSpec detector{"projection", {}};
    std::optional<ComponentSpec> recognizer;  // nullopt = detection only
    ComponentSpec decoder{"ctc-greedy", {}};  // used only with a recognizer
};

// Intermediate images for --dump-stages / debugging.
struct StageImages {
    HostImage gray;          // G: after preprocess (not binarizePrep), polarity-normalized
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
