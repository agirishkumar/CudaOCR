#pragma once

// "onnx": CTC line recognizer running an ONNX model (PP-OCRv5 English by default,
// models/README.md) through ONNX Runtime. ORT headers stay in the .cpp.
//
// Params (defaults):
//   model     path to rec.onnx   ($CUDAOCR_MODEL_DIR, the install's share/cudaocr/
//             models, or the source tree's models/ — see defaultModelDir())
//   dict      path to dict.txt   (same directory as the default model)
//   ep        cuda | cpu         (cuda)
//   maxBatch  lines per ORT run, >= 1 (16); lines are sorted by width first to
//             minimise padding, results come back in input order
//   padFrac   line-crop padding, fraction of box height (0.1, decision #23)
//   tf32      allow TF32 math on the CUDA EP (false): faster on Ampere+, but it
//             moved probabilities by up to 0.065 vs the CPU EP

#include <memory>
#include <string>

#include "cudaocr/components/line_recognizer.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

class OnnxLineRecognizer final : public LineRecognizer {
  public:
    // Loads the session and charset once; checks the charset size against the
    // model's class count and the input height against the model.
    static Result<std::unique_ptr<OnnxLineRecognizer>> create(const Params& p);
    ~OnnxLineRecognizer() override;

    std::string_view name() const override { return "onnx"; }
    const Charset& charset() const override;
    // Output i has floor(crop_i.outW / 8) timesteps: only line i's own columns,
    // never the batch's white padding.
    Result<std::vector<CtcProbs>> recognize(const DeviceImage& gray,
                                            std::span<const LineResult> lines,
                                            StageContext& ctx) override;

    // $CUDAOCR_MODEL_DIR if set; else <prefix>/share/cudaocr/models next to an
    // installed binary when it holds rec.onnx; else the source tree's models/.
    static std::string defaultModelDir();

  private:
    struct Impl;
    explicit OnnxLineRecognizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudaocr
