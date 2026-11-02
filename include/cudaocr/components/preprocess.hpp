#pragma once

// Built-in ImageStage adapters over the custom preprocessing kernels
// (stages/*.cuh). Each is created from Params, which are validated up front so
// configuration errors surface before any image is processed.

#include <memory>

#include "cudaocr/components/image_stage.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "grayscale": BT.601 RGB -> gray; 1-channel input is copied through.
class GrayscaleStage final : public ImageStage {
  public:
    static Result<std::unique_ptr<GrayscaleStage>> create(const Params& p);
    std::string_view name() const override { return "grayscale"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;
};

// "equalize": global histogram equalization.
class EqualizeStage final : public ImageStage {
  public:
    static Result<std::unique_ptr<EqualizeStage>> create(const Params& p);
    std::string_view name() const override { return "equalize"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;
};

// "median": k = 3 (default) | 5. To disable denoising, omit the stage.
class MedianStage final : public ImageStage {
  public:
    static Result<std::unique_ptr<MedianStage>> create(const Params& p);
    std::string_view name() const override { return "median"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;

  private:
    explicit MedianStage(int k) : k_(k) {}
    int k_;
};

// "gaussian": ksize odd >= 3 (default 5); sigma >= 0, 0 = derived from ksize (default).
class GaussianStage final : public ImageStage {
  public:
    static Result<std::unique_ptr<GaussianStage>> create(const Params& p);
    std::string_view name() const override { return "gaussian"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;

  private:
    GaussianStage(int ksize, double sigma) : ksize_(ksize), sigma_(sigma) {}
    int ksize_;
    double sigma_;
};

// "polarity": flips light-on-dark pages to dark-on-light; sets page->inverted.
class PolarityStage final : public ImageStage {
  public:
    static Result<std::unique_ptr<PolarityStage>> create(const Params& p);
    std::string_view name() const override { return "polarity"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;
};

// "adaptive": local-mean threshold; blockSize odd >= 3 (default 31), C (default 10).
class AdaptiveBinarizer final : public ImageStage {
  public:
    static Result<std::unique_ptr<AdaptiveBinarizer>> create(const Params& p);
    std::string_view name() const override { return "adaptive"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;

  private:
    AdaptiveBinarizer(int blockSize, int C) : blockSize_(blockSize), C_(C) {}
    int blockSize_, C_;
};

// "otsu": global Otsu threshold; recorded as page->metrics["otsu.threshold"].
class OtsuBinarizer final : public ImageStage {
  public:
    static Result<std::unique_ptr<OtsuBinarizer>> create(const Params& p);
    std::string_view name() const override { return "otsu"; }
    Status run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) override;
    int lastThreshold() const { return lastThreshold_; }  // -1 before the first run

  private:
    int lastThreshold_ = -1;
};

}  // namespace cudaocr
