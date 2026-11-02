#include "cudaocr/components/preprocess.hpp"

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/enhance.cuh"
#include "cudaocr/stages/filters.cuh"
#include "cudaocr/stages/grayscale.cuh"

namespace cudaocr {
namespace {

Status invalid(std::string_view component, const std::string& what) {
    return Status::Error(ErrorCode::InvalidArgument, std::string(component) + ": " + what);
}

// Parses an int parameter into `out`; returns the parse error otherwise.
Status readInt(const Params& p, const char* key, int fallback, int& out) {
    auto v = p.get<int>(key, fallback);
    if (!v.ok()) return v.status();
    out = v.value();
    return Status::Good();
}

// Allocates `out` with in's shape and given channel count.
Status prepareOut(const DeviceImage& in, DeviceImage& out, int channels) {
    return out.ensure(in.width(), in.height(), channels);
}

}  // namespace

// --- grayscale ---
Result<std::unique_ptr<GrayscaleStage>> GrayscaleStage::create(const Params& p) {
    Status s = p.requireKnown({}, "grayscale");
    if (!s.ok()) return s;
    return std::make_unique<GrayscaleStage>();
}

Status GrayscaleStage::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    if (in.channels() == 1) return in.copyTo(out, ctx.stream);
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return grayscale(in, out, ctx.stream);
}

// --- equalize ---
Result<std::unique_ptr<EqualizeStage>> EqualizeStage::create(const Params& p) {
    Status s = p.requireKnown({}, "equalize");
    if (!s.ok()) return s;
    return std::make_unique<EqualizeStage>();
}

Status EqualizeStage::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return histogramEqualize(in, out, ctx.stream);
}

// --- median ---
Result<std::unique_ptr<MedianStage>> MedianStage::create(const Params& p) {
    Status s = p.requireKnown({"k"}, "median");
    int k = 3;
    if (s.ok()) s = readInt(p, "k", 3, k);
    if (!s.ok()) return s;
    if (k != 3 && k != 5) return invalid("median", "k must be 3 or 5, got " + std::to_string(k));
    return std::unique_ptr<MedianStage>(new MedianStage(k));
}

Status MedianStage::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return medianFilter(in, out, k_, ctx.stream);
}

// --- gaussian ---
Result<std::unique_ptr<GaussianStage>> GaussianStage::create(const Params& p) {
    Status s = p.requireKnown({"ksize", "sigma"}, "gaussian");
    int ksize = 5;
    if (s.ok()) s = readInt(p, "ksize", 5, ksize);
    if (!s.ok()) return s;
    auto sigma = p.get<double>("sigma", 0.0);
    if (!sigma.ok()) return sigma.status();
    if (ksize < 3 || ksize % 2 == 0)
        return invalid("gaussian", "ksize must be odd and >= 3, got " + std::to_string(ksize));
    if (sigma.value() < 0) return invalid("gaussian", "sigma must be >= 0");
    return std::unique_ptr<GaussianStage>(new GaussianStage(ksize, sigma.value()));
}

Status GaussianStage::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return gaussianBlur(in, out, ksize_, sigma_, ctx.stream);
}

// --- polarity ---
Result<std::unique_ptr<PolarityStage>> PolarityStage::create(const Params& p) {
    Status s = p.requireKnown({}, "polarity");
    if (!s.ok()) return s;
    return std::make_unique<PolarityStage>();
}

Status PolarityStage::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    auto inv = isInverted(in, ctx.stream);
    if (!inv.ok()) return inv.status();
    if (ctx.page) ctx.page->inverted = inv.value();
    if (!inv.value()) return in.copyTo(out, ctx.stream);
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return invert(in, out, ctx.stream);
}

// --- adaptive ---
Result<std::unique_ptr<AdaptiveBinarizer>> AdaptiveBinarizer::create(const Params& p) {
    Status s = p.requireKnown({"blockSize", "C"}, "adaptive");
    // C = 40 (was 10): see architecture.md decision #25 for the sweep.
    int blockSize = 31, C = 40;
    if (s.ok()) s = readInt(p, "blockSize", 31, blockSize);
    if (s.ok()) s = readInt(p, "C", 40, C);
    if (!s.ok()) return s;
    if (blockSize < 3 || blockSize % 2 == 0)
        return invalid("adaptive",
                       "blockSize must be odd and >= 3, got " + std::to_string(blockSize));
    return std::unique_ptr<AdaptiveBinarizer>(new AdaptiveBinarizer(blockSize, C));
}

Status AdaptiveBinarizer::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return adaptiveThreshold(in, out, blockSize_, C_, ctx.stream);
}

// --- otsu ---
Result<std::unique_ptr<OtsuBinarizer>> OtsuBinarizer::create(const Params& p) {
    Status s = p.requireKnown({}, "otsu");
    if (!s.ok()) return s;
    return std::make_unique<OtsuBinarizer>();
}

Status OtsuBinarizer::run(const DeviceImage& in, DeviceImage& out, StageContext& ctx) {
    auto t = otsuThreshold(in, ctx.stream);
    if (!t.ok()) return t.status();
    lastThreshold_ = t.value();
    if (ctx.page) ctx.page->metrics["otsu.threshold"] = t.value();
    Status s = prepareOut(in, out, 1);
    if (!s.ok()) return s;
    return applyThreshold(in, out, t.value(), ctx.stream);
}

}  // namespace cudaocr
