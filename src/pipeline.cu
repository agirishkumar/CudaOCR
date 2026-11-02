#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/logger.hpp"
#include "cudaocr/pipeline.hpp"
#include "cudaocr/stages/binarize.cuh"
#include "cudaocr/stages/deskew.cuh"
#include "cudaocr/stages/enhance.cuh"
#include "cudaocr/stages/filters.cuh"
#include "cudaocr/stages/grayscale.cuh"
#include "cudaocr/stages/segment.cuh"

#include <utility>

namespace cudaocr {
namespace {

// Times a stage with CUDA events (includes the stream sync the stage may do).
class StageTimer {
  public:
    StageTimer(std::vector<StageTime>& sink, cudaStream_t stream) : sink_(sink), stream_(stream) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~StageTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    template <typename F>
    Status timed(const std::string& name, F&& fn) {
        cudaEventRecord(start_, stream_);
        Status s = fn();
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        sink_.push_back({name, ms});
        if (!s.ok()) Logger::log(LogLevel::Error, "stage '" + name + "': " + s.message);
        return s;
    }

  private:
    std::vector<StageTime>& sink_;
    cudaStream_t stream_;
    cudaEvent_t start_{}, stop_{};
};

}  // namespace

Result<PipelineResult> runPipeline(const HostImage& input, const PipelineConfig& cfg,
                                   bool keepStages) {
    if (input.empty()) return Status::Error(ErrorCode::InvalidArgument, "runPipeline: empty input");

    PipelineResult result;
    auto streamRes = Stream::create();
    if (!streamRes.ok()) return streamRes.status();
    Stream& stream = streamRes.value();
    StageTimer timer(result.timings, stream.get());
    Status s = Status::Good();

    // --- upload + grayscale ---
    DeviceImage gray;
    if (input.channels == 3) {
        auto rgb = DeviceImage::fromHost(input);
        if (!rgb.ok()) return rgb.status();
        auto g = DeviceImage::create(input.width, input.height, 1);
        if (!g.ok()) return g.status();
        gray = g.take();
        s = timer.timed("grayscale", [&] { return grayscale(rgb.value(), gray, stream.get()); });
        if (!s.ok()) return s;
    } else {
        auto g = DeviceImage::fromHost(input);
        if (!g.ok()) return g.status();
        gray = g.take();
    }

    // --- optional equalize ---
    if (cfg.equalize) {
        auto eq = DeviceImage::create(gray.width(), gray.height(), 1);
        if (!eq.ok()) return eq.status();
        s = timer.timed("equalize",
                        [&] { return histogramEqualize(gray, eq.value(), stream.get()); });
        if (!s.ok()) return s;
        gray = eq.take();
    }

    // --- denoise ---
    if (cfg.medianK > 0) {
        auto den = DeviceImage::create(gray.width(), gray.height(), 1);
        if (!den.ok()) return den.status();
        s = timer.timed("median",
                        [&] { return medianFilter(gray, den.value(), cfg.medianK, stream.get()); });
        if (!s.ok()) return s;
        gray = den.take();
    }

    // --- normalize polarity BEFORE binarization: adaptive thresholding
    // light-on-dark text directly produces halo artifacts, so flip the gray
    // image when dark pixels dominate. ---
    s = timer.timed("polarity", [&]() -> Status {
        auto inv = isInverted(gray, stream.get());
        if (!inv.ok()) return inv.status();
        result.wasInverted = inv.value();
        if (inv.value()) {
            auto flipped = DeviceImage::create(gray.width(), gray.height(), 1);
            if (!flipped.ok()) return flipped.status();
            Status st = invert(gray, flipped.value(), stream.get());
            if (!st.ok()) return st;
            gray = flipped.take();
        }
        return Status::Good();
    });
    if (!s.ok()) return s;
    if (keepStages) {
        s = gray.download(result.gray);
        if (!s.ok()) return s;
    }

    // --- binarize ---
    auto binRes = DeviceImage::create(gray.width(), gray.height(), 1);
    if (!binRes.ok()) return binRes.status();
    DeviceImage bin = binRes.take();
    if (cfg.useOtsu) {
        s = timer.timed("otsu", [&]() -> Status {
            auto t = otsuThreshold(gray, stream.get());
            if (!t.ok()) return t.status();
            result.otsuThreshold = t.value();
            return applyThreshold(gray, bin, t.value(), stream.get());
        });
    } else {
        s = timer.timed("adaptiveThreshold", [&] {
            return adaptiveThreshold(gray, bin, cfg.blockSize, cfg.C, stream.get());
        });
    }
    if (!s.ok()) return s;

    if (keepStages) {
        s = bin.download(result.binary);
        if (!s.ok()) return s;
    }

    // --- deskew ---
    if (cfg.deskew) {
        s = timer.timed("deskew", [&]() -> Status {
            auto angle = detectSkewAngle(bin, cfg.maxSkewDeg, cfg.skewStepDeg, stream.get());
            if (!angle.ok()) return angle.status();
            result.skewAngleDeg = angle.value();
            if (std::abs(angle.value()) >= cfg.skewStepDeg / 2) {
                auto straight = DeviceImage::create(bin.width(), bin.height(), 1);
                if (!straight.ok()) return straight.status();
                Status st = rotate(bin, straight.value(), -angle.value(), 255, stream.get());
                if (!st.ok()) return st;
                bin = straight.take();
            }
            return Status::Good();
        });
        if (!s.ok()) return s;
    }
    if (keepStages) {
        s = bin.download(result.deskewed);
        if (!s.ok()) return s;
    }

    // --- segmentation ---
    s = timer.timed("segment", [&]() -> Status {
        auto rows = rowProfile(bin, stream.get());
        if (!rows.ok()) return rows.status();
        result.lines = detectLines(rows.value(), bin.width(), cfg.minLineHeight);
        for (auto& line : result.lines) {
            Status st = segmentLine(bin, line, stream.get());
            if (!st.ok()) return st;
            result.wordCount += static_cast<int>(line.words.size());
            for (const auto& wbox : line.words)
                result.charCount += static_cast<int>(wbox.chars.size());
        }
        return Status::Good();
    });
    if (!s.ok()) return s;

    s = stream.synchronize();
    if (!s.ok()) return s;
    return result;
}

}  // namespace cudaocr
