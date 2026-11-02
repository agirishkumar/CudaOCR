#include "cudaocr/core/device_buffer.cuh"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/core/logger.hpp"
#include "cudaocr/pipeline.hpp"

#include <utility>

namespace cudaocr {
namespace {

// Times a call with CUDA events on the pipeline stream (includes any sync the
// component does) and appends it to the page's timings.
class StageTimer {
  public:
    explicit StageTimer(cudaStream_t stream) : stream_(stream) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~StageTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

    template <typename F>
    Status timed(PageResult& page, std::string_view name, F&& fn) {
        cudaEventRecord(start_, stream_);
        Status s = fn();
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        page.timings.push_back({std::string(name), ms});
        if (!s.ok()) {
            s.message = std::string(name) + ": " + s.message;
            Logger::log(LogLevel::Error, "stage " + s.message);
        }
        return s;
    }

  private:
    cudaStream_t stream_;
    cudaEvent_t start_{}, stop_{};
};

template <typename I>
Status resolve(const ComponentRegistry& reg, const ComponentSpec& spec, const std::string& slot,
               std::unique_ptr<I>& out) {
    auto made = reg.make<I>(spec);
    if (!made.ok()) return Status::Error(made.status().code, slot + " " + made.status().message);
    out = made.take();
    return Status::Good();
}

}  // namespace

struct Pipeline::Impl {
    Stream stream;
    std::unique_ptr<ImageStage> toGray;
    std::vector<std::unique_ptr<ImageStage>> preprocess;
    std::unique_ptr<ImageStage> binarizer;
    std::unique_ptr<Deskewer> deskewer;  // null = skip
    std::unique_ptr<TextDetector> detector;
    // Device buffers reused across runs (DeviceImage::ensure).
    DeviceImage input, a, b, bin, straight;
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;
Pipeline::~Pipeline() = default;

Result<Pipeline> Pipeline::create(const PipelineConfig& cfg, const ComponentRegistry& registry) {
    auto impl = std::make_unique<Impl>();
    auto stream = Stream::create();
    if (!stream.ok()) return stream.status();
    impl->stream = stream.take();

    Status s = resolve(registry, cfg.toGray, "toGray", impl->toGray);
    for (size_t i = 0; s.ok() && i < cfg.preprocess.size(); ++i) {
        std::unique_ptr<ImageStage> st;
        s = resolve(registry, cfg.preprocess[i], "preprocess[" + std::to_string(i) + "]", st);
        if (s.ok()) impl->preprocess.push_back(std::move(st));
    }
    if (s.ok()) s = resolve(registry, cfg.binarizer, "binarizer", impl->binarizer);
    if (s.ok() && cfg.deskewer) s = resolve(registry, *cfg.deskewer, "deskewer", impl->deskewer);
    if (s.ok()) s = resolve(registry, cfg.detector, "detector", impl->detector);
    if (!s.ok()) return s;
    return Pipeline(std::move(impl));
}

Result<PageResult> Pipeline::run(const HostImage& input, StageImages* keep) {
    if (input.empty())
        return Status::Error(ErrorCode::InvalidArgument, "Pipeline::run: empty input");
    Impl& m = *impl_;
    PageResult page;
    page.width = input.width;
    page.height = input.height;
    StageContext ctx{m.stream.get(), &page};
    StageTimer timer(m.stream.get());

    Status s = m.input.ensure(input.width, input.height, input.channels);
    if (s.ok()) s = m.input.upload(input, m.stream.get());
    if (!s.ok()) return s;

    // Ping-pong between a and b; `cur` always points at the latest gray image.
    const DeviceImage* cur = &m.input;
    DeviceImage* spare = &m.a;
    auto step = [&](ImageStage& stage) -> Status {
        Status st = timer.timed(page, stage.name(), [&] { return stage.run(*cur, *spare, ctx); });
        if (!st.ok()) return st;
        cur = spare;
        spare = (spare == &m.a) ? &m.b : &m.a;
        return Status::Good();
    };

    if (input.channels != 1) s = step(*m.toGray);
    for (size_t i = 0; s.ok() && i < m.preprocess.size(); ++i) s = step(*m.preprocess[i]);
    if (!s.ok()) return s;
    if (keep) {
        s = cur->download(keep->gray, m.stream.get());
        if (!s.ok()) return s;
    }

    s = timer.timed(page, m.binarizer->name(), [&] { return m.binarizer->run(*cur, m.bin, ctx); });
    if (!s.ok()) return s;
    if (keep) {
        s = m.bin.download(keep->binary, m.stream.get());
        if (!s.ok()) return s;
    }

    const DeviceImage* binary = &m.bin;
    if (m.deskewer) {
        s = timer.timed(page, m.deskewer->name(), [&]() -> Status {
            auto angle = m.deskewer->estimate(m.bin, ctx);
            if (!angle.ok()) return angle.status();
            page.skewDeg = angle.value();
            return m.deskewer->apply(-angle.value(), m.bin, m.straight, 255, ctx);
        });
        if (!s.ok()) return s;
        binary = &m.straight;
    }
    if (keep) {
        s = binary->download(keep->deskewed, m.stream.get());
        if (!s.ok()) return s;
    }

    s = timer.timed(page, m.detector->name(), [&]() -> Status {
        auto lines = m.detector->detect(*binary, ctx);
        if (!lines.ok()) return lines.status();
        page.lines = lines.take();
        return Status::Good();
    });
    if (!s.ok()) return s;

    s = m.stream.synchronize();
    if (!s.ok()) return s;
    return page;
}

}  // namespace cudaocr
