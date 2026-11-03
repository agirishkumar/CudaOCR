#include "cudaocr/components/onnx_line_recognizer.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <numeric>

#include "cudaocr/core/device_image.cuh"
#include "cudaocr/stages/line_prep.cuh"

namespace cudaocr {
namespace {

// One Env per process. It must exist before any provider-options call
// (CreateCUDAProviderOptions needs ORT's default logger; found in M3.4), and it
// is deliberately never destroyed: tearing an Env down during static
// destruction, after the CUDA EP, intermittently corrupted the heap at exit.
Ort::Env& ortEnv() {
    static Ort::Env& env =
        *new Ort::Env(ORT_LOGGING_LEVEL_ERROR,
                      "cudaocr");  // WARNING floods stderr with node-placement notes
    return env;
}

Status invalid(const std::string& msg) {
    return Status::Error(ErrorCode::InvalidArgument, "onnx: " + msg);
}

}  // namespace

struct OnnxLineRecognizer::Impl {
    std::unique_ptr<Ort::Session> session;
    Charset charset;
    bool cuda = true;
    int maxBatch = 16;
    LinePrepParams prep;
    std::string inputName, outputName;
};

OnnxLineRecognizer::OnnxLineRecognizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
OnnxLineRecognizer::~OnnxLineRecognizer() = default;
const Charset& OnnxLineRecognizer::charset() const {
    return impl_->charset;
}

std::string OnnxLineRecognizer::defaultModelDir() {
    if (const char* env = std::getenv("CUDAOCR_MODEL_DIR"); env && *env) return env;
    return CUDAOCR_DEFAULT_MODEL_DIR;
}

Result<std::unique_ptr<OnnxLineRecognizer>> OnnxLineRecognizer::create(const Params& p) {
    Status s =
        p.requireKnown({"model", "dict", "ep", "maxBatch", "padFrac", "tf32", "maxWidth"}, "onnx");
    if (!s.ok()) return s;
    const std::string dir = defaultModelDir();
    auto model = p.get<std::string>("model", dir + "/rec.onnx");
    auto ep = p.get<std::string>("ep", "cuda");
    auto maxBatch = p.get<int>("maxBatch", 16);
    auto padFrac = p.get<float>("padFrac", 0.1f);
    auto tf32 = p.get<bool>("tf32", false);
    auto maxWidth = p.get<int>("maxWidth", LinePrepParams{}.maxWidth);
    for (const Status* st : {&model.status(), &ep.status(), &maxBatch.status(), &padFrac.status(),
                             &tf32.status(), &maxWidth.status()})
        if (!st->ok()) return *st;
    const std::string dictDefault =
        (std::filesystem::path(model.value()).parent_path() / "dict.txt").string();
    auto dict = p.get<std::string>("dict", dictDefault);
    if (!dict.ok()) return dict.status();
    if (ep.value() != "cuda" && ep.value() != "cpu")
        return invalid("ep must be cuda or cpu, got '" + ep.value() + "'");
    if (maxBatch.value() < 1) return invalid("maxBatch must be >= 1");
    if (padFrac.value() < 0) return invalid("padFrac must be >= 0");
    if (maxWidth.value() < 0) return invalid("maxWidth must be >= 0 (0 = no cap)");
    if (!std::filesystem::exists(model.value()))
        return Status::Error(ErrorCode::IoError,
                             "onnx: model not found: " + model.value() +
                                 " (run scripts/fetch_models.sh or set CUDAOCR_MODEL_DIR)");

    auto impl = std::make_unique<Impl>();
    impl->cuda = ep.value() == "cuda";
    impl->maxBatch = maxBatch.value();
    impl->prep.padFrac = padFrac.value();
    impl->prep.maxWidth = maxWidth.value();
    auto cs = loadCtcCharset(dict.value());
    if (!cs.ok()) return cs.status();
    impl->charset = cs.take();

    try {
        Ort::Env& env = ortEnv();
        Ort::SessionOptions opts;
        if (impl->cuda) {
            OrtCUDAProviderOptionsV2* cuda = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
            // ORT enables TF32 by default on Ampere+; it moved probabilities by up
            // to 0.065 vs the CPU EP (M3.8), so it is opt-in.
            const char* keys[] = {"use_tf32"};
            const char* vals[] = {tf32.value() ? "1" : "0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda, keys, vals, 1));
            opts.AppendExecutionProvider_CUDA_V2(*cuda);
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
        }
        impl->session = std::make_unique<Ort::Session>(env, model.value().c_str(), opts);
        Ort::Session& sess = *impl->session;
        Ort::AllocatorWithDefaultOptions alloc;
        impl->inputName = sess.GetInputNameAllocated(0, alloc).get();
        impl->outputName = sess.GetOutputNameAllocated(0, alloc).get();
        const Ort::TypeInfo inInfo = sess.GetInputTypeInfo(0);
        const auto inShape = inInfo.GetTensorTypeAndShapeInfo().GetShape();
        const Ort::TypeInfo outInfo = sess.GetOutputTypeInfo(0);
        const auto outShape = outInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inShape.size() != 4 || outShape.size() != 3)
            return invalid("expected input [N,C,H,W] and output [N,T,classes]");
        if (inShape[1] > 0) impl->prep.channels = static_cast<int>(inShape[1]);
        if (inShape[2] > 0) impl->prep.height = static_cast<int>(inShape[2]);
        if (outShape[2] > 0 && outShape[2] != static_cast<int64_t>(impl->charset.symbols.size()))
            return invalid("model has " + std::to_string(outShape[2]) + " classes but " +
                           dict.value() + " gives " + std::to_string(impl->charset.symbols.size()) +
                           " (blank + entries + space)");
    } catch (const Ort::Exception& e) {
        return Status::Error(ErrorCode::InternalError,
                             std::string("onnx: cannot load ") + model.value() + ": " + e.what());
    }
    return std::unique_ptr<OnnxLineRecognizer>(new OnnxLineRecognizer(std::move(impl)));
}

Result<std::vector<CtcProbs>> OnnxLineRecognizer::recognize(const DeviceImage& gray,
                                                            std::span<const LineResult> lines,
                                                            StageContext& ctx) {
    Impl& m = *impl_;
    std::vector<CtcProbs> out(lines.size());
    if (lines.empty()) return out;

    std::vector<BBox> boxes;
    boxes.reserve(lines.size());
    for (const auto& l : lines) boxes.push_back(l.box);
    auto plan = planLineCrops(boxes, gray.width(), gray.height(), m.prep);
    if (!plan.ok()) return plan.status();
    // Narrow lines batched with narrow lines: less white padding to run through.
    std::vector<size_t> order(lines.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return plan.value()[a].outW < plan.value()[b].outW;
    });

    try {
        const Ort::MemoryInfo cpuMem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::MemoryInfo gpuMem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        const char* inName = m.inputName.c_str();
        const char* outName = m.outputName.c_str();
        for (size_t start = 0; start < order.size(); start += m.maxBatch) {
            const size_t n = std::min(order.size() - start, static_cast<size_t>(m.maxBatch));
            std::vector<BBox> chunk;
            for (size_t k = 0; k < n; ++k) chunk.push_back(boxes[order[start + k]]);
            // prepareLines synchronizes ctx.stream, so the batch is complete here.
            auto batch = prepareLines(gray, chunk, m.prep, ctx.stream);
            if (!batch.ok()) return batch.status();
            LineBatch& b = batch.value();
            const std::array<int64_t, 4> shape{b.count, b.channels, b.height, b.width};
            const size_t elems = static_cast<size_t>(b.count) * b.channels * b.height * b.width;

            std::vector<float> host;  // CPU EP input copy (empty on CUDA)
            Ort::Value input{nullptr};
            if (m.cuda) {
                input = Ort::Value::CreateTensor<float>(gpuMem, b.data.data(), elems, shape.data(),
                                                        shape.size());
            } else {
                Status st = b.data.download(host, ctx.stream);
                if (!st.ok()) return st;
                input = Ort::Value::CreateTensor<float>(cpuMem, host.data(), host.size(),
                                                        shape.data(), shape.size());
            }
            auto outputs =
                m.session->Run(Ort::RunOptions{nullptr}, &inName, &input, 1, &outName, 1);
            const auto os = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* probs = outputs[0].GetTensorData<float>();
            const int64_t T = os[1], C = os[2];
            for (size_t k = 0; k < n; ++k) {
                const int64_t own = std::min<int64_t>(T, b.crops[k].outW / 8);
                CtcProbs p(static_cast<int>(own), static_cast<int>(C));
                // One timestep = 8 resized columns = 8 * w / outW crop columns.
                const LineCrop& crop = b.crops[k];
                p.x0 = static_cast<float>(crop.x0);
                p.stepWidth = 8.0f * static_cast<float>(crop.w) / static_cast<float>(crop.outW);
                const float* src = probs + static_cast<int64_t>(k) * T * C;
                std::copy(src, src + own * C, p.data().begin());
                out[order[start + k]] = std::move(p);
            }
        }
    } catch (const Ort::Exception& e) {
        return Status::Error(ErrorCode::InternalError,
                             std::string("onnx: inference failed: ") + e.what());
    }
    return out;
}

}  // namespace cudaocr
