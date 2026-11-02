#include "cudaocr/components/ctc_greedy_decoder.hpp"
#include "cudaocr/components/hocr_formatter.hpp"
#include "cudaocr/components/hough_deskewer.hpp"
#include "cudaocr/components/json_formatter.hpp"
#ifdef CUDAOCR_WITH_ORT
#include "cudaocr/components/onnx_line_recognizer.hpp"
#endif
#include "cudaocr/components/preprocess.hpp"
#include "cudaocr/components/projection_detector.hpp"
#include "cudaocr/components/txt_formatter.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

// Adapts T::create (returning unique_ptr<T>) to a Factory<I>.
template <typename T, typename I>
ComponentRegistry::Factory<I> factoryOf() {
    return [](const Params& p) -> Result<std::unique_ptr<I>> {
        auto made = T::create(p);
        if (!made.ok()) return made.status();
        return std::unique_ptr<I>(made.take());
    };
}

}  // namespace

ComponentRegistry ComponentRegistry::withBuiltins() {
    ComponentRegistry r;
    // Names are unique by construction, so add() cannot fail here.
    (void)r.add<ImageStage>("grayscale", factoryOf<GrayscaleStage, ImageStage>());
    (void)r.add<ImageStage>("equalize", factoryOf<EqualizeStage, ImageStage>());
    (void)r.add<ImageStage>("median", factoryOf<MedianStage, ImageStage>());
    (void)r.add<ImageStage>("gaussian", factoryOf<GaussianStage, ImageStage>());
    (void)r.add<ImageStage>("polarity", factoryOf<PolarityStage, ImageStage>());
    (void)r.add<ImageStage>("adaptive", factoryOf<AdaptiveBinarizer, ImageStage>());
    (void)r.add<ImageStage>("otsu", factoryOf<OtsuBinarizer, ImageStage>());
    (void)r.add<Deskewer>("hough", factoryOf<HoughDeskewer, Deskewer>());
    (void)r.add<TextDetector>("projection", factoryOf<ProjectionDetector, TextDetector>());
    (void)r.add<Decoder>("ctc-greedy", factoryOf<CtcGreedyDecoder, Decoder>());
    (void)r.add<Formatter>("txt", factoryOf<TxtFormatter, Formatter>());
    (void)r.add<Formatter>("json", factoryOf<JsonFormatter, Formatter>());
    (void)r.add<Formatter>("hocr", factoryOf<HocrFormatter, Formatter>());
#ifdef CUDAOCR_WITH_ORT
    (void)r.add<LineRecognizer>("onnx", factoryOf<OnnxLineRecognizer, LineRecognizer>());
#endif
    return r;
}

}  // namespace cudaocr
