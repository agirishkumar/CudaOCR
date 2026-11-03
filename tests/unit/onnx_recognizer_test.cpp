// M3.8: OnnxLineRecognizer on detected lines of the synthetic fixtures, both EPs.
// Skipped when models/ is empty (scripts/fetch_models.sh).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "cudaocr/components/ctc_greedy_decoder.hpp"
#include "cudaocr/components/onnx_line_recognizer.hpp"
#include "cudaocr/core/device_image.cuh"
#include "cudaocr/pipeline.hpp"
#include "cudaocr/stages/line_prep.cuh"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;
const fs::path kModelDir = CUDAOCR_MODEL_DIR;

#define REQUIRE_MODEL()                                                             \
    if (!fs::exists(kModelDir / "rec.onnx") || !fs::exists(kModelDir / "dict.txt")) \
    GTEST_SKIP() << "models not fetched; run scripts/fetch_models.sh"

struct Page {
    HostImage gray;  // deskewed gray, what the recognizer reads
    std::vector<LineResult> lines;
};

Page detect(const std::string& fixture) {
    auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / (fixture + ".png")).string());
    EXPECT_TRUE(img.ok());
    auto pipe = Pipeline::create(PipelineConfig{});
    EXPECT_TRUE(pipe.ok());
    StageImages keep;
    auto r = pipe.value().run(img.value(), &keep);
    EXPECT_TRUE(r.ok());
    return {keep.deskewedGray, r.value().lines};
}

std::unique_ptr<OnnxLineRecognizer> make(const std::string& ep, int maxBatch = 16) {
    auto r = OnnxLineRecognizer::create(Params()
                                            .set("model", (kModelDir / "rec.onnx").string())
                                            .set("dict", (kModelDir / "dict.txt").string())
                                            .set("ep", ep)
                                            .set("maxBatch", maxBatch));
    EXPECT_TRUE(r.ok()) << r.status().message;
    return r.ok() ? r.take() : nullptr;
}

std::vector<CtcProbs> recognize(LineRecognizer& rec, const Page& page) {
    auto d = DeviceImage::fromHost(page.gray);
    EXPECT_TRUE(d.ok());
    StageContext ctx;
    auto r = rec.recognize(d.value(), page.lines, ctx);
    EXPECT_TRUE(r.ok()) << r.status().message;
    return r.ok() ? r.take() : std::vector<CtcProbs>{};
}

std::vector<std::string> texts(LineRecognizer& rec, const std::vector<CtcProbs>& probs) {
    CtcGreedyDecoder dec;
    std::vector<std::string> out;
    for (const auto& p : probs) {
        auto d = dec.decode(p, rec.charset());
        EXPECT_TRUE(d.ok()) << d.status().message;
        out.push_back(d.ok() ? d.value().text : "<error>");
    }
    return out;
}

std::string groundTruth(const std::string& fixture) {
    std::ifstream in(fs::path(CUDAOCR_SYNTH_DIR) / (fixture + ".gt.txt"));
    std::string line;
    std::getline(in, line);
    return line;
}

TEST(OnnxRecognizer, ReadsLineFixturesExactlyOnBothProviders) {
    REQUIRE_MODEL();
    for (const char* ep : {"cpu", "cuda"}) {
        auto rec = make(ep);
        ASSERT_TRUE(rec);
        for (const char* f : {"line_pangram_sans", "line_digits_serif", "line_punct_mono"}) {
            SCOPED_TRACE(std::string(ep) + " " + f);
            const Page page = detect(f);
            ASSERT_EQ(page.lines.size(), 1u);
            EXPECT_EQ(texts(*rec, recognize(*rec, page)), std::vector<std::string>{groundTruth(f)});
        }
    }
}

// Measured: max |cpu - cuda| = 1.8e-5 with TF32 off (default); 0.065 with ORT's
// default TF32 on, which is why tf32 is opt-in.
TEST(OnnxRecognizer, CpuAndCudaProbabilitiesAgree) {
    REQUIRE_MODEL();
    auto cpu = make("cpu"), cuda = make("cuda");
    ASSERT_TRUE(cpu && cuda);
    const Page page = detect("page_sans_clean");
    const auto a = recognize(*cpu, page), b = recognize(*cuda, page);
    ASSERT_EQ(a.size(), b.size());
    double maxDiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i].timesteps(), b[i].timesteps());
        ASSERT_EQ(a[i].classes(), b[i].classes());
        for (size_t k = 0; k < a[i].data().size(); ++k)
            maxDiff = std::max(maxDiff, std::abs(double(a[i].data()[k]) - b[i].data()[k]));
    }
    RecordProperty("maxAbsDiff", std::to_string(maxDiff));
    EXPECT_LE(maxDiff, 1e-3);
}

TEST(OnnxRecognizer, OutputMatchesInputOrderAndLineWidth) {
    REQUIRE_MODEL();
    auto rec = make("cuda", /*maxBatch=*/3);  // forces several sorted batches
    const Page page = detect("page_serif_clean");
    const auto probs = recognize(*rec, page);
    ASSERT_EQ(probs.size(), page.lines.size());
    std::vector<BBox> boxes;
    for (const auto& l : page.lines) boxes.push_back(l.box);
    auto plan = planLineCrops(boxes, page.gray.width, page.gray.height, LinePrepParams{});
    ASSERT_TRUE(plan.ok());
    for (size_t i = 0; i < probs.size(); ++i) {
        EXPECT_EQ(probs[i].timesteps(), plan.value()[i].outW / 8) << "line " << i;
        EXPECT_EQ(probs[i].classes(), 438);
    }
    // Batch size must not change the result.
    auto one = make("cuda", /*maxBatch=*/1);
    EXPECT_EQ(texts(*rec, probs), texts(*one, recognize(*one, page)));
}

TEST(OnnxRecognizer, CharsetIsBlankDictSpace) {
    REQUIRE_MODEL();
    auto rec = make("cpu");
    ASSERT_TRUE(rec);
    const Charset& cs = rec->charset();
    ASSERT_EQ(cs.symbols.size(), 438u);
    EXPECT_EQ(cs.blank, 0);
    EXPECT_EQ(cs.symbols[1], "0");
    EXPECT_EQ(cs.symbols.back(), " ");
}

// Regression (M4.6): a 2500 x 4 px rule detected as a line made the recognizer
// request gigabytes of GPU memory. With the width cap it costs at most 768
// timesteps and the call succeeds.
TEST(OnnxRecognizer, DegenerateThinLineIsCappedNotFatal) {
    REQUIRE_MODEL();
    auto rec = make("cuda");
    ASSERT_TRUE(rec);
    HostImage page = HostImage::create(2560, 400, 1, 255);
    for (int y = 100; y < 104; ++y)
        for (int x = 0; x < 2500; ++x) page.at(x, y) = 0;
    LineResult rule;
    rule.box = {0, 100, 2499, 103};
    auto d = DeviceImage::fromHost(page);
    ASSERT_TRUE(d.ok());
    StageContext ctx;
    auto r = rec->recognize(d.value(), std::vector<LineResult>{rule}, ctx);
    ASSERT_TRUE(r.ok()) << r.status().message;
    ASSERT_EQ(r.value().size(), 1u);
    EXPECT_LE(r.value()[0].timesteps(), 6144 / 8);
}

TEST(OnnxRecognizer, NoLinesNoWork) {
    REQUIRE_MODEL();
    auto rec = make("cpu");
    auto d = DeviceImage::create(10, 10, 1);
    ASSERT_TRUE(d.ok());
    StageContext ctx;
    auto r = rec->recognize(d.value(), std::vector<LineResult>{}, ctx);
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(r.value().empty());
}

TEST(OnnxRecognizer, RejectsBadParams) {
    REQUIRE_MODEL();
    const std::string model = (kModelDir / "rec.onnx").string();
    EXPECT_FALSE(OnnxLineRecognizer::create(Params().set("model", model).set("ep", "tpu")).ok());
    EXPECT_FALSE(OnnxLineRecognizer::create(Params().set("model", model).set("maxBatch", 0)).ok());
    EXPECT_FALSE(OnnxLineRecognizer::create(Params().set("model", model).set("modle", "x")).ok());
    EXPECT_FALSE(
        OnnxLineRecognizer::create(Params().set("model", model).set("tf32", "maybe")).ok());
    auto missing = OnnxLineRecognizer::create(Params().set("model", "/nonexistent/rec.onnx"));
    ASSERT_FALSE(missing.ok());
    EXPECT_NE(missing.status().message.find("/nonexistent/rec.onnx"), std::string::npos);
    // A dictionary that doesn't match the model's 438 classes.
    const fs::path shortDict = fs::temp_directory_path() / "cudaocr_short_dict.txt";
    std::ofstream(shortDict) << "a\nb\n";
    EXPECT_FALSE(
        OnnxLineRecognizer::create(Params().set("model", model).set("dict", shortDict.string()))
            .ok());
    fs::remove(shortDict);
}

TEST(OnnxRecognizer, RegisteredAsOnnx) {
    REQUIRE_MODEL();
    auto reg = ComponentRegistry::withBuiltins();
    auto rec = reg.make<LineRecognizer>(
        {"onnx", Params().set("model", (kModelDir / "rec.onnx").string()).set("ep", "cpu")});
    ASSERT_TRUE(rec.ok()) << rec.status().message;
    EXPECT_EQ(rec.value()->name(), "onnx");
}

}  // namespace
}  // namespace cudaocr
