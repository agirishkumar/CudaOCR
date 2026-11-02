// M4.3: PipelineConfig <-> TOML.
#include <gtest/gtest.h>

#include "cudaocr/config.hpp"

namespace cudaocr {
namespace {

std::string print(const PipelineConfig& c) {
    return toToml(c);
}

PipelineConfig parseOk(const std::string& toml) {
    auto r = parsePipelineConfig(toml, "test.toml");
    EXPECT_TRUE(r.ok()) << r.status().message;
    if (!r.ok()) return PipelineConfig{};
    return r.take();
}

std::string parseErr(const std::string& toml) {
    auto r = parsePipelineConfig(toml, "test.toml");
    EXPECT_FALSE(r.ok());
    if (!r.ok()) {
        EXPECT_EQ(r.status().code, ErrorCode::InvalidArgument);
    }
    return r.ok() ? "" : r.status().message;
}

TEST(Config, EmptyFileIsTheDefaultConfig) {
    EXPECT_EQ(print(parseOk("")), print(PipelineConfig{}));
}

TEST(Config, DefaultPrintsCanonically) {
    EXPECT_EQ(print(PipelineConfig{}),
              "# cudaocr pipeline config (cudaocr --print-config)\n"
              "toGray = { name = \"grayscale\" }\n"
              "preprocess = [\n"
              "  { name = \"median\", k = 3 },\n"
              "  { name = \"polarity\" },\n"
              "]\n"
              "binarizePrep = []\n"
              "binarizer = { name = \"adaptive\" }\n"
              "deskewer = { name = \"hough\" }\n"
              "detector = { name = \"projection\" }\n"
              "recognizer = false\n"
              "decoder = { name = \"ctc-greedy\" }\n");
}

TEST(Config, FullCustomConfigRoundTrips) {
    const std::string toml = R"(
        preprocess = [ { name = "equalize" }, { name = "polarity" } ]
        binarizePrep = [ { name = "gaussian", ksize = 5, sigma = 1.25 } ]
        binarizer = { name = "adaptive", blockSize = 41, C = 35 }
        deskewer = false
        detector = { name = "projection", minLineHeight = 4 }
        recognizer = { name = "onnx", ep = "cpu", tf32 = true, model = "/m/rec.onnx" }
    )";
    const PipelineConfig c = parseOk(toml);
    ASSERT_EQ(c.preprocess.size(), 2u);
    EXPECT_EQ(c.preprocess[0].name, "equalize");
    ASSERT_EQ(c.binarizePrep.size(), 1u);
    EXPECT_EQ(c.binarizePrep[0].params.get<double>("sigma", 0).value(), 1.25);
    EXPECT_EQ(c.binarizer.params.get<int>("C", 0).value(), 35);
    EXPECT_FALSE(c.deskewer.has_value());
    ASSERT_TRUE(c.recognizer.has_value());
    EXPECT_EQ(c.recognizer->params.get<std::string>("ep", "").value(), "cpu");
    EXPECT_TRUE(c.recognizer->params.get<bool>("tf32", false).value());
    EXPECT_EQ(c.recognizer->params.get<std::string>("model", "").value(), "/m/rec.onnx");
    // parse(print(cfg)) == cfg, and printing is a fixed point.
    const std::string once = print(c);
    EXPECT_EQ(print(parseOk(once)), once);
}

TEST(Config, PartialFileKeepsDefaultsForMissingKeys) {
    const PipelineConfig c = parseOk("binarizer = { name = \"otsu\" }\n");
    EXPECT_EQ(c.binarizer.name, "otsu");
    EXPECT_EQ(c.preprocess.size(), PipelineConfig{}.preprocess.size());
    EXPECT_TRUE(c.deskewer.has_value());
}

TEST(Config, StringThatLooksNumericSurvivesRoundTrip) {
    const PipelineConfig c = parseOk("recognizer = { name = \"onnx\", model = \"123\" }\n");
    EXPECT_EQ(print(parseOk(print(c))), print(c));
    EXPECT_EQ(c.recognizer->params.get<std::string>("model", "").value(), "123");
}

TEST(Config, ErrorsNameTheLocation) {
    EXPECT_NE(parseErr("binarizer = { name = \"adaptive\" \n").find("test.toml:"),
              std::string::npos);  // TOML syntax error with line/column
    const std::string unknown = parseErr("\n\nbinariser = { name = \"otsu\" }\n");
    EXPECT_NE(unknown.find("test.toml:3"), std::string::npos) << unknown;
    EXPECT_NE(unknown.find("binariser"), std::string::npos) << unknown;
    EXPECT_NE(parseErr("binarizer = { C = 3 }\n").find("name"), std::string::npos);
    EXPECT_NE(parseErr("preprocess = { name = \"median\" }\n").find("array"), std::string::npos);
    EXPECT_NE(parseErr("binarizer = { name = \"adaptive\", C = [1, 2] }\n").find("C"),
              std::string::npos);
    EXPECT_NE(parseErr("deskewer = true\n").find("deskewer"), std::string::npos);
}

TEST(Config, LoadReportsMissingFile) {
    auto r = loadPipelineConfig("/nonexistent/cfg.toml");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.status().code, ErrorCode::IoError);
    EXPECT_NE(r.status().message.find("/nonexistent/cfg.toml"), std::string::npos);
}

}  // namespace
}  // namespace cudaocr
