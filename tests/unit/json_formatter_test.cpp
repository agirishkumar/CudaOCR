// M4.1: JSON output, schema version 1 (docs/output-formats.md).
#include <gtest/gtest.h>

#include "cudaocr/components/json_formatter.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

PageResult page() {
    PageResult p;
    p.width = 200;
    p.height = 100;
    p.skewDeg = -1.5f;
    p.inverted = true;
    p.metrics["otsu.threshold"] = 127;
    p.timings = {{"onnx", 2.5f}};
    LineResult l;
    l.box = {10, 20, 150, 40};
    l.text = "say \"hi\"\\ok\n";
    l.conf = 0.875f;
    WordResult w;
    w.box = {10, 20, 50, 40};
    w.text = "say";
    w.conf = 0.5f;
    l.words.push_back(w);
    WordResult g;  // detection-only word with glyph boxes
    g.box = {60, 20, 90, 40};
    g.glyphs = {{60, 20, 70, 40}};
    l.words.push_back(g);
    p.lines.push_back(l);
    return p;
}

TEST(JsonFormatter, Schema1ExactOutput) {
    auto f = JsonFormatter::create({});
    ASSERT_TRUE(f.ok());
    const std::string want =
        "{\n"
        "  \"schema\": \"cudaocr.page/1\",\n"
        "  \"width\": 200,\n"
        "  \"height\": 100,\n"
        "  \"skewDeg\": -1.5,\n"
        "  \"inverted\": true,\n"
        "  \"text\": \"say \\\"hi\\\"\\\\ok\\n\\n\",\n"
        "  \"metrics\": {\"otsu.threshold\": 127},\n"
        "  \"timingsMs\": {\"onnx\": 2.5},\n"
        "  \"lines\": [\n"
        "    {\"bbox\": [10, 20, 150, 40], \"conf\": 0.875, \"text\": \"say \\\"hi\\\"\\\\ok\\n\", "
        "\"words\": [\n"
        "      {\"bbox\": [10, 20, 50, 40], \"conf\": 0.5, \"text\": \"say\"},\n"
        "      {\"bbox\": [60, 20, 90, 40], \"conf\": 0, \"text\": \"\", "
        "\"glyphs\": [[60, 20, 70, 40]]}\n"
        "    ]}\n"
        "  ]\n"
        "}\n";
    EXPECT_EQ(f.value()->format(page()).value(), want);
    EXPECT_EQ(f.value()->extension(), "json");
}

TEST(JsonFormatter, EscapesControlCharsAndKeepsUtf8) {
    PageResult p;
    LineResult l;
    l.text = "caf\xC3\xA9\t\x01";
    p.lines.push_back(l);
    auto f = JsonFormatter::create({});
    const std::string out = f.value()->format(p).value();
    EXPECT_NE(out.find("\"caf\xC3\xA9\\t\\u0001\""), std::string::npos) << out;
}

TEST(JsonFormatter, EmptyPageHasEmptyArraysAndNoGeometry) {
    auto f = JsonFormatter::create({});
    const std::string out = f.value()->format(PageResult{}).value();
    EXPECT_NE(out.find("\"lines\": []"), std::string::npos) << out;
    EXPECT_NE(out.find("\"metrics\": {}"), std::string::npos) << out;
}

TEST(JsonFormatter, RegisteredAsJson) {
    EXPECT_TRUE(ComponentRegistry::withBuiltins().make<Formatter>({"json", {}}).ok());
}

}  // namespace
}  // namespace cudaocr
