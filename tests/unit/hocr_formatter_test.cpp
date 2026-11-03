// M4.2: hOCR output (docs/output-formats.md).
#include <gtest/gtest.h>

#include "cudaocr/components/hocr_formatter.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

TEST(HocrFormatter, ExactOutputWithEscaping) {
    PageResult p;
    p.width = 200;
    p.height = 100;
    LineResult l;
    l.box = {10, 20, 149, 39};
    l.conf = 0.875f;
    l.text = "a<b & \"c\"";
    WordResult w1{{10, 20, 49, 39}, "a<b", 0.5f, {}};
    WordResult w2{{60, 20, 70, 39}, "&", 0.994f, {}};
    WordResult w3{{}, "\"c\"", 1.0f, {}};  // unknown geometry: no bbox
    l.words = {w1, w2, w3};
    p.lines.push_back(l);
    LineResult textOnly;  // recognized line without words
    textOnly.box = {10, 50, 99, 69};
    textOnly.text = "it's";
    textOnly.conf = 0.2f;
    p.lines.push_back(textOnly);

    auto f = HocrFormatter::create({});
    ASSERT_TRUE(f.ok());
    const std::string want =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" "
        "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
        "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n"
        " <head>\n"
        "  <title>cudaocr</title>\n"
        "  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/>\n"
        "  <meta name=\"ocr-system\" content=\"cudaocr\"/>\n"
        "  <meta name=\"ocr-capabilities\" content=\"ocr_page ocr_line ocrx_word\"/>\n"
        " </head>\n"
        " <body>\n"
        "  <div class=\"ocr_page\" id=\"page_1\" title=\"bbox 0 0 200 100\">\n"
        "   <span class=\"ocr_line\" id=\"line_1_1\" title=\"bbox 10 20 150 40; x_wconf 88\">\n"
        "    <span class=\"ocrx_word\" id=\"word_1_1_1\" title=\"bbox 10 20 50 40; x_wconf 50\">"
        "a&lt;b</span>\n"
        "    <span class=\"ocrx_word\" id=\"word_1_1_2\" title=\"bbox 60 20 71 40; x_wconf 99\">"
        "&amp;</span>\n"
        "    <span class=\"ocrx_word\" id=\"word_1_1_3\" title=\"x_wconf "
        "100\">&quot;c&quot;</span>\n"
        "   </span>\n"
        "   <span class=\"ocr_line\" id=\"line_1_2\" title=\"bbox 10 50 100 70; x_wconf 20\">"
        "it&apos;s</span>\n"
        "  </div>\n"
        " </body>\n"
        "</html>\n";
    EXPECT_EQ(f.value()->format(p).value(), want);
    EXPECT_EQ(f.value()->extension(), "hocr");
}

TEST(HocrFormatter, EmptyPage) {
    auto f = HocrFormatter::create({});
    const std::string out = f.value()->format(PageResult{}).value();
    EXPECT_NE(out.find("<div class=\"ocr_page\" id=\"page_1\" title=\"bbox 0 0 0 0\">\n  </div>"),
              std::string::npos)
        << out;
}

TEST(HocrFormatter, RegisteredAsHocr) {
    EXPECT_TRUE(ComponentRegistry::withBuiltins().make<Formatter>({"hocr", {}}).ok());
}

}  // namespace
}  // namespace cudaocr
