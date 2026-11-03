// M4.7: PDF pages rendered to gray images (poppler-cpp, optional at build time).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "cudaocr/core/pdf.hpp"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;
const std::string kPdf = (fs::path(CUDAOCR_PDF_FIXTURE_DIR) / "two_pages.pdf").string();

#define REQUIRE_PDF() \
    if (!PdfDocument::supported()) GTEST_SKIP() << "built without poppler-cpp"

TEST(Pdf, PageCount) {
    REQUIRE_PDF();
    auto doc = PdfDocument::open(kPdf);
    ASSERT_TRUE(doc.ok()) << doc.status().message;
    EXPECT_EQ(doc.value().pageCount(), 2);
}

TEST(Pdf, RendersAt300DpiLikeTheSourceScan) {
    REQUIRE_PDF();
    auto doc = PdfDocument::open(kPdf);
    ASSERT_TRUE(doc.ok());
    auto src = loadImage(std::string(CUDAOCR_SYNTH_DIR) + "/page_sans_clean.png");
    ASSERT_TRUE(src.ok());
    auto page = doc.value().renderPage(0, 300.0);
    ASSERT_TRUE(page.ok()) << page.status().message;
    const HostImage& p = page.value();
    EXPECT_EQ(p.channels, 1);
    EXPECT_NEAR(p.width, src.value().width, 1);
    EXPECT_NEAR(p.height, src.value().height, 1);
    // Mean abs difference in gray levels. Measured 7.07: JPEG embedding alone is
    // 1.00, the rest is poppler's image resampling (pdftoppm -r 300 gives the
    // same 7.07; no 1-px shift matches better, so it is not misalignment).
    double diff = 0;
    const int w = std::min(p.width, src.value().width), h = std::min(p.height, src.value().height);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) diff += std::abs(int(p.at(x, y)) - int(src.value().at(x, y)));
    EXPECT_LT(diff / (double(w) * h), 10.0);
}

TEST(Pdf, DpiScalesThePage) {
    REQUIRE_PDF();
    auto doc = PdfDocument::open(kPdf);
    ASSERT_TRUE(doc.ok());
    auto half = doc.value().renderPage(1, 150.0);
    ASSERT_TRUE(half.ok()) << half.status().message;
    EXPECT_NEAR(half.value().width, 1252 / 2, 1);
    EXPECT_NEAR(half.value().height, 159 / 2, 1);
}

TEST(Pdf, ErrorsAreClear) {
    REQUIRE_PDF();
    auto missing = PdfDocument::open("/nonexistent/scan.pdf");
    ASSERT_FALSE(missing.ok());
    EXPECT_EQ(missing.status().code, ErrorCode::IoError);
    EXPECT_NE(missing.status().message.find("/nonexistent/scan.pdf"), std::string::npos);

    auto notPdf = PdfDocument::open(std::string(CUDAOCR_SYNTH_DIR) + "/page_sans_clean.png");
    EXPECT_FALSE(notPdf.ok());

    auto doc = PdfDocument::open(kPdf);
    ASSERT_TRUE(doc.ok());
    for (int bad : {-1, 2}) {
        auto r = doc.value().renderPage(bad, 300.0);
        ASSERT_FALSE(r.ok()) << bad;
        EXPECT_EQ(r.status().code, ErrorCode::InvalidArgument);
    }
    for (double dpi : {0.0, -5.0, 2400.1}) EXPECT_FALSE(doc.value().renderPage(0, dpi).ok()) << dpi;
}

TEST(Pdf, UnsupportedBuildSaysHowToEnable) {
    if (PdfDocument::supported()) GTEST_SKIP() << "built with poppler-cpp";
    auto doc = PdfDocument::open(kPdf);
    ASSERT_FALSE(doc.ok());
    EXPECT_NE(doc.status().message.find("libpoppler-cpp-dev"), std::string::npos);
}

}  // namespace
}  // namespace cudaocr
