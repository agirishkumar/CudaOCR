// M3.1: the committed synthetic fixtures (scripts/gen_fixtures.py) must be
// consumable by the current pipeline: detection finds exactly the ground-truth
// line count, polarity and skew on every variant.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "cudaocr/pipeline.hpp"

namespace cudaocr {
namespace {

namespace fs = std::filesystem;

struct Fixture {
    std::string name, variant;
    float angle;
    int lines;
};

// manifest.tsv: name, font, size_pt, variant, angle_deg, lines (header row first).
std::vector<Fixture> manifest() {
    std::ifstream in(fs::path(CUDAOCR_SYNTH_DIR) / "manifest.tsv");
    std::vector<Fixture> out;
    std::string row;
    std::getline(in, row);  // header
    while (std::getline(in, row)) {
        std::istringstream ss(row);
        Fixture f;
        std::string font, size, angle, lines;
        std::getline(ss, f.name, '\t');
        std::getline(ss, font, '\t');
        std::getline(ss, size, '\t');
        std::getline(ss, f.variant, '\t');
        std::getline(ss, angle, '\t');
        std::getline(ss, lines, '\t');
        f.angle = std::stof(angle);
        f.lines = std::stoi(lines);
        out.push_back(f);
    }
    return out;
}

int gtLineCount(const std::string& name) {
    std::ifstream in(fs::path(CUDAOCR_SYNTH_DIR) / (name + ".gt.txt"));
    int n = 0;
    for (std::string l; std::getline(in, l);) ++n;
    return n;
}

TEST(SynthFixtures, ManifestMatchesFilesAndGroundTruth) {
    const auto all = manifest();
    ASSERT_GE(all.size(), 8u);
    for (const auto& f : all) {
        SCOPED_TRACE(f.name);
        EXPECT_TRUE(fs::exists(fs::path(CUDAOCR_SYNTH_DIR) / (f.name + ".png")));
        EXPECT_EQ(gtLineCount(f.name), f.lines);
    }
}

// Every variant, including the noisy page (fixed by the M3.11 default C = 40;
// it yielded 1/6 lines with C = 10), is segmented into the ground-truth line count.
TEST(SynthFixtures, AllVariantsHaveExpectedLineCount) {
    auto defaults = Pipeline::create(PipelineConfig{});
    ASSERT_TRUE(defaults.ok());
    int checked = 0;
    for (const auto& f : manifest()) {
        SCOPED_TRACE(f.name);
        auto img = loadImage((fs::path(CUDAOCR_SYNTH_DIR) / (f.name + ".png")).string());
        ASSERT_TRUE(img.ok()) << img.status().message;
        Pipeline& p = defaults.value();
        auto r = p.run(img.value());
        ASSERT_TRUE(r.ok()) << r.status().message;
        EXPECT_EQ(static_cast<int>(r.value().lines.size()), f.lines);
        EXPECT_EQ(r.value().inverted, f.variant == "inverted");
        // manifest angle is PIL's rotate() angle (counter-clockwise on screen);
        // the detector reports the page's skew with the opposite sign.
        EXPECT_NEAR(r.value().skewDeg, -f.angle, 0.5f);
        ++checked;
    }
    EXPECT_EQ(checked, 10);
}

}  // namespace
}  // namespace cudaocr
