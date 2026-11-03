// Golden pin of the pipeline's geometry output. The expected values were
// generated from the original hard-wired pipeline (runPipeline) when it was
// replaced by the component-based Pipeline in MOD.4, after a direct
// equivalence test passed on every case — so this test keeps that equivalence
// enforced now that the old code is gone. A change that alters detection on
// purpose must regenerate these values and say why in the commit.
//
// Intentional changes since generation:
// - 2026-11-02 repo cleanup: sample.png (personal), sample.jpg (copyrighted poster)
//   and vigorous-before.png (unknown licence) removed from resources/ with their
//   rows; sherlock.png rows added, pinned from the then-current pipeline.
// - M3.1 (rows marked "M3.1"): detectLines merges descender/i-dot slivers
//   (< 1/3 typical line height, within 1/3 line) into the adjacent line. Each
//   affected case lost exactly one 4-5 row band sitting 2 rows below a text
//   line (verified visually on noisy_text.png: italic f/p/g descenders).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "cudaocr/pipeline.hpp"
#include "oracles/oracles.hpp"

namespace cudaocr {
namespace {

// Knobs mirroring the CLI flags; toConfig is the same mapping the CLI uses.
struct Knobs {
    int median = 3;  // 0 = off
    bool equalize = false;
    bool otsu = false;
    int blockSize = 31, C = 10;
    bool deskew = true;
};

PipelineConfig toConfig(const Knobs& k) {
    PipelineConfig cfg;
    cfg.preprocess.clear();
    if (k.equalize) cfg.preprocess.push_back({"equalize", {}});
    if (k.median > 0) cfg.preprocess.push_back({"median", Params().set("k", k.median)});
    cfg.preprocess.push_back({"polarity", {}});
    cfg.binarizer =
        k.otsu ? ComponentSpec{"otsu", {}}
               : ComponentSpec{"adaptive", Params().set("blockSize", k.blockSize).set("C", k.C)};
    if (!k.deskew) cfg.deskewer.reset();
    return cfg;
}

// FNV-1a over skew bits, polarity and every line/word/glyph box, in order.
std::uint64_t digest(const PageResult& p) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::int64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<std::uint8_t>(v >> (8 * i));
            h *= 1099511628211ull;
        }
    };
    auto box = [&](const BBox& b) {
        mix(b.x0);
        mix(b.y0);
        mix(b.x1);
        mix(b.y1);
    };
    std::uint32_t skewBits;
    std::memcpy(&skewBits, &p.skewDeg, sizeof skewBits);
    mix(skewBits);
    mix(p.inverted);
    mix(static_cast<std::int64_t>(p.lines.size()));
    for (const auto& l : p.lines) {
        box(l.box);
        mix(static_cast<std::int64_t>(l.words.size()));
        for (const auto& w : l.words) {
            box(w.box);
            mix(static_cast<std::int64_t>(w.glyphs.size()));
            for (const auto& g : w.glyphs) box(g);
        }
    }
    return h;
}

struct Counts {
    int lines = 0, words = 0, glyphs = 0;
};
Counts count(const PageResult& p) {
    Counts c;
    c.lines = static_cast<int>(p.lines.size());
    for (const auto& l : p.lines)
        for (const auto& w : l.words) {
            ++c.words;
            c.glyphs += static_cast<int>(w.glyphs.size());
        }
    return c;
}

struct Golden {
    const char* name;
    int lines, words, glyphs;
    float skewDeg;
    bool inverted;
    std::uint64_t digest;
};

struct Case {
    std::string name;
    HostImage image;
    Knobs knobs;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    const auto page = oracle::makeSyntheticPage();
    Knobs crisp;
    crisp.median = 0;
    out.push_back({"synthetic_skewed", oracle::rotate(page.image, 2.5f, 255), crisp});
    Knobs inv = crisp;
    inv.deskew = false;
    out.push_back({"synthetic_inverted", oracle::invert(page.image), inv});
    Knobs otsu = inv;
    otsu.otsu = true;
    out.push_back({"synthetic_otsu", page.image, otsu});

    Knobs def, otsuNoDeskew, eqMedian5;
    otsuNoDeskew.otsu = true;
    otsuNoDeskew.deskew = false;
    eqMedian5.equalize = true;
    eqMedian5.median = 5;
    eqMedian5.blockSize = 15;
    eqMedian5.C = 4;
    for (const char* f : {"noisy_text.png", "noisy_text2.png", "sherlock.png"}) {
        auto img = loadImage(std::string(CUDAOCR_FIXTURE_DIR) + "/" + f);
        EXPECT_TRUE(img.ok()) << f;
        if (!img.ok()) continue;
        out.push_back({std::string(f) + "/default", img.value(), def});
        out.push_back({std::string(f) + "/otsu_nodeskew", img.value(), otsuNoDeskew});
        out.push_back({std::string(f) + "/equalize_median5", img.value(), eqMedian5});
    }
    return out;
}

// Generated from the legacy pipeline (MOD.4); see the header comment.
const Golden kGolden[] = {
    {"synthetic_skewed", 3, 9, 34, 2.5000f, false, 0x62f1d2a1789382faull},
    {"synthetic_inverted", 3, 9, 34, 0.0000f, true, 0x54d0666fb9d2b570ull},
    {"synthetic_otsu", 3, 9, 34, 0.0000f, false, 0xcdf55127979f49c9ull},
    {"noisy_text.png/default", 3, 12, 20, 0.0000f, false, 0x4d232622824ae89aull},
    {"noisy_text.png/otsu_nodeskew", 3, 16, 99, 0.0000f, false, 0xf257a9769275e1dcull},  // M3.1
    {"noisy_text.png/equalize_median5", 1, 1, 1, 0.0000f, false, 0x23834f81e5309b7dull},
    {"noisy_text2.png/default", 2, 2, 3, 0.0000f, false, 0x3a903881d09f40f4ull},
    {"noisy_text2.png/otsu_nodeskew", 8, 44, 216, 0.0000f, false, 0xb0cfcf5a211cfa06ull},  // M3.1
    {"noisy_text2.png/equalize_median5", 1, 1, 1, 0.0000f, true, 0xc9c1e938e878a717ull},
    // Added 2026-11-02 (repo cleanup): pinned from the current pipeline, not from
    // the legacy one (removed images: sample.png, sample.jpg, vigorous-before.png).
    {"sherlock.png/default", 32, 713, 2338, 0.0000f, false, 0x490f2cb01e945c83ull},
    {"sherlock.png/otsu_nodeskew", 31, 722, 4176, 0.0000f, false, 0x5384271af35f9657ull},
    {"sherlock.png/equalize_median5", 31, 667, 745, 0.0000f, false, 0xaf056b376dbdd9ebull},
};

TEST(PipelineGolden, MatchesPinnedLegacyOutput) {
    const auto all = cases();
    ASSERT_EQ(all.size(), std::size(kGolden));
    for (size_t i = 0; i < all.size(); ++i) {
        const Case& c = all[i];
        const Golden& g = kGolden[i];
        SCOPED_TRACE(c.name);
        ASSERT_EQ(c.name, g.name);
        auto pipe = Pipeline::create(toConfig(c.knobs));
        ASSERT_TRUE(pipe.ok()) << pipe.status().message;
        auto r = pipe.value().run(c.image);
        ASSERT_TRUE(r.ok()) << r.status().message;
        const Counts n = count(r.value());
        EXPECT_EQ(n.lines, g.lines);
        EXPECT_EQ(n.words, g.words);
        EXPECT_EQ(n.glyphs, g.glyphs);
        EXPECT_EQ(r.value().skewDeg, g.skewDeg);
        EXPECT_EQ(r.value().inverted, g.inverted);
        EXPECT_EQ(digest(r.value()), g.digest) << "boxes changed (counts may still match)";
    }
}

}  // namespace
}  // namespace cudaocr
