#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/logger.hpp"
#include "cudaocr/pipeline.hpp"

namespace {

using namespace cudaocr;

void usage() {
    std::printf(
        "usage: cudaocr <input-image> [options]\n"
        "  -o <file.json>        write boxes/metadata as JSON\n"
        "  --dump-stages <dir>   write gray/binary/deskewed/overlay PNGs\n"
        "  --otsu                global Otsu instead of adaptive threshold\n"
        "  --block-size <n>      adaptive threshold window (odd, default 31)\n"
        "  --C <n>               adaptive threshold offset (default 10)\n"
        "  --median <0|3|5>      denoise kernel (default 3, 0 = off)\n"
        "  --equalize            histogram equalization before binarization\n"
        "  --no-deskew           skip skew detection/correction\n"
        "  --benchmark           print per-stage GPU timings\n"
        "  --log-level <trace|debug|info|warn|error>\n");
}

void drawRect(HostImage& img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g,
              std::uint8_t b) {
    auto set = [&](int x, int y) {
        if (x < 0 || x >= img.width || y < 0 || y >= img.height) return;
        img.at(x, y, 0) = r;
        img.at(x, y, 1) = g;
        img.at(x, y, 2) = b;
    };
    for (int x = x0; x <= x1; ++x) {
        set(x, y0);
        set(x, y1);
    }
    for (int y = y0; y <= y1; ++y) {
        set(x0, y);
        set(x1, y);
    }
}

int glyphCount(const PageResult& page) {
    int n = 0;
    for (const auto& l : page.lines)
        for (const auto& w : l.words) n += static_cast<int>(w.glyphs.size());
    return n;
}

std::string boxJson(const BBox& b) {
    return "\"x0\": " + std::to_string(b.x0) + ", \"y0\": " + std::to_string(b.y0) +
           ", \"x1\": " + std::to_string(b.x1) + ", \"y1\": " + std::to_string(b.y1);
}

std::string toJson(const PageResult& res) {
    std::string j = "{\n";
    j += "  \"skewAngleDeg\": " + std::to_string(res.skewDeg) + ",\n";
    j += "  \"wasInverted\": " + std::string(res.inverted ? "true" : "false") + ",\n";
    for (const auto& [key, value] : res.metrics)
        j += "  \"" + key + "\": " + std::to_string(value) + ",\n";
    j += "  \"lineCount\": " + std::to_string(res.lines.size()) + ",\n";
    j += "  \"wordCount\": " + std::to_string(res.wordCount()) + ",\n";
    j += "  \"charCount\": " + std::to_string(glyphCount(res)) + ",\n";
    j += "  \"lines\": [\n";
    for (size_t i = 0; i < res.lines.size(); ++i) {
        const auto& line = res.lines[i];
        j += "    {" + boxJson(line.box) + ", \"words\": [";
        for (size_t wi = 0; wi < line.words.size(); ++wi) {
            const auto& w = line.words[wi];
            j += "\n      {" + boxJson(w.box) + ", \"chars\": " + std::to_string(w.glyphs.size()) +
                 "}";
            if (wi + 1 < line.words.size()) j += ",";
        }
        j += "]}";
        if (i + 1 < res.lines.size()) j += ",";
        j += "\n";
    }
    j += "  ],\n  \"timingsMs\": {";
    for (size_t i = 0; i < res.timings.size(); ++i) {
        j += "\"" + res.timings[i].name + "\": " + std::to_string(res.timings[i].ms);
        if (i + 1 < res.timings.size()) j += ", ";
    }
    j += "}\n}\n";
    return j;
}

// CLI knobs, kept as text so the components' Params validation reports bad
// values (e.g. "--median 4") instead of atoi silently turning them into 0.
struct Knobs {
    std::string median = "3";  // "0" = off
    bool equalize = false;
    bool otsu = false;
    std::string blockSize, C;  // empty = component default
    bool deskew = true;
};

PipelineConfig toConfig(const Knobs& k) {
    PipelineConfig cfg;
    cfg.preprocess.clear();
    if (k.equalize) cfg.preprocess.push_back({"equalize", {}});
    if (k.median != "0") cfg.preprocess.push_back({"median", Params().set("k", k.median)});
    cfg.preprocess.push_back({"polarity", {}});
    if (k.otsu) {
        cfg.binarizer = {"otsu", {}};
    } else {
        Params p;
        if (!k.blockSize.empty()) p.set("blockSize", k.blockSize);
        if (!k.C.empty()) p.set("C", k.C);
        cfg.binarizer = {"adaptive", p};
    }
    if (!k.deskew) cfg.deskewer.reset();
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string input, jsonOut, dumpDir;
    Knobs knobs;
    bool benchmark = false;
    LogLevel level = LogLevel::Info;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "-o") {
            if (const char* v = next()) jsonOut = v;
        } else if (a == "--dump-stages") {
            if (const char* v = next()) dumpDir = v;
        } else if (a == "--otsu") {
            knobs.otsu = true;
        } else if (a == "--block-size") {
            if (const char* v = next()) knobs.blockSize = v;
        } else if (a == "--C") {
            if (const char* v = next()) knobs.C = v;
        } else if (a == "--median") {
            if (const char* v = next()) knobs.median = v;
        } else if (a == "--equalize") {
            knobs.equalize = true;
        } else if (a == "--no-deskew") {
            knobs.deskew = false;
        } else if (a == "--benchmark") {
            benchmark = true;
        } else if (a == "--log-level") {
            const char* v = next();
            if (v) {
                const std::string s = v;
                level = s == "trace"   ? LogLevel::Trace
                        : s == "debug" ? LogLevel::Debug
                        : s == "warn"  ? LogLevel::Warn
                        : s == "error" ? LogLevel::Error
                                       : LogLevel::Info;
            }
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a[0] != '-') {
            input = a;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage();
            return 1;
        }
    }
    if (input.empty()) {
        usage();
        return 1;
    }

    // Only litter a log file when explicitly debugging; warnings/errors hit stderr anyway.
    if (level <= LogLevel::Debug) Logger::init("cudaocr.log");
    Logger::setLevel(level);
    Logger::log(LogLevel::Info, "processing " + input);

    auto img = loadImage(input);
    if (!img.ok()) {
        std::fprintf(stderr, "error: %s\n", img.status().message.c_str());
        return 1;
    }

    auto pipeline = Pipeline::create(toConfig(knobs));
    if (!pipeline.ok()) {
        std::fprintf(stderr, "config error: %s\n", pipeline.status().message.c_str());
        return 1;
    }
    const bool keepStages = !dumpDir.empty();
    StageImages stages;
    auto res = pipeline.value().run(img.value(), keepStages ? &stages : nullptr);
    if (!res.ok()) {
        std::fprintf(stderr, "pipeline error: %s\n", res.status().message.c_str());
        return 2;
    }
    const PageResult& r = res.value();

    std::printf("image:        %dx%d (%d channel%s)\n", img.value().width, img.value().height,
                img.value().channels, img.value().channels > 1 ? "s" : "");
    std::printf("skew angle:   %+.2f deg%s\n", r.skewDeg, knobs.deskew ? " (corrected)" : "");
    if (auto it = r.metrics.find("otsu.threshold"); it != r.metrics.end())
        std::printf("otsu thresh:  %d\n", static_cast<int>(it->second));
    std::printf("polarity:     %s\n", r.inverted ? "light-on-dark (inverted)" : "dark-on-light");
    std::printf("lines: %zu   words: %d   chars: %d\n", r.lines.size(), r.wordCount(),
                glyphCount(r));
    if (benchmark) {
        std::printf("\n%-20s %10s\n", "stage", "ms");
        float total = 0;
        for (const auto& t : r.timings) {
            std::printf("%-20s %10.3f\n", t.name.c_str(), t.ms);
            total += t.ms;
        }
        std::printf("%-20s %10.3f\n", "TOTAL", total);
    }

    if (!jsonOut.empty()) {
        std::ofstream f(jsonOut);
        f << toJson(r);
        std::printf("wrote %s\n", jsonOut.c_str());
    }

    if (keepStages) {
        std::filesystem::create_directories(dumpDir);
        auto save = [&](const std::string& name, const HostImage& im) {
            if (im.empty()) return;
            Status s = saveImage(dumpDir + "/" + name, im);
            if (!s.ok()) std::fprintf(stderr, "warn: %s\n", s.message.c_str());
        };
        save("01_gray.png", stages.gray);
        save("02_binary.png", stages.binary);
        save("03_deskewed.png", stages.deskewed);
        save("05_deskewed_gray.png", stages.deskewedGray);
        // Overlay: deskewed binary as RGB with line bands (blue), words (red), chars (green).
        if (!stages.deskewed.empty()) {
            const HostImage& base = stages.deskewed;
            HostImage overlay = HostImage::create(base.width, base.height, 3);
            for (int y = 0; y < overlay.height; ++y)
                for (int x = 0; x < overlay.width; ++x)
                    for (int c = 0; c < 3; ++c) overlay.at(x, y, c) = base.at(x, y);
            for (const auto& line : r.lines) {
                drawRect(overlay, 0, line.box.y0, overlay.width - 1, line.box.y1, 80, 80, 255);
                for (const auto& w : line.words) {
                    for (const auto& g : w.glyphs)
                        drawRect(overlay, g.x0, g.y0, g.x1, g.y1, 0, 200, 0);
                    drawRect(overlay, w.box.x0, w.box.y0, w.box.x1, w.box.y1, 255, 0, 0);
                }
            }
            save("04_overlay.png", overlay);
        }
        std::printf("stage dumps in %s/\n", dumpDir.c_str());
    }

    Logger::close();
    return r.lines.empty() ? 3 : 0;
}
