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
        "Prints the recognized text to stdout (diagnostics go to stderr).\n"
        "  -o <file>             write output to a file (.json -> JSON, .hocr/.html -> hOCR,\n"
        "                        else text)\n"
        "                        JSON schema: docs/output-formats.md\n"
        "  --format <txt|json|hocr>  output format (default: from -o's extension, else txt)\n"
        "  --no-recognize        detection only (no model needed); output has boxes, no text\n"
        "  --ep <cuda|cpu>       recognizer execution provider (default cuda)\n"
        "  --model-dir <dir>     directory with rec.onnx + dict.txt (default: $CUDAOCR_MODEL_DIR\n"
        "                        or the build's models/; fetch with scripts/fetch_models.sh)\n"
        "  --tf32                allow TF32 on the CUDA EP (faster, slightly less exact)\n"
        "  -v, --verbose         print a page summary to stderr\n"
        "  --dump-stages <dir>   write gray/binary/deskewed/overlay PNGs\n"
        "  --otsu                global Otsu instead of adaptive threshold\n"
        "  --block-size <n>      adaptive threshold window (odd, default 31)\n"
        "  --C <n>               adaptive threshold offset (default 40; 50 helps stained scans,\n"
        "                        > 50 loses faint text)\n"
        "  --median <0|3|5>      denoise kernel (default 3, 0 = off)\n"
        "  --equalize            histogram equalization before binarization\n"
        "  --no-deskew           skip skew detection/correction\n"
        "  --benchmark           print per-stage GPU timings to stderr\n"
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

// CLI knobs, kept as text so the components' Params validation reports bad
// values (e.g. "--median 4") instead of atoi silently turning them into 0.
struct Knobs {
    std::string median = "3";  // "0" = off
    bool equalize = false;
    bool otsu = false;
    std::string blockSize, C;  // empty = component default
    bool deskew = true;
#ifdef CUDAOCR_WITH_ORT
    bool recognize = true;
#else
    bool recognize = false;  // built without ONNX Runtime
#endif
    std::string ep = "cuda";
    std::string modelDir;  // empty = recognizer default
    bool tf32 = false;
};

PipelineConfig toConfig(const Knobs& k) {
    PipelineConfig cfg;
    cfg.preprocess.clear();
    cfg.binarizePrep.clear();
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
    if (k.recognize) {
        Params p;
        p.set("ep", k.ep).set("tf32", k.tf32);
        if (!k.modelDir.empty())
            p.set("model", k.modelDir + "/rec.onnx").set("dict", k.modelDir + "/dict.txt");
        cfg.recognizer = ComponentSpec{"onnx", p};
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string input, outPath, format, dumpDir;
    bool verbose = false;
    Knobs knobs;
    bool benchmark = false;
    LogLevel level = LogLevel::Info;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "-o") {
            if (const char* v = next()) outPath = v;
        } else if (a == "--format") {
            if (const char* v = next()) format = v;
        } else if (a == "--no-recognize") {
            knobs.recognize = false;
        } else if (a == "--ep") {
            if (const char* v = next()) knobs.ep = v;
        } else if (a == "--model-dir") {
            if (const char* v = next()) knobs.modelDir = v;
        } else if (a == "--tf32") {
            knobs.tf32 = true;
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
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
    if (format.empty()) {
        auto endsWith = [&](const char* ext) {
            const size_t n = std::strlen(ext);
            return outPath.size() >= n && outPath.compare(outPath.size() - n, n, ext) == 0;
        };
        format = endsWith(".json")                        ? "json"
                 : endsWith(".hocr") || endsWith(".html") ? "hocr"
                                                          : "txt";
    }
    if (format != "txt" && format != "json" && format != "hocr") {
        std::fprintf(stderr, "error: --format must be txt, json or hocr\n");
        return 1;
    }
#ifndef CUDAOCR_WITH_ORT
    if (knobs.ep != "cuda" || !knobs.modelDir.empty() || knobs.tf32)
        std::fprintf(stderr, "warn: built without ONNX Runtime; recognizer options ignored\n");
#endif

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

    if (verbose) {
        std::fprintf(stderr, "image:        %dx%d (%d channel%s)\n", img.value().width,
                     img.value().height, img.value().channels, img.value().channels > 1 ? "s" : "");
        std::fprintf(stderr, "skew angle:   %+.2f deg%s\n", r.skewDeg,
                     knobs.deskew ? " (corrected)" : "");
        if (auto it = r.metrics.find("otsu.threshold"); it != r.metrics.end())
            std::fprintf(stderr, "otsu thresh:  %d\n", static_cast<int>(it->second));
        std::fprintf(stderr, "polarity:     %s\n",
                     r.inverted ? "light-on-dark (inverted)" : "dark-on-light");
        std::fprintf(stderr, "lines: %zu   words: %d   chars: %d\n", r.lines.size(), r.wordCount(),
                     glyphCount(r));
    }
    if (benchmark) {
        std::fprintf(stderr, "\n%-20s %10s\n", "stage", "ms");
        float total = 0;
        for (const auto& t : r.timings) {
            std::fprintf(stderr, "%-20s %10.3f\n", t.name.c_str(), t.ms);
            total += t.ms;
        }
        std::fprintf(stderr, "%-20s %10.3f\n", "TOTAL", total);
    }

    auto fmt = ComponentRegistry::withBuiltins().make<Formatter>({format, {}});
    auto formatted = fmt.ok() ? fmt.value()->format(r) : Result<std::string>(fmt.status());
    if (!formatted.ok()) {
        std::fprintf(stderr, "format error: %s\n", formatted.status().message.c_str());
        return 2;
    }
    const std::string output = formatted.take();
    if (outPath.empty()) {
        std::fwrite(output.data(), 1, output.size(), stdout);
    } else {
        std::ofstream f(outPath, std::ios::binary);
        f << output;
        if (!f) {
            std::fprintf(stderr, "error: cannot write %s\n", outPath.c_str());
            return 2;
        }
        if (verbose) std::fprintf(stderr, "wrote %s\n", outPath.c_str());
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
        if (verbose) std::fprintf(stderr, "stage dumps in %s/\n", dumpDir.c_str());
    }

    Logger::close();
    // 3 = nothing found: no text when recognizing, no lines when detecting only.
    bool found = false;
    for (const auto& l : r.lines) found = found || !knobs.recognize || !l.text.empty();
    return found ? 0 : 3;
}
