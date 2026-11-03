#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "cudaocr/config.hpp"
#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/logger.hpp"
#include "cudaocr/core/pdf.hpp"
#include "cudaocr/pipeline.hpp"

namespace {

using namespace cudaocr;

void usage() {
    std::printf(
        "usage: cudaocr <image|pdf|dir>... [options]\n"
        "Prints the recognized text to stdout (diagnostics go to stderr). Several images or a\n"
        "directory are processed with one pipeline: -o then names an output directory\n"
        "(<stem>.<ext> per page; PDF pages are <stem>-p001, ...); without -o, txt pages are\n"
        "separated by form feeds.\n"
        "  --pdf-dpi <n>         render PDF pages at n dpi (default 300)\n"
        "  --config <file.toml>  pipeline config (docs/config.md); flags below override it\n"
        "  --print-config        print the effective config as TOML and exit\n"
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

// Command-line overrides, applied on top of the defaults and --config. Values
// stay text so the components' Params validation reports bad ones (e.g.
// "--median 4") instead of atoi silently turning them into 0. Only flags that
// were given change anything.
struct Overrides {
    std::optional<std::string> median, blockSize, C, ep, modelDir;
    bool equalize = false, otsu = false, noDeskew = false, noRecognize = false, tf32 = false;
};

// CLI defaults: the library defaults plus recognition when built with ORT.
PipelineConfig cliDefaults() {
    PipelineConfig cfg;
#ifdef CUDAOCR_WITH_ORT
    cfg.recognizer = ComponentSpec{"onnx", {}};
#endif
    return cfg;
}

Status applyOverrides(PipelineConfig& cfg, const Overrides& o) {
    auto find = [&](const char* name) {
        return std::find_if(cfg.preprocess.begin(), cfg.preprocess.end(),
                            [&](const ComponentSpec& s) { return s.name == name; });
    };
    if (o.median) {
        auto it = find("median");
        if (*o.median == "0") {
            if (it != cfg.preprocess.end()) cfg.preprocess.erase(it);
        } else if (it != cfg.preprocess.end()) {
            it->params.set("k", *o.median);
        } else {
            cfg.preprocess.insert(cfg.preprocess.begin(),
                                  ComponentSpec{"median", Params().set("k", *o.median)});
        }
    }
    if (o.equalize && find("equalize") == cfg.preprocess.end())
        cfg.preprocess.insert(cfg.preprocess.begin(), ComponentSpec{"equalize", {}});
    if (o.otsu && (o.blockSize || o.C))
        return Status::Error(ErrorCode::InvalidArgument,
                             "--otsu conflicts with --block-size/--C (adaptive threshold)");
    if (o.otsu) cfg.binarizer = {"otsu", {}};
    if (o.blockSize || o.C) {
        if (cfg.binarizer.name != "adaptive") cfg.binarizer = {"adaptive", {}};
        if (o.blockSize) cfg.binarizer.params.set("blockSize", *o.blockSize);
        if (o.C) cfg.binarizer.params.set("C", *o.C);
    }
    if (o.noDeskew) cfg.deskewer.reset();
    if (o.noRecognize) cfg.recognizer.reset();
    if (o.ep || o.modelDir || o.tf32) {
        if (!cfg.recognizer) {
            std::fprintf(stderr, "warn: recognition is off; --ep/--model-dir/--tf32 ignored\n");
        } else {
            Params& p = cfg.recognizer->params;
            if (o.ep) p.set("ep", *o.ep);
            if (o.tf32) p.set("tf32", true);
            if (o.modelDir)
                p.set("model", *o.modelDir + "/rec.onnx").set("dict", *o.modelDir + "/dict.txt");
        }
    }
    return Status::Good();
}

namespace fs = std::filesystem;

bool isImageFile(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* e : {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".pgm", ".ppm", ".pnm"})
        if (ext == e) return true;
    return false;
}

bool isPdf(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".pdf";
}

// One page to OCR: an image file, or page `pdfPage` (0-based) of a PDF.
struct PageSource {
    fs::path path;
    int pdfPage = -1;
    std::string stem;  // output name: file stem, or "<pdf stem>-p001" for PDF pages
    std::string label() const {
        return pdfPage < 0 ? path.string() : path.string() + " page " + std::to_string(pdfPage + 1);
    }
};

// Files as given; directories expand to their image and PDF files (non-recursive,
// sorted); PDFs expand to one entry per page. Opened PDFs are kept in `pdfs`.
std::vector<PageSource> expandInputs(const std::vector<std::string>& inputs, bool& sawDirectory,
                                     std::map<std::string, PdfDocument>& pdfs, int& openErrors) {
    std::vector<fs::path> files;
    sawDirectory = false;
    for (const auto& in : inputs) {
        std::error_code ec;
        if (fs::is_directory(in, ec)) {
            sawDirectory = true;
            std::vector<fs::path> found;
            for (const auto& e : fs::directory_iterator(in, ec))
                if (e.is_regular_file() && (isImageFile(e.path()) || isPdf(e.path())))
                    found.push_back(e.path());
            std::sort(found.begin(), found.end());
            files.insert(files.end(), found.begin(), found.end());
        } else {
            files.emplace_back(in);  // unreadable images are reported when loaded
        }
    }
    std::vector<PageSource> pages;
    for (const auto& f : files) {
        if (!isPdf(f)) {
            pages.push_back({f, -1, f.stem().string()});
            continue;
        }
        auto doc = PdfDocument::open(f.string());
        if (!doc.ok()) {
            std::fprintf(stderr, "error: %s\n", doc.status().message.c_str());
            ++openErrors;
            continue;
        }
        const int n = doc.value().pageCount();
        for (int i = 0; i < n; ++i) {
            char suffix[16];
            std::snprintf(suffix, sizeof suffix, "-p%03d", i + 1);
            pages.push_back({f, i, f.stem().string() + suffix});
        }
        pdfs.emplace(f.string(), doc.take());
    }
    return pages;
}

void dumpStages(const std::string& dir, const StageImages& stages, const PageResult& r) {
    fs::create_directories(dir);
    auto save = [&](const std::string& name, const HostImage& im) {
        if (im.empty()) return;
        Status s = saveImage(dir + "/" + name, im);
        if (!s.ok()) std::fprintf(stderr, "warn: %s\n", s.message.c_str());
    };
    save("01_gray.png", stages.gray);
    save("02_binary.png", stages.binary);
    save("03_deskewed.png", stages.deskewed);
    save("05_deskewed_gray.png", stages.deskewedGray);
    // Overlay: deskewed binary as RGB with line bands (blue), words (red), glyphs (green).
    if (stages.deskewed.empty()) return;
    const HostImage& base = stages.deskewed;
    HostImage overlay = HostImage::create(base.width, base.height, 3);
    for (int y = 0; y < overlay.height; ++y)
        for (int x = 0; x < overlay.width; ++x)
            for (int c = 0; c < 3; ++c) overlay.at(x, y, c) = base.at(x, y);
    for (const auto& line : r.lines) {
        drawRect(overlay, 0, line.box.y0, overlay.width - 1, line.box.y1, 80, 80, 255);
        for (const auto& w : line.words) {
            for (const auto& g : w.glyphs) drawRect(overlay, g.x0, g.y0, g.x1, g.y1, 0, 200, 0);
            if (!w.box.empty())
                drawRect(overlay, w.box.x0, w.box.y0, w.box.x1, w.box.y1, 255, 0, 0);
        }
    }
    save("04_overlay.png", overlay);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::vector<std::string> inputs;
    double pdfDpi = 300.0;
    std::string outPath, format, dumpDir, configPath;
    bool verbose = false, printConfig = false;
    Overrides knobs;
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
            knobs.noRecognize = true;
        } else if (a == "--ep") {
            if (const char* v = next()) knobs.ep = v;
        } else if (a == "--model-dir") {
            if (const char* v = next()) knobs.modelDir = v;
        } else if (a == "--tf32") {
            knobs.tf32 = true;
        } else if (a == "--config") {
            if (const char* v = next()) configPath = v;
        } else if (a == "--pdf-dpi") {
            const char* v = next();
            char* end = nullptr;
            pdfDpi = v ? std::strtod(v, &end) : 0;
            if (!v || *end != '\0' || !(pdfDpi > 0 && pdfDpi <= 2400)) {
                std::fprintf(stderr, "error: --pdf-dpi must be a number in (0, 2400]\n");
                return 1;
            }
        } else if (a == "--print-config") {
            printConfig = true;
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
            knobs.noDeskew = true;
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
            inputs.push_back(a);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage();
            return 1;
        }
    }
    PipelineConfig cfg = cliDefaults();
    if (!configPath.empty()) {
        auto loaded = loadPipelineConfig(configPath, cfg);
        if (!loaded.ok()) {
            std::fprintf(stderr, "config error: %s\n", loaded.status().message.c_str());
            return 1;
        }
        cfg = loaded.take();
    }
    if (Status s = applyOverrides(cfg, knobs); !s.ok()) {
        std::fprintf(stderr, "config error: %s\n", s.message.c_str());
        return 1;
    }
    if (printConfig) {
        std::fputs(toToml(cfg).c_str(), stdout);
        return 0;
    }
    if (inputs.empty()) {
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

    // Only litter a log file when explicitly debugging; warnings/errors hit stderr anyway.
    if (level <= LogLevel::Debug) Logger::init("cudaocr.log");
    Logger::setLevel(level);

    bool sawDirectory = false;
    std::map<std::string, PdfDocument> pdfs;
    int loadErrors = 0;
    const std::vector<PageSource> pages = expandInputs(inputs, sawDirectory, pdfs, loadErrors);
    if (pages.empty()) {
        if (!loadErrors) std::fprintf(stderr, "error: no images found in the given directories\n");
        return 1;
    }
    // Several pages (or any directory): -o names an output directory. A single
    // page writes into a directory too when -o is an existing directory or ends in '/'.
    const bool multi = pages.size() > 1 || sawDirectory;
    std::error_code dirEc;
    const bool outIsDir =
        !outPath.empty() && (multi || fs::is_directory(outPath, dirEc) || outPath.back() == '/');
    if (multi && outPath.empty() && format != "txt") {
        std::fprintf(stderr, "error: --format %s with several pages needs -o <dir>\n",
                     format.c_str());
        return 1;
    }
    if (outIsDir) {
        std::set<std::string> stems;
        for (const auto& p : pages)
            if (!stems.insert(p.stem).second) {
                std::fprintf(stderr, "error: two inputs named '%s'; outputs would collide in %s\n",
                             p.stem.c_str(), outPath.c_str());
                return 1;
            }
        std::error_code ec;
        fs::create_directories(outPath, ec);
        if (ec) {
            std::fprintf(stderr, "error: cannot create %s: %s\n", outPath.c_str(),
                         ec.message().c_str());
            return 1;
        }
    }

    // One pipeline (and one ORT session) for every page.
    auto pipeline = Pipeline::create(cfg);
    if (!pipeline.ok()) {
        std::fprintf(stderr, "config error: %s\n", pipeline.status().message.c_str());
        return 1;
    }
    auto fmt = ComponentRegistry::withBuiltins().make<Formatter>({format, {}});
    if (!fmt.ok()) {
        std::fprintf(stderr, "format error: %s\n", fmt.status().message.c_str());
        return 1;
    }

    int runErrors = 0, noText = 0, done = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const PageSource& page : pages) {
        const std::string path = page.label();
        Logger::log(LogLevel::Info, "processing " + path);
        auto img = page.pdfPage < 0 ? loadImage(page.path.string())
                                    : pdfs.at(page.path.string()).renderPage(page.pdfPage, pdfDpi);
        if (!img.ok()) {
            std::fprintf(stderr, "error: %s\n", img.status().message.c_str());
            ++loadErrors;
            continue;
        }
        StageImages stages;
        auto res = pipeline.value().run(img.value(), dumpDir.empty() ? nullptr : &stages);
        if (!res.ok()) {
            std::fprintf(stderr, "pipeline error: %s: %s\n", path.c_str(),
                         res.status().message.c_str());
            ++runErrors;
            continue;
        }
        const PageResult& r = res.value();
        ++done;

        if (verbose) {
            if (multi) std::fprintf(stderr, "== %s\n", path.c_str());
            std::fprintf(stderr, "image:        %dx%d (%d channel%s)\n", img.value().width,
                         img.value().height, img.value().channels,
                         img.value().channels > 1 ? "s" : "");
            std::fprintf(stderr, "skew angle:   %+.2f deg%s\n", r.skewDeg,
                         cfg.deskewer ? " (corrected)" : "");
            if (auto it = r.metrics.find("otsu.threshold"); it != r.metrics.end())
                std::fprintf(stderr, "otsu thresh:  %d\n", static_cast<int>(it->second));
            std::fprintf(stderr, "polarity:     %s\n",
                         r.inverted ? "light-on-dark (inverted)" : "dark-on-light");
            std::fprintf(stderr, "lines: %zu   words: %d   chars: %d\n", r.lines.size(),
                         r.wordCount(), glyphCount(r));
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

        auto formatted = fmt.value()->format(r);
        if (!formatted.ok()) {
            std::fprintf(stderr, "format error: %s: %s\n", path.c_str(),
                         formatted.status().message.c_str());
            ++runErrors;
            continue;
        }
        const std::string& output = formatted.value();
        std::string target = outPath;
        if (outIsDir)
            target = (fs::path(outPath) / (page.stem + "." + std::string(fmt.value()->extension())))
                         .string();
        if (target.empty()) {
            if (multi && done > 1) std::fputs("\f", stdout);  // page separator (as Tesseract)
            std::fwrite(output.data(), 1, output.size(), stdout);
        } else {
            std::ofstream f(target, std::ios::binary);
            f << output;
            if (!f) {
                std::fprintf(stderr, "error: cannot write %s\n", target.c_str());
                ++runErrors;
                continue;
            }
            if (verbose) std::fprintf(stderr, "wrote %s\n", target.c_str());
        }
        if (!dumpDir.empty()) {
            const std::string dir = multi ? dumpDir + "/" + page.stem : dumpDir;
            dumpStages(dir, stages, r);
            if (verbose) std::fprintf(stderr, "stage dumps in %s/\n", dir.c_str());
        }

        // Nothing found: no text when recognizing, no lines when detecting only.
        bool found = false;
        for (const auto& l : r.lines) found = found || !cfg.recognizer || !l.text.empty();
        if (!found) ++noText;
    }
    if (verbose && multi) {
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::fprintf(stderr, "pages: %d done, %d unreadable, %d failed, %d without text; %.0f ms\n",
                     done, loadErrors, runErrors, noText, ms);
    }
    Logger::close();
    // Worst outcome wins: 2 runtime error, 1 unreadable input, 3 no text, 0 ok.
    if (runErrors) return 2;
    if (loadErrors) return 1;
    return noText ? 3 : 0;
}
