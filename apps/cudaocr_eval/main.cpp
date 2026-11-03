// cudaocr-eval: score OCR output against ground truth.
//
//   cudaocr-eval <pred_dir> <gt_dir> [--gt-ext .gt.txt] [--pred-ext .txt] [--max-cer X]
//                [--ref-wildcard C]
//
// For every <name><gt-ext> in gt_dir, reads <name><pred-ext> from pred_dir (a
// missing prediction counts as empty output) and prints a markdown table of
// per-file and aggregate CER/WER. Aggregates are total edits / total reference
// length. With --max-cer, exits 1 if the aggregate CER exceeds X (0..1).
// --ref-wildcard C: reference character C matches any one output character
// (UNLV ground truth uses '~' for undeterminable characters).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cudaocr/eval/metrics.hpp"
#include "cudaocr/version.hpp"

namespace {

namespace fs = std::filesystem;
using cudaocr::eval::ErrorCounts;

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool readFile(const fs::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::string pct(const ErrorCounts& e) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f%%", 100.0 * e.rate());
    return buf;
}

std::string frac(const ErrorCounts& e) {
    return std::to_string(e.edits) + "/" + std::to_string(e.refLength);
}

int usage() {
    std::fprintf(stderr,
                 "usage: cudaocr-eval <pred_dir> <gt_dir> [--gt-ext .gt.txt] [--pred-ext .txt] "
                 "[--max-cer X] [--ref-wildcard C]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    std::string gtExt = ".gt.txt", predExt = ".txt";
    double maxCer = -1;
    char32_t wildcard = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--ref-wildcard") {
            const char* v = next();
            const std::u32string w = v ? cudaocr::eval::utf8Decode(v) : std::u32string{};
            if (w.size() != 1) return usage();
            wildcard = w[0];
        } else if (a == "--gt-ext" || a == "--pred-ext" || a == "--max-cer") {
            const char* v = next();
            if (!v) return usage();
            if (a == "--gt-ext")
                gtExt = v;
            else if (a == "--pred-ext")
                predExt = v;
            else {
                char* end = nullptr;
                maxCer = std::strtod(v, &end);
                if (*end != '\0' || maxCer < 0) return usage();
            }
        } else if (a == "--version") {
            std::printf("cudaocr-eval %s\n", cudaocr::version());
            return 0;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return usage();
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 2) return usage();
    const fs::path predDir = pos[0], gtDir = pos[1];
    if (!fs::is_directory(gtDir)) {
        std::fprintf(stderr, "error: not a directory: %s\n", gtDir.c_str());
        return 2;
    }

    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(gtDir)) {
        const std::string f = entry.path().filename().string();
        if (entry.is_regular_file() && endsWith(f, gtExt))
            names.push_back(f.substr(0, f.size() - gtExt.size()));
    }
    std::sort(names.begin(), names.end());
    if (names.empty()) {
        std::fprintf(stderr, "error: no *%s files in %s\n", gtExt.c_str(), gtDir.c_str());
        return 2;
    }

    ErrorCounts chars, words;
    int missing = 0;
    std::printf("| file | CER | char edits/ref | WER | word edits/ref | note |\n");
    std::printf("|---|---:|---:|---:|---:|---|\n");
    for (const auto& name : names) {
        std::string ref, hyp;
        if (!readFile(gtDir / (name + gtExt), ref)) {
            std::fprintf(stderr, "error: cannot read %s\n", (gtDir / (name + gtExt)).c_str());
            return 2;
        }
        const bool found = readFile(predDir / (name + predExt), hyp);
        if (!found) ++missing;
        const ErrorCounts c = cudaocr::eval::charErrors(ref, hyp, wildcard);
        const ErrorCounts w = cudaocr::eval::wordErrors(ref, hyp, wildcard);
        chars += c;
        words += w;
        std::printf("| %s | %s | %s | %s | %s | %s |\n", name.c_str(), pct(c).c_str(),
                    frac(c).c_str(), pct(w).c_str(), frac(w).c_str(),
                    found ? "" : "missing prediction");
    }
    std::printf("| **total** (%zu files) | **%s** | %s | **%s** | %s | missing: %d |\n",
                names.size(), pct(chars).c_str(), frac(chars).c_str(), pct(words).c_str(),
                frac(words).c_str(), missing);

    if (maxCer >= 0 && chars.rate() > maxCer) {
        std::fprintf(stderr, "FAIL: aggregate CER %.4f > --max-cer %.4f\n", chars.rate(), maxCer);
        return 1;
    }
    return 0;
}
