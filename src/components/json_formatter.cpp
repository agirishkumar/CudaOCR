#include "cudaocr/components/json_formatter.hpp"

#include <charconv>
#include <cstdio>

namespace cudaocr {
namespace {

// Shortest round-trip representation (0.9f -> "0.9", 127.0 -> "127").
template <typename T>
std::string num(T v) {
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, res.ptr);
}

std::string str(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\r':
                out += "\\r";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);  // UTF-8 passes through
                }
        }
    }
    return out + "\"";
}

std::string bbox(const BBox& b) {
    if (b.empty()) return "null";
    return "[" + num(b.x0) + ", " + num(b.y0) + ", " + num(b.x1) + ", " + num(b.y1) + "]";
}

}  // namespace

Result<std::unique_ptr<JsonFormatter>> JsonFormatter::create(const Params& p) {
    Status s = p.requireKnown({}, "json");
    if (!s.ok()) return s;
    return std::make_unique<JsonFormatter>();
}

Result<std::string> JsonFormatter::format(const PageResult& page) {
    std::string text;
    for (const auto& l : page.lines)
        if (!l.text.empty()) text += l.text + "\n";

    std::string j = "{\n";
    j += "  \"schema\": \"cudaocr.page/1\",\n";
    j += "  \"width\": " + num(page.width) + ",\n";
    j += "  \"height\": " + num(page.height) + ",\n";
    j += "  \"skewDeg\": " + num(page.skewDeg) + ",\n";
    j += std::string("  \"inverted\": ") + (page.inverted ? "true" : "false") + ",\n";
    j += "  \"text\": " + str(text) + ",\n";
    j += "  \"metrics\": {";
    for (auto it = page.metrics.begin(); it != page.metrics.end(); ++it)
        j += (it == page.metrics.begin() ? "" : ", ") + str(it->first) + ": " + num(it->second);
    j += "},\n  \"timingsMs\": {";
    for (size_t i = 0; i < page.timings.size(); ++i)
        j += (i ? ", " : "") + str(page.timings[i].name) + ": " + num(page.timings[i].ms);
    j += "},\n  \"lines\": [";
    for (size_t i = 0; i < page.lines.size(); ++i) {
        const LineResult& l = page.lines[i];
        j += (i ? ",\n" : "\n");
        j += "    {\"bbox\": " + bbox(l.box) + ", \"conf\": " + num(l.conf) +
             ", \"text\": " + str(l.text) + ", \"words\": [";
        for (size_t k = 0; k < l.words.size(); ++k) {
            const WordResult& w = l.words[k];
            j += (k ? ",\n" : "\n");
            j += "      {\"bbox\": " + bbox(w.box) + ", \"conf\": " + num(w.conf) +
                 ", \"text\": " + str(w.text);
            if (!w.glyphs.empty()) {
                j += ", \"glyphs\": [";
                for (size_t g = 0; g < w.glyphs.size(); ++g)
                    j += (g ? ", " : "") + bbox(w.glyphs[g]);
                j += "]";
            }
            j += "}";
        }
        j += l.words.empty() ? "]}" : "\n    ]}";
    }
    j += page.lines.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return j;
}

}  // namespace cudaocr
