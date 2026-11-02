#include "cudaocr/components/hocr_formatter.hpp"

#include <cmath>
#include <string>

namespace cudaocr {
namespace {

std::string xml(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&quot;";
                break;
            case '\'':
                out += "&apos;";
                break;
            default:
                out += c;
        }
    }
    return out;
}

// "bbox x0 y0 x1 y1; x_wconf N" (bbox omitted when unknown).
std::string title(const BBox& b, float conf) {
    std::string t;
    if (!b.empty())
        t = "bbox " + std::to_string(b.x0) + " " + std::to_string(b.y0) + " " +
            std::to_string(b.x1 + 1) + " " + std::to_string(b.y1 + 1) + "; ";
    return t + "x_wconf " + std::to_string(std::lround(conf * 100.0f));
}

}  // namespace

Result<std::unique_ptr<HocrFormatter>> HocrFormatter::create(const Params& p) {
    Status s = p.requireKnown({}, "hocr");
    if (!s.ok()) return s;
    return std::make_unique<HocrFormatter>();
}

Result<std::string> HocrFormatter::format(const PageResult& page) {
    std::string h =
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
        " <body>\n";
    h += "  <div class=\"ocr_page\" id=\"page_1\" title=\"bbox 0 0 " + std::to_string(page.width) +
         " " + std::to_string(page.height) + "\">\n";
    for (size_t i = 0; i < page.lines.size(); ++i) {
        const LineResult& l = page.lines[i];
        const std::string lineId = "line_1_" + std::to_string(i + 1);
        h += "   <span class=\"ocr_line\" id=\"" + lineId + "\" title=\"" + title(l.box, l.conf) +
             "\">";
        if (l.words.empty()) {
            h += xml(l.text) + "</span>\n";
            continue;
        }
        h += "\n";
        for (size_t k = 0; k < l.words.size(); ++k) {
            const WordResult& w = l.words[k];
            h += "    <span class=\"ocrx_word\" id=\"word_1_" + std::to_string(i + 1) + "_" +
                 std::to_string(k + 1) + "\" title=\"" + title(w.box, w.conf) + "\">" +
                 xml(w.text) + "</span>\n";
        }
        h += "   </span>\n";
    }
    h += "  </div>\n </body>\n</html>\n";
    return h;
}

}  // namespace cudaocr
