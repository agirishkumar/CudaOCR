#include "cudaocr/components/txt_formatter.hpp"

namespace cudaocr {

Result<std::unique_ptr<TxtFormatter>> TxtFormatter::create(const Params& p) {
    Status s = p.requireKnown({}, "txt");
    if (!s.ok()) return s;
    return std::make_unique<TxtFormatter>();
}

Result<std::string> TxtFormatter::format(const PageResult& page) {
    std::string out;
    for (const auto& line : page.lines) {
        if (line.text.empty()) continue;
        out += line.text;
        out += '\n';
    }
    return out;
}

}  // namespace cudaocr
