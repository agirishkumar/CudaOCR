#include <fstream>

#include "cudaocr/components/ctc.hpp"

namespace cudaocr {

Result<Charset> loadCtcCharset(const std::string& dictPath) {
    std::ifstream in(dictPath, std::ios::binary);
    if (!in) return Status::Error(ErrorCode::IoError, "cannot read dictionary " + dictPath);
    Charset cs;
    cs.symbols.push_back("");  // CTC blank
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        cs.symbols.push_back(line);
    }
    if (cs.symbols.size() == 1)
        return Status::Error(ErrorCode::IoError, "empty dictionary " + dictPath);
    cs.symbols.push_back(" ");
    cs.blank = 0;
    return cs;
}

}  // namespace cudaocr
