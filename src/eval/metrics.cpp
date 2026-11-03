#include "cudaocr/eval/metrics.hpp"

namespace cudaocr::eval {
namespace {

bool isSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::vector<std::string_view> splitWords(std::string_view normalized) {
    std::vector<std::string_view> words;
    size_t start = 0;
    while (start < normalized.size()) {
        size_t end = normalized.find(' ', start);
        if (end == std::string_view::npos) end = normalized.size();
        words.push_back(normalized.substr(start, end - start));
        start = end + 1;
    }
    return words;
}

}  // namespace

std::u32string utf8Decode(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        int len = 0;
        char32_t cp = 0, min = 0;
        if (b0 < 0x80) {
            out.push_back(b0);
            ++i;
            continue;
        } else if ((b0 & 0xE0) == 0xC0) {
            len = 2, cp = b0 & 0x1F, min = 0x80;
        } else if ((b0 & 0xF0) == 0xE0) {
            len = 3, cp = b0 & 0x0F, min = 0x800;
        } else if ((b0 & 0xF8) == 0xF0) {
            len = 4, cp = b0 & 0x07, min = 0x10000;
        }
        bool ok = len > 0 && i + len <= s.size();
        for (int k = 1; ok && k < len; ++k) {
            const auto b = static_cast<unsigned char>(s[i + k]);
            ok = (b & 0xC0) == 0x80;
            cp = (cp << 6) | (b & 0x3F);
        }
        ok = ok && cp >= min && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
        if (ok) {
            out.push_back(cp);
            i += len;
        } else {
            out.push_back(U'�');  // resynchronize on the next byte
            ++i;
        }
    }
    return out;
}

std::string normalizeWhitespace(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    bool pendingSpace = false;
    for (char c : s) {
        if (isSpace(static_cast<unsigned char>(c))) {
            pendingSpace = !out.empty();
            continue;
        }
        if (pendingSpace) out.push_back(' ');
        pendingSpace = false;
        out.push_back(c);
    }
    return out;
}

ErrorCounts charErrors(std::string_view reference, std::string_view hypothesis,
                       char32_t refWildcard) {
    const std::u32string ref = utf8Decode(normalizeWhitespace(reference));
    const std::u32string hyp = utf8Decode(normalizeWhitespace(hypothesis));
    auto same = [&](char32_t r, char32_t h) { return r == h || (refWildcard && r == refWildcard); };
    return {editDistance(ref, hyp, same), ref.size()};
}

ErrorCounts wordErrors(std::string_view reference, std::string_view hypothesis,
                       char32_t refWildcard) {
    const std::string refNorm = normalizeWhitespace(reference);
    const std::string hypNorm = normalizeWhitespace(hypothesis);
    std::vector<std::u32string> ref, hyp;
    for (auto w : splitWords(refNorm)) ref.push_back(utf8Decode(w));
    for (auto w : splitWords(hypNorm)) hyp.push_back(utf8Decode(w));
    auto same = [&](const std::u32string& r, const std::u32string& h) {
        if (r.size() != h.size()) return false;
        for (size_t i = 0; i < r.size(); ++i)
            if (r[i] != h[i] && !(refWildcard && r[i] == refWildcard)) return false;
        return true;
    };
    return {editDistance(ref, hyp, same), ref.size()};
}

}  // namespace cudaocr::eval
