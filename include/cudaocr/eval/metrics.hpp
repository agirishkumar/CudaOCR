#pragma once

// OCR accuracy metrics (host-only). CER/WER are Levenshtein edits divided by the
// reference length, computed after whitespace normalization so line-break and
// spacing differences between a recognizer's output and the ground truth are not
// counted. Characters are Unicode code points, not bytes.

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cudaocr::eval {

// Invalid or truncated sequences (incl. overlong forms and surrogates) decode to
// one U+FFFD per offending byte.
std::u32string utf8Decode(std::string_view s);

// Trims and collapses every whitespace run (space, tab, CR, LF, VT, FF) to one space.
std::string normalizeWhitespace(std::string_view s);

// Unit-cost insert/delete/substitute distance; O(|a|*|b|) time, O(|b|) memory.
// `same(a_i, b_j)` decides whether a substitution is free (default: ==).
template <typename Seq, typename Eq>
size_t editDistance(const Seq& a, const Seq& b, Eq same) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t sub = prev[j - 1] + (same(a[i - 1], b[j - 1]) ? 0 : 1);
            cur[j] = std::min({sub, prev[j] + 1, cur[j - 1] + 1});
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

template <typename Seq>
size_t editDistance(const Seq& a, const Seq& b) {
    return editDistance(a, b, [](const auto& x, const auto& y) { return x == y; });
}

struct ErrorCounts {
    size_t edits = 0;
    size_t refLength = 0;

    // edits / refLength; may exceed 1 (many insertions). Empty reference:
    // 0 if the hypothesis is empty too, else 1.
    double rate() const {
        if (refLength == 0) return edits == 0 ? 0.0 : 1.0;
        return static_cast<double>(edits) / static_cast<double>(refLength);
    }
    // Aggregation = total edits / total reference length (micro-average).
    ErrorCounts& operator+=(const ErrorCounts& o) {
        edits += o.edits;
        refLength += o.refLength;
        return *this;
    }
};

// `refWildcard` (0 = none): a reference code point that matches any single
// hypothesis character for free (UNLV ground truth writes undeterminable
// characters as '~'); deleting it still costs 1. A wildcard inside a reference
// word makes that position match anything in an equal-length hypothesis word.
ErrorCounts charErrors(std::string_view reference, std::string_view hypothesis,
                       char32_t refWildcard = 0);
ErrorCounts wordErrors(std::string_view reference, std::string_view hypothesis,
                       char32_t refWildcard = 0);

}  // namespace cudaocr::eval
