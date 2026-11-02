#include <gtest/gtest.h>

#include "cudaocr/eval/metrics.hpp"

namespace cudaocr::eval {
namespace {

TEST(Utf8Decode, DecodesMultiByteAndReplacesInvalid) {
    EXPECT_EQ(utf8Decode("abc"), (std::u32string{U'a', U'b', U'c'}));
    EXPECT_EQ(utf8Decode("caf\xC3\xA9"), (std::u32string{U'c', U'a', U'f', U'é'}));
    EXPECT_EQ(utf8Decode("\xE2\x82\xAC"), std::u32string{U'€'});               // euro, 3 bytes
    EXPECT_EQ(utf8Decode("\xF0\x9F\x98\x80"), std::u32string{U'\U0001F600'});  // 4 bytes
    EXPECT_EQ(utf8Decode("a\xFF"
                         "b"),
              (std::u32string{U'a', U'�', U'b'}));                    // bad lead
    EXPECT_EQ(utf8Decode("\xC3"), std::u32string{U'�'});              // truncated
    EXPECT_EQ(utf8Decode("\xC0\xAF"), (std::u32string{U'�', U'�'}));  // overlong
    EXPECT_TRUE(utf8Decode("").empty());
}

TEST(EditDistance, ClassicCases) {
    EXPECT_EQ(editDistance(std::u32string(U"kitten"), std::u32string(U"sitting")), 3u);
    EXPECT_EQ(editDistance(std::u32string(U""), std::u32string(U"abc")), 3u);
    EXPECT_EQ(editDistance(std::u32string(U"abc"), std::u32string(U"")), 3u);
    EXPECT_EQ(editDistance(std::u32string(U"abc"), std::u32string(U"abc")), 0u);
}

TEST(NormalizeWhitespace, CollapsesRunsAndTrims) {
    EXPECT_EQ(normalizeWhitespace("  a \t b\n\nc  \n"), "a b c");
    EXPECT_EQ(normalizeWhitespace("\n \t"), "");
}

TEST(CharErrors, IdenticalIsZero) {
    const auto e = charErrors("The quick fox", "The quick fox");
    EXPECT_EQ(e.edits, 0u);
    EXPECT_EQ(e.refLength, 13u);
    EXPECT_DOUBLE_EQ(e.rate(), 0.0);
}

TEST(CharErrors, OneSubstitution) {
    const auto e = charErrors("hello", "hallo");
    EXPECT_EQ(e.edits, 1u);
    EXPECT_DOUBLE_EQ(e.rate(), 0.2);
}

TEST(CharErrors, InsertionAndDeletion) {
    EXPECT_EQ(charErrors("hello", "helllo").edits, 1u);      // insertion
    EXPECT_EQ(charErrors("hello", "helo").edits, 1u);        // deletion
    EXPECT_DOUBLE_EQ(charErrors("ab", "abcd").rate(), 1.0);  // 2 insertions / 2
    EXPECT_DOUBLE_EQ(charErrors("a", "xyz").rate(), 3.0);    // CER is not capped at 1
}

TEST(CharErrors, EmptyStrings) {
    EXPECT_EQ(charErrors("", "").edits, 0u);
    EXPECT_DOUBLE_EQ(charErrors("", "").rate(), 0.0);
    const auto missing = charErrors("abc", "");  // no output at all: every char deleted
    EXPECT_EQ(missing.edits, 3u);
    EXPECT_DOUBLE_EQ(missing.rate(), 1.0);
    const auto spurious = charErrors("", "xyz");  // text where none exists
    EXPECT_EQ(spurious.edits, 3u);
    EXPECT_DOUBLE_EQ(spurious.rate(), 1.0);  // defined as 1 when the reference is empty
}

TEST(CharErrors, CountsCodePointsNotBytes) {
    const auto e = charErrors("caf\xC3\xA9", "cafe");  // é (2 bytes) vs e
    EXPECT_EQ(e.refLength, 4u);
    EXPECT_EQ(e.edits, 1u);
    EXPECT_EQ(charErrors("\xE2\x82\xAC"
                         "5",
                         "\xE2\x82\xAC"
                         "5")
                  .edits,
              0u);
}

TEST(CharErrors, WhitespaceDifferencesAreNotErrors) {
    EXPECT_EQ(charErrors("line one\nline two\n", "line one line  two").edits, 0u);
}

TEST(WordErrors, CountsWordEdits) {
    const auto e = wordErrors("the quick brown fox", "the quack brown");
    EXPECT_EQ(e.refLength, 4u);
    EXPECT_EQ(e.edits, 2u);  // substitution + deletion
    EXPECT_DOUBLE_EQ(e.rate(), 0.5);
    EXPECT_EQ(wordErrors("a  b\nc", "a b c").edits, 0u);
    EXPECT_EQ(wordErrors("", "").edits, 0u);
}

TEST(ErrorCounts, AggregatesAsTotalEditsOverTotalLength) {
    ErrorCounts total;
    total += charErrors("aaaa", "aaab");      // 1 / 4
    total += charErrors("bbbbbbbbbbbb", "");  // 12 / 12
    EXPECT_EQ(total.edits, 13u);
    EXPECT_EQ(total.refLength, 16u);
    EXPECT_DOUBLE_EQ(total.rate(), 13.0 / 16.0);
}

}  // namespace
}  // namespace cudaocr::eval
