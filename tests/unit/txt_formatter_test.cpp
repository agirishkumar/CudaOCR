#include <gtest/gtest.h>

#include "cudaocr/components/txt_formatter.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

TEST(TxtFormatter, OneLinePerRecognizedLineSkippingEmpty) {
    PageResult page;
    page.lines.resize(4);
    page.lines[0].text = "first line";
    page.lines[1].text = "";  // detected but nothing recognized
    page.lines[2].text = "second";
    page.lines[3].text = "  third  ";  // kept verbatim
    auto f = TxtFormatter::create({});
    ASSERT_TRUE(f.ok());
    EXPECT_EQ(f.value()->format(page).value(), "first line\nsecond\n  third  \n");
    EXPECT_EQ(f.value()->extension(), "txt");
}

TEST(TxtFormatter, EmptyPageIsEmptyString) {
    auto f = TxtFormatter::create({});
    EXPECT_EQ(f.value()->format(PageResult{}).value(), "");
}

TEST(TxtFormatter, RegisteredAsTxt) {
    auto f = ComponentRegistry::withBuiltins().make<Formatter>({"txt", {}});
    ASSERT_TRUE(f.ok()) << f.status().message;
    EXPECT_EQ(f.value()->name(), "txt");
}

}  // namespace
}  // namespace cudaocr
