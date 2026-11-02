#include <gtest/gtest.h>

#include "cudaocr/components/preprocess.hpp"
#include "cudaocr/registry.hpp"

namespace cudaocr {
namespace {

TEST(ComponentRegistry, BuiltinsAreRegisteredByName) {
    const auto reg = ComponentRegistry::withBuiltins();
    const std::vector<std::string> stages = reg.names<ImageStage>();
    for (const char* n :
         {"grayscale", "equalize", "median", "gaussian", "polarity", "adaptive", "otsu"})
        EXPECT_NE(std::find(stages.begin(), stages.end(), n), stages.end()) << n;
    EXPECT_EQ(reg.names<Deskewer>(), std::vector<std::string>{"hough"});
    EXPECT_EQ(reg.names<TextDetector>(), std::vector<std::string>{"projection"});

    auto median = reg.make<ImageStage>({"median", Params().set("k", 5)});
    ASSERT_TRUE(median.ok()) << median.status().message;
    EXPECT_EQ(median.value()->name(), "median");
}

TEST(ComponentRegistry, UnknownNameIsInvalidArgumentListingChoices) {
    const auto reg = ComponentRegistry::withBuiltins();
    auto r = reg.make<ImageStage>({"medain", {}});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.status().code, ErrorCode::InvalidArgument);
    EXPECT_NE(r.status().message.find("'medain'"), std::string::npos) << r.status().message;
    EXPECT_NE(r.status().message.find("median"), std::string::npos) << r.status().message;
}

TEST(ComponentRegistry, FactoryErrorsArePrefixedWithComponentName) {
    const auto reg = ComponentRegistry::withBuiltins();
    auto r = reg.make<ImageStage>({"median", Params().set("k", "big")});
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.status().message.find("median"), std::string::npos) << r.status().message;
    EXPECT_NE(r.status().message.find("big"), std::string::npos) << r.status().message;
}

TEST(ComponentRegistry, CustomComponentsCanBeAddedButNotDuplicated) {
    auto reg = ComponentRegistry::withBuiltins();
    struct Identity : ImageStage {
        std::string_view name() const override { return "identity"; }
        Status run(const DeviceImage&, DeviceImage&, StageContext&) override {
            return Status::Good();
        }
    };
    ComponentRegistry::Factory<ImageStage> f =
        [](const Params&) -> Result<std::unique_ptr<ImageStage>> {
        return std::unique_ptr<ImageStage>(std::make_unique<Identity>());
    };
    EXPECT_TRUE(reg.add<ImageStage>("identity", f).ok());
    EXPECT_TRUE(reg.make<ImageStage>({"identity", {}}).ok());
    Status dup = reg.add<ImageStage>("median", f);
    EXPECT_EQ(dup.code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace cudaocr
