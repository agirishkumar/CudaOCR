#include <gtest/gtest.h>

#include "cudaocr/components/params.hpp"

namespace cudaocr {
namespace {

TEST(Params, MissingKeyReturnsDefault) {
    const Params p;
    auto v = p.get<int>("blockSize", 31);
    ASSERT_TRUE(v.ok());
    EXPECT_EQ(v.value(), 31);
    EXPECT_EQ(p.get<std::string>("ep", "cuda").value(), "cuda");
    EXPECT_FALSE(p.has("blockSize"));
}

TEST(Params, ParsesTypedValues) {
    Params p;
    p.set("i", "-42").set("f", "0.25").set("d", "1e-3").set("s", "hello world");
    p.set("b1", "true").set("b2", "0").set("b3", "Yes").set("b4", "off");
    EXPECT_EQ(p.get<int>("i", 0).value(), -42);
    EXPECT_FLOAT_EQ(p.get<float>("f", 0).value(), 0.25f);
    EXPECT_DOUBLE_EQ(p.get<double>("d", 0).value(), 1e-3);
    EXPECT_EQ(p.get<std::string>("s", "").value(), "hello world");
    EXPECT_TRUE(p.get<bool>("b1", false).value());
    EXPECT_FALSE(p.get<bool>("b2", true).value());
    EXPECT_TRUE(p.get<bool>("b3", false).value());
    EXPECT_FALSE(p.get<bool>("b4", true).value());
}

TEST(Params, TypedSettersRoundTrip) {
    Params p;
    p.set("k", 5).set("x", 2.5).set("on", true).set("tiny", 1e-9).set("third", 1.0 / 3);
    EXPECT_EQ(p.get<int>("k", 0).value(), 5);
    EXPECT_DOUBLE_EQ(p.get<double>("x", 0).value(), 2.5);
    EXPECT_TRUE(p.get<bool>("on", false).value());
    EXPECT_DOUBLE_EQ(p.get<double>("tiny", 0).value(), 1e-9);  // no precision loss
    EXPECT_EQ(p.get<double>("third", 0).value(), 1.0 / 3);     // exact round trip
}

TEST(Params, BadTypeIsInvalidArgumentNamingKeyAndValue) {
    Params p;
    p.set("blockSize", "abc").set("C", "12x").set("sigma", "").set("flag", "maybe");
    p.set("big", "99999999999");
    for (const char* key : {"blockSize", "C", "big"}) {
        auto v = p.get<int>(key, 0);
        ASSERT_FALSE(v.ok()) << key;
        EXPECT_EQ(v.status().code, ErrorCode::InvalidArgument);
        EXPECT_NE(v.status().message.find(key), std::string::npos) << v.status().message;
    }
    EXPECT_NE(p.get<int>("blockSize", 0).status().message.find("abc"), std::string::npos);
    EXPECT_FALSE(p.get<double>("sigma", 1.0).ok());
    EXPECT_FALSE(p.get<bool>("flag", false).ok());
}

TEST(Params, RequireKnownRejectsTypos) {
    Params p;
    p.set("blockSize", 31).set("blocksize", 15);
    Status s = p.requireKnown({"blockSize", "C"}, "adaptive");
    EXPECT_EQ(s.code, ErrorCode::InvalidArgument);
    EXPECT_NE(s.message.find("blocksize"), std::string::npos) << s.message;
    EXPECT_NE(s.message.find("adaptive"), std::string::npos) << s.message;

    Params ok;
    ok.set("C", 2);
    EXPECT_TRUE(ok.requireKnown({"blockSize", "C"}, "adaptive").ok());
}

}  // namespace
}  // namespace cudaocr
