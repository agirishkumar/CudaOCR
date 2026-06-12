#include <gtest/gtest.h>

#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {
namespace {

TEST(Status, OkByDefault) {
    Status s;
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(Status::Good().ok());
}

TEST(Status, CarriesErrorInfo) {
    Status s = Status::Error(ErrorCode::IoError, "boom");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, ErrorCode::IoError);
    EXPECT_EQ(s.message, "boom");
}

TEST(Result, HoldsValueOrStatus) {
    Result<int> good(42);
    ASSERT_TRUE(good.ok());
    EXPECT_EQ(good.value(), 42);

    Result<int> bad(Status::Error(ErrorCode::InvalidArgument, "nope"));
    ASSERT_FALSE(bad.ok());
    EXPECT_EQ(bad.status().code, ErrorCode::InvalidArgument);
}

Status failingCudaCall() {
    CUDAOCR_CHECK(cudaSetDevice(987654));  // no such device
    return Status::Good();
}

TEST(CudaCheck, ConvertsErrorToStatusWithLocation) {
    Status s = failingCudaCall();
    ASSERT_FALSE(s.ok());
    EXPECT_EQ(s.code, ErrorCode::CudaError);
    EXPECT_NE(s.message.find("cudaSetDevice"), std::string::npos);
    EXPECT_NE(s.message.find("status_test.cpp"), std::string::npos);
    // The sticky per-thread error must not poison later calls.
    cudaGetLastError();
}

TEST(CudaCheck, PassesThroughSuccess) {
    auto fn = []() -> Status {
        CUDAOCR_CHECK(cudaSuccess);
        return Status::Good();
    };
    EXPECT_TRUE(fn().ok());
}

}  // namespace
}  // namespace cudaocr
