#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cudaocr/core/logger.hpp"

namespace cudaocr {
namespace {

class LoggerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Trace);
        Logger::setSink([this](LogLevel level, const std::string& line) {
            std::lock_guard<std::mutex> lock(mutex_);
            captured_.emplace_back(level, line);
        });
    }
    void TearDown() override {
        Logger::setSink(nullptr);
        Logger::setLevel(LogLevel::Info);
    }
    std::mutex mutex_;
    std::vector<std::pair<LogLevel, std::string>> captured_;
};

TEST_F(LoggerTest, FormatsLevelAndMessage) {
    Logger::log(LogLevel::Warn, "hello");
    ASSERT_EQ(captured_.size(), 1u);
    EXPECT_EQ(captured_[0].first, LogLevel::Warn);
    EXPECT_NE(captured_[0].second.find("[WARN] hello"), std::string::npos);
}

TEST_F(LoggerTest, FiltersBelowMinLevel) {
    Logger::setLevel(LogLevel::Warn);
    Logger::log(LogLevel::Info, "dropped");
    Logger::log(LogLevel::Error, "kept");
    ASSERT_EQ(captured_.size(), 1u);
    EXPECT_NE(captured_[0].second.find("kept"), std::string::npos);
}

TEST_F(LoggerTest, ThreadSafeNoTornMessages) {
    constexpr int kThreads = 8, kPerThread = 200;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerThread; ++i)
                Logger::log(LogLevel::Info, "t" + std::to_string(t) + "-m" + std::to_string(i));
        });
    }
    for (auto& th : threads) th.join();
    ASSERT_EQ(captured_.size(), size_t(kThreads) * kPerThread);
    for (const auto& [level, line] : captured_) {
        // every line is a complete, untorn record
        EXPECT_NE(line.find("[INFO] t"), std::string::npos) << line;
        EXPECT_NE(line.find("-m"), std::string::npos) << line;
    }
}

}  // namespace
}  // namespace cudaocr
