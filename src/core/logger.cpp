#include "cudaocr/core/logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <mutex>

namespace cudaocr {
namespace {

std::mutex g_mutex;
std::ofstream g_file;
LogLevel g_minLevel = LogLevel::Info;
Logger::Sink g_sink;  // empty => default sink

std::string timestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

}  // namespace

const char* Logger::levelName(LogLevel level) {
    switch (level) {
        case LogLevel::Trace:
            return "TRACE";
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
    }
    return "?";
}

void Logger::init(const std::string& logFilePath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_file.open(logFilePath, std::ios::app);
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file.is_open()) g_file.close();
}

void Logger::setLevel(LogLevel minLevel) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_minLevel = minLevel;
}

LogLevel Logger::level() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_minLevel;
}

void Logger::setSink(Sink sink) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sink = std::move(sink);
}

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (level < g_minLevel) return;
    const std::string line = timestamp() + " [" + levelName(level) + "] " + message;
    if (g_sink) {
        g_sink(level, line);
        return;
    }
    if (g_file.is_open()) g_file << line << '\n' << std::flush;
    if (level >= LogLevel::Warn) std::fprintf(stderr, "%s\n", line.c_str());
}

}  // namespace cudaocr
