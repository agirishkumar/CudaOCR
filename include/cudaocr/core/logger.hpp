#pragma once

#include <functional>
#include <string>

namespace cudaocr {

enum class LogLevel { Trace = 0, Debug, Info, Warn, Error };

// Thread-safe logger. Default sink: file (if init'ed) + stderr for >= Warn.
// A custom sink can be injected for tests.
class Logger {
  public:
    using Sink = std::function<void(LogLevel, const std::string& formattedLine)>;

    static void init(const std::string& logFilePath);
    static void close();
    static void setLevel(LogLevel minLevel);
    static LogLevel level();
    static void setSink(Sink sink);   // pass nullptr to restore the default sink
    static void log(LogLevel level, const std::string& message);

    static const char* levelName(LogLevel level);
};

}  // namespace cudaocr
