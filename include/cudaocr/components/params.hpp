#pragma once

// String-keyed component parameters. Values are stored as text (as they arrive
// from the CLI or a config file) and parsed on read, so every component reports
// bad values the same way: InvalidArgument naming the key and the raw value.

#include <charconv>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>

#include "cudaocr/core/status.hpp"

namespace cudaocr {

class Params {
  public:
    Params& set(const std::string& key, std::string value);
    Params& set(const std::string& key, const char* value) { return set(key, std::string(value)); }
    Params& set(const std::string& key, bool value) { return set(key, value ? "true" : "false"); }
    template <typename T>
        requires std::is_arithmetic_v<T>
    Params& set(const std::string& key, T value) {
        char buf[64];  // to_chars: shortest round-trip form (to_string drops precision)
        auto res = std::to_chars(buf, buf + sizeof buf, value);
        return set(key, std::string(buf, res.ptr));
    }

    bool has(const std::string& key) const { return values_.count(key) != 0; }
    const std::map<std::string, std::string>& raw() const { return values_; }

    // Supported T: int, float, double, bool, std::string. Missing key -> fallback.
    // bool accepts true/false, 1/0, yes/no, on/off (case-insensitive).
    template <typename T>
    Result<T> get(const std::string& key, T fallback) const;

    // InvalidArgument listing every key not in `known` (catches typos such as
    // "blocksize" for "blockSize"). `component` names the owner in the message.
    Status requireKnown(std::initializer_list<std::string_view> known,
                        std::string_view component) const;

  private:
    std::map<std::string, std::string> values_;
};

extern template Result<int> Params::get<int>(const std::string&, int) const;
extern template Result<float> Params::get<float>(const std::string&, float) const;
extern template Result<double> Params::get<double>(const std::string&, double) const;
extern template Result<bool> Params::get<bool>(const std::string&, bool) const;
extern template Result<std::string> Params::get<std::string>(const std::string&, std::string) const;

}  // namespace cudaocr
