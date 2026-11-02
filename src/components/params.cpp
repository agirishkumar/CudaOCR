#include "cudaocr/components/params.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace cudaocr {
namespace {

Status badValue(const std::string& key, const std::string& raw, const char* type) {
    return Status::Error(ErrorCode::InvalidArgument,
                         "parameter '" + key + "': '" + raw + "' is not a valid " + type);
}

template <typename T>
Result<T> parseNumber(const std::string& key, const std::string& raw, const char* type) {
    T v{};
    const char* end = raw.data() + raw.size();
    auto [ptr, ec] = std::from_chars(raw.data(), end, v);
    if (raw.empty() || ec != std::errc() || ptr != end) return badValue(key, raw, type);
    return v;
}

}  // namespace

Params& Params::set(const std::string& key, std::string value) {
    values_[key] = std::move(value);
    return *this;
}

template <typename T>
Result<T> Params::get(const std::string& key, T fallback) const {
    auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string& raw = it->second;
    if constexpr (std::is_same_v<T, std::string>) {
        return raw;
    } else if constexpr (std::is_same_v<T, bool>) {
        std::string s = raw;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
        if (s == "false" || s == "0" || s == "no" || s == "off") return false;
        return badValue(key, raw, "bool");
    } else if constexpr (std::is_same_v<T, int>) {
        return parseNumber<int>(key, raw, "int");
    } else {
        return parseNumber<T>(key, raw, "number");
    }
}

template Result<int> Params::get<int>(const std::string&, int) const;
template Result<float> Params::get<float>(const std::string&, float) const;
template Result<double> Params::get<double>(const std::string&, double) const;
template Result<bool> Params::get<bool>(const std::string&, bool) const;
template Result<std::string> Params::get<std::string>(const std::string&, std::string) const;

Status Params::requireKnown(std::initializer_list<std::string_view> known,
                            std::string_view component) const {
    std::string unknown;
    for (const auto& [key, value] : values_) {
        if (std::find(known.begin(), known.end(), key) != known.end()) continue;
        unknown += (unknown.empty() ? "'" : ", '") + key + "'";
    }
    if (unknown.empty()) return Status::Good();
    return Status::Error(ErrorCode::InvalidArgument,
                         std::string(component) + ": unknown parameter(s) " + unknown);
}

}  // namespace cudaocr
