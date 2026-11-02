#pragma once

#include <string>
#include <string_view>

#include "cudaocr/core/status.hpp"
#include "cudaocr/types.hpp"

namespace cudaocr {

// PageResult -> output document (txt, json, hOCR).
class Formatter {
  public:
    virtual ~Formatter() = default;
    virtual std::string_view name() const = 0;
    virtual std::string_view extension() const = 0;  // without the dot
    virtual Result<std::string> format(const PageResult& page) = 0;
};

}  // namespace cudaocr
