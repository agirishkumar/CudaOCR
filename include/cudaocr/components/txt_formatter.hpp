#pragma once

#include <memory>

#include "cudaocr/components/formatter.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "txt": recognized line texts in reading order, one per line, each ending in
// '\n'; lines with no recognized text are skipped. No params.
class TxtFormatter final : public Formatter {
  public:
    static Result<std::unique_ptr<TxtFormatter>> create(const Params& p);
    std::string_view name() const override { return "txt"; }
    std::string_view extension() const override { return "txt"; }
    Result<std::string> format(const PageResult& page) override;
};

}  // namespace cudaocr
