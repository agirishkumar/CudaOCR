#pragma once

#include <memory>

#include "cudaocr/components/formatter.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "json": schema "cudaocr.page/1" (docs/output-formats.md). Deterministic layout
// (one word per line) so outputs diff cleanly. No params.
class JsonFormatter final : public Formatter {
  public:
    static Result<std::unique_ptr<JsonFormatter>> create(const Params& p);
    std::string_view name() const override { return "json"; }
    std::string_view extension() const override { return "json"; }
    Result<std::string> format(const PageResult& page) override;
};

}  // namespace cudaocr
