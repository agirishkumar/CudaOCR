#pragma once

#include <memory>

#include "cudaocr/components/formatter.hpp"
#include "cudaocr/components/params.hpp"

namespace cudaocr {

// "hocr": hOCR 1.2 XHTML (docs/output-formats.md): ocr_page > ocr_line > ocrx_word,
// `bbox x0 y0 x1 y1` with exclusive right/bottom (Tesseract's convention; our
// boxes are inclusive internally) and `x_wconf` = round(conf * 100). Words with
// unknown geometry omit bbox; a recognized line without words holds its text.
class HocrFormatter final : public Formatter {
  public:
    static Result<std::unique_ptr<HocrFormatter>> create(const Params& p);
    std::string_view name() const override { return "hocr"; }
    std::string_view extension() const override { return "hocr"; }
    Result<std::string> format(const PageResult& page) override;
};

}  // namespace cudaocr
