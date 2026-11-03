#pragma once

// PDF pages rendered to 8-bit gray images for OCR (scanned PDFs). Backed by
// poppler-cpp when the build has it (CUDAOCR_WITH_PDF); otherwise every call
// fails with a message saying how to enable it. Host-only header.

#include <memory>
#include <string>

#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

class PdfDocument {
  public:
    static bool supported();
    // IoError if the file is missing or unreadable; InvalidArgument if it is not
    // a PDF or is encrypted.
    static Result<PdfDocument> open(const std::string& path);

    int pageCount() const;
    // Renders page `index` (0-based) at `dpi` (0, 2400] to 1-channel gray.
    Result<HostImage> renderPage(int index, double dpi = 300.0) const;

    PdfDocument(PdfDocument&&) noexcept;
    PdfDocument& operator=(PdfDocument&&) noexcept;
    ~PdfDocument();

  private:
    struct Impl;
    explicit PdfDocument(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudaocr
