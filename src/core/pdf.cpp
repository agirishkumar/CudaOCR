#include "cudaocr/core/pdf.hpp"

#include <filesystem>

#ifdef CUDAOCR_WITH_PDF
#include <poppler-document.h>
#include <poppler-global.h>
#include <poppler-image.h>
#include <poppler-page-renderer.h>
#include <poppler-page.h>

#include <cstring>

#include "cudaocr/core/logger.hpp"
#endif

namespace cudaocr {

#ifdef CUDAOCR_WITH_PDF

namespace {
// Poppler prints parse errors to stderr by default; keep them in our debug log
// so a broken file produces one clean error message.
void popplerToLogger(const std::string& msg, void*) {
    Logger::log(LogLevel::Debug, "poppler: " + msg);
}
}  // namespace

struct PdfDocument::Impl {
    std::unique_ptr<poppler::document> doc;
    std::string path;
};

bool PdfDocument::supported() {
    return true;
}

Result<PdfDocument> PdfDocument::open(const std::string& path) {
    poppler::set_debug_error_function(popplerToLogger, nullptr);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return Status::Error(ErrorCode::IoError, "cannot read PDF " + path);
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    impl->doc.reset(poppler::document::load_from_file(path));
    if (!impl->doc)
        return Status::Error(ErrorCode::InvalidArgument, path + " is not a readable PDF");
    if (impl->doc->is_locked())
        return Status::Error(ErrorCode::InvalidArgument, path + " is encrypted");
    return PdfDocument(std::move(impl));
}

int PdfDocument::pageCount() const {
    return impl_->doc->pages();
}

Result<HostImage> PdfDocument::renderPage(int index, double dpi) const {
    if (index < 0 || index >= pageCount())
        return Status::Error(ErrorCode::InvalidArgument,
                             impl_->path + ": page " + std::to_string(index + 1) +
                                 " out of range (1.." + std::to_string(pageCount()) + ")");
    if (!(dpi > 0 && dpi <= 2400))
        return Status::Error(ErrorCode::InvalidArgument, "PDF dpi must be in (0, 2400]");
    std::unique_ptr<poppler::page> page(impl_->doc->create_page(index));
    if (!page)
        return Status::Error(ErrorCode::IoError,
                             impl_->path + ": cannot load page " + std::to_string(index + 1));
    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);
    renderer.set_image_format(poppler::image::format_gray8);
    const poppler::image img = renderer.render_page(page.get(), dpi, dpi);
    if (!img.is_valid() || img.format() != poppler::image::format_gray8)
        return Status::Error(ErrorCode::IoError,
                             impl_->path + ": cannot render page " + std::to_string(index + 1));
    HostImage out = HostImage::create(img.width(), img.height(), 1);
    for (int y = 0; y < img.height(); ++y)
        std::memcpy(&out.at(0, y), img.const_data() + static_cast<size_t>(y) * img.bytes_per_row(),
                    static_cast<size_t>(img.width()));
    return out;
}

#else  // built without poppler-cpp

struct PdfDocument::Impl {};

bool PdfDocument::supported() {
    return false;
}

Result<PdfDocument> PdfDocument::open(const std::string& path) {
    return Status::Error(ErrorCode::InvalidArgument,
                         "cannot open " + path +
                             ": built without PDF support (install libpoppler-cpp-dev and "
                             "reconfigure with -DCUDAOCR_WITH_PDF=ON)");
}

int PdfDocument::pageCount() const {
    return 0;
}

Result<HostImage> PdfDocument::renderPage(int, double) const {
    return Status::Error(ErrorCode::InvalidArgument, "built without PDF support");
}

#endif

PdfDocument::PdfDocument(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PdfDocument::PdfDocument(PdfDocument&&) noexcept = default;
PdfDocument& PdfDocument::operator=(PdfDocument&&) noexcept = default;
PdfDocument::~PdfDocument() = default;

}  // namespace cudaocr
