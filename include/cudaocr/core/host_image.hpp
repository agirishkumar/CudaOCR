#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Interleaved 8-bit image on the host (1 = gray, 3 = RGB).
struct HostImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::uint8_t> data;

    static HostImage create(int w, int h, int ch, std::uint8_t fill = 0) {
        HostImage img;
        img.width = w;
        img.height = h;
        img.channels = ch;
        img.data.assign(static_cast<size_t>(w) * h * ch, fill);
        return img;
    }
    [[nodiscard]] bool empty() const { return data.empty(); }
    [[nodiscard]] size_t rowBytes() const { return static_cast<size_t>(width) * channels; }
    std::uint8_t& at(int x, int y, int c = 0) {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }
    [[nodiscard]] std::uint8_t at(int x, int y, int c = 0) const {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }
};

// PNG/JPEG/BMP/TIFF-subset via stb. 2/4-channel inputs are folded to 1/3 (alpha dropped).
Result<HostImage> loadImage(const std::string& path);
// Format chosen by extension: .png or .jpg/.jpeg.
Status saveImage(const std::string& path, const HostImage& img);

}  // namespace cudaocr
