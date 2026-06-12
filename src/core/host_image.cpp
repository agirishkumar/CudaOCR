#include "cudaocr/core/host_image.hpp"

#include <algorithm>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace cudaocr {

Result<HostImage> loadImage(const std::string& path) {
    int w = 0, h = 0, ch = 0;
    if (!stbi_info(path.c_str(), &w, &h, &ch)) {
        return Status::Error(ErrorCode::IoError,
                             "cannot read '" + path + "': " + stbi_failure_reason());
    }
    const int want = (ch >= 3) ? 3 : 1;  // fold alpha away
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &ch, want);
    if (!pixels) {
        return Status::Error(ErrorCode::IoError,
                             "decode failed for '" + path + "': " + stbi_failure_reason());
    }
    HostImage img = HostImage::create(w, h, want);
    std::memcpy(img.data.data(), pixels, img.data.size());
    stbi_image_free(pixels);
    return img;
}

Status saveImage(const std::string& path, const HostImage& img) {
    if (img.empty()) return Status::Error(ErrorCode::InvalidArgument, "saveImage: empty image");
    auto endsWith = [&](const char* suf) {
        const size_t n = std::strlen(suf);
        return path.size() >= n &&
               std::equal(path.end() - n, path.end(), suf, [](char a, char b) {
                   return std::tolower(a) == std::tolower(b);
               });
    };
    int ok = 0;
    if (endsWith(".png")) {
        ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels, img.data.data(),
                            static_cast<int>(img.rowBytes()));
    } else if (endsWith(".jpg") || endsWith(".jpeg")) {
        ok = stbi_write_jpg(path.c_str(), img.width, img.height, img.channels, img.data.data(), 95);
    } else {
        return Status::Error(ErrorCode::InvalidArgument,
                             "saveImage: unsupported extension in '" + path + "'");
    }
    if (!ok) return Status::Error(ErrorCode::IoError, "saveImage: write failed for '" + path + "'");
    return Status::Good();
}

}  // namespace cudaocr
