#include "oracles/oracles.hpp"

#include <algorithm>
#include <cmath>

#include "cudaocr/stages/filters.cuh"  // gaussianFixedWeights (host-only inline)

namespace cudaocr::oracle {
namespace {
int clampi(int v, int lo, int hi) { return std::clamp(v, lo, hi); }
}  // namespace

HostImage grayscale(const HostImage& rgb) {
    HostImage out = HostImage::create(rgb.width, rgb.height, 1);
    for (int y = 0; y < rgb.height; ++y)
        for (int x = 0; x < rgb.width; ++x) {
            const int r = rgb.at(x, y, 0), g = rgb.at(x, y, 1), b = rgb.at(x, y, 2);
            out.at(x, y) = static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
        }
    return out;
}

HostImage boxMean(const HostImage& in, int blockSize) {
    const int r = blockSize / 2;
    const std::uint32_t n = static_cast<std::uint32_t>(blockSize) * blockSize;
    // Two-pass to mirror the GPU implementation exactly (same order, same rounding).
    std::vector<std::uint32_t> tmp(static_cast<size_t>(in.width) * in.height);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            std::uint32_t s = 0;
            for (int dx = -r; dx <= r; ++dx) s += in.at(clampi(x + dx, 0, in.width - 1), y);
            tmp[static_cast<size_t>(y) * in.width + x] = s;
        }
    HostImage out = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            std::uint32_t s = 0;
            for (int dy = -r; dy <= r; ++dy)
                s += tmp[static_cast<size_t>(clampi(y + dy, 0, in.height - 1)) * in.width + x];
            out.at(x, y) = static_cast<std::uint8_t>((s + n / 2) / n);
        }
    return out;
}

HostImage median(const HostImage& in, int k) {
    const int r = k / 2;
    HostImage out = HostImage::create(in.width, in.height, 1);
    std::vector<std::uint8_t> v;
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            v.clear();
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    v.push_back(
                        in.at(clampi(x + dx, 0, in.width - 1), clampi(y + dy, 0, in.height - 1)));
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            out.at(x, y) = v[v.size() / 2];
        }
    return out;
}

HostImage gaussian(const HostImage& in, int ksize, double sigma) {
    const std::vector<int> w = gaussianFixedWeights(ksize, sigma);
    const int r = ksize / 2;
    HostImage tmp = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            std::uint32_t acc = 0;
            for (int d = -r; d <= r; ++d)
                acc += static_cast<std::uint32_t>(w[d + r]) * in.at(clampi(x + d, 0, in.width - 1), y);
            tmp.at(x, y) = static_cast<std::uint8_t>((acc + 32768u) >> 16);
        }
    HostImage out = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            std::uint32_t acc = 0;
            for (int d = -r; d <= r; ++d)
                acc += static_cast<std::uint32_t>(w[d + r]) *
                       tmp.at(x, clampi(y + d, 0, in.height - 1));
            out.at(x, y) = static_cast<std::uint8_t>((acc + 32768u) >> 16);
        }
    return out;
}

HostImage adaptiveThreshold(const HostImage& in, int blockSize, int C) {
    HostImage mean = boxMean(in, blockSize);
    HostImage out = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x)
            out.at(x, y) = (in.at(x, y) > mean.at(x, y) - C) ? 255 : 0;
    return out;
}

std::vector<unsigned int> histogram256(const HostImage& in) {
    std::vector<unsigned int> hist(256, 0);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) ++hist[in.at(x, y)];
    return hist;
}

int otsuThreshold(const std::vector<unsigned int>& hist) {
    double total = 0, sumAll = 0;
    for (int i = 0; i < 256; ++i) {
        total += hist[i];
        sumAll += static_cast<double>(i) * hist[i];
    }
    double sumB = 0, wB = 0, best = -1.0;
    int bestT = 0;
    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        const double wF = total - wB;
        if (wF == 0) break;
        sumB += static_cast<double>(t) * hist[t];
        const double mB = sumB / wB;
        const double mF = (sumAll - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) {
            best = between;
            bestT = t;
        }
    }
    return bestT;
}

HostImage applyThreshold(const HostImage& in, int t) {
    HostImage out = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) out.at(x, y) = (in.at(x, y) > t) ? 255 : 0;
    return out;
}

HostImage equalize(const HostImage& in) {
    const auto hist = histogram256(in);
    const std::uint64_t total = static_cast<std::uint64_t>(in.width) * in.height;
    std::uint64_t cdf = 0, cdfMin = 0;
    std::uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        if (cdfMin == 0 && cdf != 0) cdfMin = cdf;
        if (total == cdfMin) {
            lut[i] = static_cast<std::uint8_t>(i);
        } else {
            const double v = 255.0 * (static_cast<double>(cdf) - static_cast<double>(cdfMin)) /
                             (static_cast<double>(total) - static_cast<double>(cdfMin));
            lut[i] = static_cast<std::uint8_t>(std::lround(std::max(0.0, v)));
        }
    }
    HostImage out = HostImage::create(in.width, in.height, 1);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) out.at(x, y) = lut[in.at(x, y)];
    return out;
}

HostImage invert(const HostImage& in) {
    HostImage out = in;
    for (auto& p : out.data) p = static_cast<std::uint8_t>(255 - p);
    return out;
}

HostImage rotate(const HostImage& in, float angleDeg, std::uint8_t border) {
    const double a = angleDeg * M_PI / 180.0;
    const double cosA = std::cos(a), sinA = std::sin(a);
    const double cx = (in.width - 1) / 2.0, cy = (in.height - 1) / 2.0;
    HostImage out = HostImage::create(in.width, in.height, 1);
    auto sample = [&](int xx, int yy) -> double {
        if (xx < 0 || xx >= in.width || yy < 0 || yy >= in.height) return border;
        return in.at(xx, yy);
    };
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            const double dx = x - cx, dy = y - cy;
            const double sx = cosA * dx + sinA * dy + cx;
            const double sy = -sinA * dx + cosA * dy + cy;
            std::uint8_t result = border;
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            if (x0 >= -1 && x0 < in.width && y0 >= -1 && y0 < in.height) {
                const double fx = sx - x0, fy = sy - y0;
                const double v = (1 - fx) * (1 - fy) * sample(x0, y0) +
                                 fx * (1 - fy) * sample(x0 + 1, y0) +
                                 (1 - fx) * fy * sample(x0, y0 + 1) +
                                 fx * fy * sample(x0 + 1, y0 + 1);
                result = static_cast<std::uint8_t>(
                    std::min(255.0, std::max(0.0, std::nearbyint(v))));
            }
            out.at(x, y) = result;
        }
    return out;
}

std::vector<int> rowProfile(const HostImage& bin) {
    std::vector<int> rows(bin.height, 0);
    for (int y = 0; y < bin.height; ++y)
        for (int x = 0; x < bin.width; ++x)
            if (bin.at(x, y) < 128) ++rows[y];
    return rows;
}

std::vector<int> colProfile(const HostImage& bin, int y0, int y1) {
    std::vector<int> cols(bin.width, 0);
    for (int y = y0; y <= y1; ++y)
        for (int x = 0; x < bin.width; ++x)
            if (bin.at(x, y) < 128) ++cols[x];
    return cols;
}

HostImage randomImage(int w, int h, int ch, std::uint32_t seed) {
    HostImage img = HostImage::create(w, h, ch);
    std::uint32_t state = seed * 2654435761u + 1u;
    for (auto& p : img.data) {
        state = state * 1664525u + 1013904223u;
        p = static_cast<std::uint8_t>(state >> 24);
    }
    return img;
}

SyntheticPage makeSyntheticPage(int w, int h) {
    SyntheticPage page;
    page.image = HostImage::create(w, h, 1, 255);
    auto fill = [&](int x0, int y0, int x1, int y1) {
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) page.image.at(x, y) = 0;
    };
    // 3 lines, 18px tall, 60px apart; words = groups of 12px chars with 3px gaps,
    // separated by 24px gaps (word gap threshold = lineHeight/3 = 6 < 24; char gap 3 in [1,6)).
    const int lineHeight = 18;
    const int charW = 12, charGap = 3, wordGap = 24;
    const std::vector<std::vector<int>> charsPerWord = {{3, 5, 4}, {2, 6}, {4, 3, 2, 5}};
    page.lineCount = static_cast<int>(charsPerWord.size());
    for (int li = 0; li < page.lineCount; ++li) {
        const int y0 = 30 + li * 60;
        int x = 40;
        page.wordsPerLine.push_back(static_cast<int>(charsPerWord[li].size()));
        int chars = 0;
        for (int nChars : charsPerWord[li]) {
            for (int c = 0; c < nChars; ++c) {
                fill(x, y0, x + charW - 1, y0 + lineHeight - 1);
                x += charW + charGap;
                ++chars;
            }
            x += wordGap - charGap;
        }
        page.charsPerLine.push_back(chars);
    }
    return page;
}

}  // namespace cudaocr::oracle
