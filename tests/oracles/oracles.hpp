#pragma once

// CPU reference implementations ("oracles") for every GPU kernel, plus
// deterministic fixture generators. Simple, obviously-correct, unoptimized.

#include <cstdint>
#include <vector>

#include "cudaocr/core/host_image.hpp"

namespace cudaocr::oracle {

HostImage grayscale(const HostImage& rgb);
HostImage boxMean(const HostImage& in, int blockSize);
HostImage median(const HostImage& in, int k);
HostImage gaussian(const HostImage& in, int ksize, double sigma);
HostImage adaptiveThreshold(const HostImage& in, int blockSize, int C);
std::vector<unsigned int> histogram256(const HostImage& in);
int otsuThreshold(const std::vector<unsigned int>& hist);
HostImage applyThreshold(const HostImage& in, int t);
HostImage equalize(const HostImage& in);
HostImage invert(const HostImage& in);
HostImage rotate(const HostImage& in, float angleDeg, std::uint8_t border);
std::vector<int> rowProfile(const HostImage& bin);
std::vector<int> colProfile(const HostImage& bin, int y0, int y1);

// Deterministic pseudo-random image (LCG; no global RNG state).
HostImage randomImage(int w, int h, int ch, std::uint32_t seed);

// Synthetic "document": white page with 3 lines of word-like black boxes,
// each word split into character boxes. Returns the expected counts.
struct SyntheticPage {
    HostImage image;  // 1-channel, text black (0) on white (255)
    int lineCount;
    std::vector<int> wordsPerLine;
    std::vector<int> charsPerLine;
};
SyntheticPage makeSyntheticPage(int w = 640, int h = 240);

}  // namespace cudaocr::oracle
