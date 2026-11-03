#pragma once

// Data exchanged between a CTC line recognizer and a decoder. Host-only.

#include <string>
#include <vector>

#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Per-timestep class probabilities for one line, row-major [T x C]. Recognizers
// must emit probabilities (apply softmax if the model outputs raw logits).
class CtcProbs {
  public:
    CtcProbs() = default;
    CtcProbs(int timesteps, int classes)
        : t_(timesteps), c_(classes), data_(static_cast<size_t>(timesteps) * classes, 0.0f) {}

    int timesteps() const { return t_; }
    int classes() const { return c_; }
    float& at(int t, int c) { return data_[static_cast<size_t>(t) * c_ + c]; }
    float at(int t, int c) const { return data_[static_cast<size_t>(t) * c_ + c]; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    // Page geometry of the timesteps, filled by recognizers that know it:
    // timestep t covers page columns [x0 + t * stepWidth, x0 + (t + 1) * stepWidth).
    // stepWidth == 0 means unknown (no word boxes can be derived).
    float x0 = 0.0f;
    float stepWidth = 0.0f;

  private:
    int t_ = 0, c_ = 0;
    std::vector<float> data_;
};

// Class index -> UTF-8 symbol. `blank` is the CTC blank class.
struct Charset {
    std::vector<std::string> symbols;
    int blank = 0;
};

// PaddleOCR-style CTC charset: blank at 0, then one symbol per dictionary line
// (CRLF tolerated), then " " appended last (use_space_char). IoError if unreadable.
Result<Charset> loadCtcCharset(const std::string& dictPath);

}  // namespace cudaocr
