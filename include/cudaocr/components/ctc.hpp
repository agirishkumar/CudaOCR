#pragma once

// Data exchanged between a CTC line recognizer and a decoder. Host-only.

#include <string>
#include <vector>

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

  private:
    int t_ = 0, c_ = 0;
    std::vector<float> data_;
};

// Class index -> UTF-8 symbol. `blank` is the CTC blank class.
struct Charset {
    std::vector<std::string> symbols;
    int blank = 0;
};

}  // namespace cudaocr
