#pragma once

// Per-run state handed to every component call. Host-only.

#include "cudaocr/types.hpp"

struct CUstream_st;

namespace cudaocr {

class DeviceImage;  // core/device_image.cuh — interfaces take it by reference only

struct StageContext {
    CUstream_st* stream = nullptr;  // == cudaStream_t; null = default stream
    // Page being produced. Components may record metadata here (e.g. polarity
    // sets `inverted`); null in isolated unit tests.
    PageResult* page = nullptr;
};

}  // namespace cudaocr
