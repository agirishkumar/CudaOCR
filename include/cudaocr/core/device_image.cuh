#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cudaocr/core/host_image.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Move-only RAII pitched 2D 8-bit device image (interleaved channels).
class DeviceImage {
  public:
    DeviceImage() = default;
    ~DeviceImage();
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&& other) noexcept;
    DeviceImage& operator=(DeviceImage&& other) noexcept;

    static Result<DeviceImage> create(int width, int height, int channels);
    static Result<DeviceImage> fromHost(const HostImage& host, cudaStream_t stream = nullptr);

    // Keeps the allocation if the shape already matches, otherwise (re)allocates.
    // Contents are unspecified afterwards.
    Status ensure(int width, int height, int channels);
    // Device-to-device copy into `dst` (allocated/reshaped as needed).
    Status copyTo(DeviceImage& dst, cudaStream_t stream = nullptr) const;

    // Ordered on `stream` (pipeline streams are non-blocking, so the default
    // stream would race their kernels); both return once the copy completed.
    Status upload(const HostImage& host, cudaStream_t stream = nullptr);
    Status download(HostImage& host, cudaStream_t stream = nullptr) const;

    std::uint8_t* data() { return data_; }
    const std::uint8_t* data() const { return data_; }
    [[nodiscard]] size_t pitch() const { return pitch_; }
    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] int channels() const { return channels_; }
    [[nodiscard]] bool empty() const { return data_ == nullptr; }

  private:
    std::uint8_t* data_ = nullptr;
    size_t pitch_ = 0;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
};

}  // namespace cudaocr
