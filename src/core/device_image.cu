#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/device_image.cuh"

#include <utility>

namespace cudaocr {

DeviceImage::~DeviceImage() {
    if (data_) cudaFree(data_);
}

DeviceImage::DeviceImage(DeviceImage&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(pitch_, other.pitch_);
    std::swap(width_, other.width_);
    std::swap(height_, other.height_);
    std::swap(channels_, other.channels_);
}

DeviceImage& DeviceImage::operator=(DeviceImage&& other) noexcept {
    if (this != &other) {
        if (data_) cudaFree(data_);
        data_ = nullptr;
        pitch_ = 0;
        width_ = height_ = channels_ = 0;
        std::swap(data_, other.data_);
        std::swap(pitch_, other.pitch_);
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
        std::swap(channels_, other.channels_);
    }
    return *this;
}

Result<DeviceImage> DeviceImage::create(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        return Status::Error(ErrorCode::InvalidArgument,
                             "DeviceImage::create: bad dimensions " + std::to_string(width) + "x" +
                                 std::to_string(height) + "x" + std::to_string(channels));
    }
    DeviceImage img;
    void* ptr = nullptr;
    size_t pitch = 0;
    CUDAOCR_CHECK_R(cudaMallocPitch(&ptr, &pitch, static_cast<size_t>(width) * channels, height));
    img.data_ = static_cast<std::uint8_t*>(ptr);
    img.pitch_ = pitch;
    img.width_ = width;
    img.height_ = height;
    img.channels_ = channels;
    return img;
}

Status DeviceImage::ensure(int width, int height, int channels) {
    if (data_ && width == width_ && height == height_ && channels == channels_)
        return Status::Good();
    auto img = create(width, height, channels);
    if (!img.ok()) return img.status();
    *this = img.take();
    return Status::Good();
}

Status DeviceImage::copyTo(DeviceImage& dst, cudaStream_t stream) const {
    if (&dst == this)
        return Status::Error(ErrorCode::InvalidArgument, "DeviceImage::copyTo: dst is src");
    Status s = dst.ensure(width_, height_, channels_);
    if (!s.ok()) return s;
    CUDAOCR_CHECK(cudaMemcpy2DAsync(dst.data_, dst.pitch_, data_, pitch_,
                                    static_cast<size_t>(width_) * channels_, height_,
                                    cudaMemcpyDeviceToDevice, stream));
    return Status::Good();
}

Result<DeviceImage> DeviceImage::fromHost(const HostImage& host, cudaStream_t stream) {
    auto img = create(host.width, host.height, host.channels);
    if (!img.ok()) return img.status();
    Status s = img.value().upload(host, stream);
    if (!s.ok()) return s;
    return img;
}

Status DeviceImage::upload(const HostImage& host, cudaStream_t stream) {
    if (host.width != width_ || host.height != height_ || host.channels != channels_) {
        return Status::Error(ErrorCode::InvalidArgument, "DeviceImage::upload: shape mismatch");
    }
    CUDAOCR_CHECK(cudaMemcpy2DAsync(data_, pitch_, host.data.data(), host.rowBytes(),
                                    host.rowBytes(), height_, cudaMemcpyHostToDevice, stream));
    CUDAOCR_CHECK(cudaStreamSynchronize(stream));
    return Status::Good();
}

Status DeviceImage::download(HostImage& host, cudaStream_t stream) const {
    host = HostImage::create(width_, height_, channels_);
    CUDAOCR_CHECK(cudaMemcpy2DAsync(host.data.data(), host.rowBytes(), data_, pitch_,
                                    host.rowBytes(), height_, cudaMemcpyDeviceToHost, stream));
    CUDAOCR_CHECK(cudaStreamSynchronize(stream));
    return Status::Good();
}

}  // namespace cudaocr
