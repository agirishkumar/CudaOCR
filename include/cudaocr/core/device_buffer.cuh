#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cudaocr/core/cuda_check.hpp"
#include "cudaocr/core/status.hpp"

namespace cudaocr {

// Move-only RAII linear device allocation of n elements of T.
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    static Result<DeviceBuffer> create(size_t n) {
        DeviceBuffer buf;
        if (n == 0) return Status::Error(ErrorCode::InvalidArgument, "DeviceBuffer: n == 0");
        CUDAOCR_CHECK_R(cudaMalloc(&buf.data_, n * sizeof(T)));
        buf.size_ = n;
        return buf;
    }

    Status zero() {
        CUDAOCR_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
        return Status::Good();
    }

    Status upload(const std::vector<T>& host) {
        if (host.size() != size_)
            return Status::Error(ErrorCode::InvalidArgument, "DeviceBuffer::upload size mismatch");
        CUDAOCR_CHECK(cudaMemcpy(data_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
        return Status::Good();
    }

    Status download(std::vector<T>& host) const {
        host.resize(size_);
        CUDAOCR_CHECK(cudaMemcpy(host.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return Status::Good();
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    [[nodiscard]] size_t size() const { return size_; }

  private:
    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    void reset() {
        if (data_) cudaFree(data_);
        data_ = nullptr;
        size_ = 0;
    }
    T* data_ = nullptr;
    size_t size_ = 0;
};

// RAII non-blocking CUDA stream.
class Stream {
  public:
    Stream() = default;
    ~Stream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& other) noexcept { std::swap(stream_, other.stream_); }
    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) std::swap(stream_, other.stream_);
        return *this;
    }

    static Result<Stream> create() {
        Stream s;
        CUDAOCR_CHECK_R(cudaStreamCreateWithFlags(&s.stream_, cudaStreamNonBlocking));
        return s;
    }
    cudaStream_t get() const { return stream_; }
    Status synchronize() {
        CUDAOCR_CHECK(cudaStreamSynchronize(stream_));
        return Status::Good();
    }

  private:
    cudaStream_t stream_ = nullptr;  // nullptr == legacy default stream
};

}  // namespace cudaocr
