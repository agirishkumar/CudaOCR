#pragma once

#include <optional>
#include <string>
#include <utility>

namespace cudaocr {

enum class ErrorCode { Ok, InvalidArgument, CudaError, IoError, InternalError };

struct Status {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    [[nodiscard]] bool ok() const { return code == ErrorCode::Ok; }
    static Status Good() { return {}; }
    static Status Error(ErrorCode c, std::string msg) { return {c, std::move(msg)}; }
};

// Minimal expected-like type: holds T on success, Status on failure.
template <typename T>
class Result {
  public:
    Result(T value) : value_(std::move(value)) {}
    Result(Status status) : status_(std::move(status)) {}

    [[nodiscard]] bool ok() const { return value_.has_value(); }
    [[nodiscard]] const Status& status() const { return status_; }
    [[nodiscard]] T& value() { return *value_; }
    [[nodiscard]] const T& value() const { return *value_; }
    T take() { return std::move(*value_); }

  private:
    std::optional<T> value_;
    Status status_ = Status::Good();
};

}  // namespace cudaocr
