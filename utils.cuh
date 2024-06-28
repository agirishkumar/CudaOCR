#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

class Logger {
private:
    static std::ofstream logFile;
    static LogLevel currentLevel;

    static std::string getTimestamp();
    static std::string getLevelString(LogLevel level);

public:
    static void init(const std::string& filename, LogLevel level);
    static void log(LogLevel level, const std::string& message, const char* file, int line);
    static void close();
};

#define LOG(level, message) \
    do { \
        std::ostringstream oss; \
        oss << message; \
        Logger::log(level, oss.str(), __FILE__, __LINE__); \
    } while(0)
    
// CUDA error checking
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA error: " << cudaGetErrorString(err); \
        LOG(ERROR, ss.str()); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA error checking macro
#define CUDA_SAFE_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        std::stringstream ss; \
        ss << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
           << cudaGetErrorString(err); \
        LOG(ERROR, ss.str()); \
        throw std::runtime_error(ss.str()); \
    } \
} while(0)

// Wrapper for cudaMalloc with error checking
inline void* safeCudaMalloc(size_t size) {
    void* ptr;
    CUDA_SAFE_CALL(cudaMalloc(&ptr, size));
    return ptr;
}

// Wrapper for cudaFree with error checking
inline void safeCudaFree(void* ptr) {
    if (ptr) {
        CUDA_SAFE_CALL(cudaFree(ptr));
    }
}

// CUDA Timer class
class CudaTimer {
private:
    cudaEvent_t start, stop;
    float elapsedTime;

public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }

    float ElapsedMilliseconds() const {
        return elapsedTime;
    }
};


// Other utility functions...