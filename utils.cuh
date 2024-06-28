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

#define LOG(level, message) Logger::log(level, message, __FILE__, __LINE__)

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

// Other utility functions...