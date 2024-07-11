#ifndef UTILS_CUH
#define UTILS_CUH

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <fstream>

enum class LogLevel {
    INFO,
    DEBUG,
    ERROR
};

class Logger {
public:
    static void init(const std::string& logFilePath);
    static void log(LogLevel level, const std::string& message);
    static void close();

private:
    static std::ofstream logFile;
};

// Function Declarations
bool loadImageFromPath(const std::string& filePath, cv::Mat& cpuImage);
bool uploadImageToGPU(const cv::Mat& cpuImage, cv::cuda::GpuMat& gpuImage);
bool downloadImageFromGPU(const cv::cuda::GpuMat& gpuImage, cv::Mat& cpuImage);
bool saveImage(const cv::Mat& cpuImage, const std::string& filePath);
void printGpuMatInfo(const cv::cuda::GpuMat& mat, const std::string& name);

#endif 
