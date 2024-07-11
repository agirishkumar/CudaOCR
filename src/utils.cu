#include "utils.cuh"
#include <iostream>
#include <chrono>
#include <iomanip>


std::ofstream Logger::logFile;

void Logger::init(const std::string& logFilePath) {
    logFile.open(logFilePath, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!logFile.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = std::localtime(&now_c);

    logFile << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " ";

    switch (level) {
        case LogLevel::INFO:  logFile << "[INFO] "; break;
        case LogLevel::DEBUG: logFile << "[DEBUG] "; break;
        case LogLevel::ERROR: logFile << "[ERROR] "; break;
    }

    logFile << message << std::endl;
}

void Logger::close() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

bool loadImageFromPath(const std::string& filePath, cv::Mat& cpuImage) {
    cpuImage = cv::imread(filePath);
    if (cpuImage.empty()) {
        std::cerr << "Error: Could not load image from " << filePath << std::endl;
        return false;
    }
    return true;
}

bool uploadImageToGPU(const cv::Mat& cpuImage, cv::cuda::GpuMat& gpuImage) {
    gpuImage.upload(cpuImage);
    if (gpuImage.empty()) {
        std::cerr << "Error: Could not upload image to GPU" << std::endl;
        return false;
    }
    return true;
}

bool downloadImageFromGPU(const cv::cuda::GpuMat& gpuImage, cv::Mat& cpuImage) {
    gpuImage.download(cpuImage);
    return true; 
}

bool saveImage(const cv::Mat& cpuImage, const std::string& filePath) {
    if (!cv::imwrite(filePath, cpuImage)) {
        std::cerr << "Error: Could not save image to " << filePath << std::endl;
        return false;
    }
    return true;
}

void printGpuMatInfo(const cv::cuda::GpuMat& mat, const std::string& name) {
    std::stringstream ss;
    ss << "GpuMat " << name << ": "
       << "rows=" << mat.rows << ", "
       << "cols=" << mat.cols << ", "
       << "type=" << mat.type() << ", "
       << "channels=" << mat.channels() << ", "
       << "elem_size=" << mat.elemSize();
    Logger::log(LogLevel::DEBUG, ss.str());
}
