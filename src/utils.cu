#include "utils.cuh"
#include <iostream>

// Function Definitions

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
