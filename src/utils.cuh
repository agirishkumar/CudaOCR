#ifndef UTILS_CUH
#define UTILS_CUH

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>

// Function Declarations
bool loadImageFromPath(const std::string& filePath, cv::Mat& cpuImage);
bool uploadImageToGPU(const cv::Mat& cpuImage, cv::cuda::GpuMat& gpuImage);
bool downloadImageFromGPU(const cv::cuda::GpuMat& gpuImage, cv::Mat& cpuImage);
bool saveImage(const cv::Mat& cpuImage, const std::string& filePath);

#endif 
