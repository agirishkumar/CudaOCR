#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include "config.h" 
#include "utils.cuh"
#include "preprocessing.cuh"

using namespace std;



int main(){

    std::string filePath = INPUT_FILE_PATH;
    cv::Mat cpuImage;
    cv::cuda::GpuMat gpuImage;

    if (!loadImageFromPath(filePath, cpuImage)) {
        return 1;
    }
    if (!uploadImageToGPU(cpuImage, gpuImage)) {
        return 1;
    }

    // preprocessing tasks
    cv::cuda::GpuMat grayGpuImage, denoiseImageNPPGaussianBlurImage, denoiseImageNPPMedianFilterImage, binaryImage;

    if (!convertToGrayscaleNPP(gpuImage, grayGpuImage)) {
        return 1; // Handle the error appropriately
    }

    if (!denoiseImageNPPMedianFilter(grayGpuImage, denoiseImageNPPMedianFilterImage, 3)) {
        return 1;
    }
    if (!denoiseImageNPPMedianFilter(grayGpuImage, denoiseImageNPPMedianFilterImage, 1)) {
        return 1;
    }
    if (!adaptiveThresholdOpenCV(denoiseImageNPPMedianFilterImage, binaryImage, 3, 2.0)) {
    return 1;
    }

    if (!downloadImageFromGPU(binaryImage, cpuImage)) {
        return 1;
    }
    if (!saveImage(cpuImage, "output/processed_binary_image.jpg")) {
        return 1;
    }

    // if (!downloadImageFromGPU(denoiseImageNPPMedianFilterImage, cpuImage)) {
    //     return 1;
    // }
    // if (!saveImage(cpuImage, "output/processed_denoised_median_image.jpg")) {
    //     return 1;
    // }

    // if (!denoiseImageNPP(grayGpuImage, denoiseImageNPPGaussianBlurImage, 3)) {
    //     return 1;
    // }

    // if (!downloadImageFromGPU(denoiseImageNPPGaussianBlurImage, cpuImage)) {
    //     return 1;
    // }
    // if (!saveImage(cpuImage, "output/processed_denoised_blurred_image.jpg")) {
    //     return 1;
    // }

    // if (!downloadImageFromGPU(grayGpuImage, cpuImage)) {
    //     return 1;
    // }

    // if (!saveImage(cpuImage, "output/processed_gray_image.jpg")) {
    //     return 1;
    // }
    return 0;
}