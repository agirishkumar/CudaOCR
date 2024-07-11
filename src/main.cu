#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include "config.h" 
#include "utils.cuh"
#include "preprocessing.cuh"
#include "page_analysis.cuh"

using namespace std;



int main(){

    Logger::init("ocr_pipeline.log");
    Logger::log(LogLevel::INFO, "OCR Application started");

    std::string filePath = INPUT_FILE_PATH;
    cv::Mat cpuImage;
    cv::cuda::GpuMat gpuImage;

    if (!loadImageFromPath(filePath, cpuImage)) {
        Logger::log(LogLevel::ERROR, "Failed to load image. Exiting.");
        Logger::close();
        return 1;
    }
    if (!uploadImageToGPU(cpuImage, gpuImage)) {
        Logger::log(LogLevel::ERROR, "Failed to upload image to GPU. Exiting.");
        Logger::close();
        return 1;
    }

    printGpuMatInfo(gpuImage, "Input image on GPU");

    // preprocessing tasks
    cv::cuda::GpuMat grayGpuImage, denoiseImageNPPGaussianBlurImage, denoiseImageNPPMedianFilterImage, binaryImage;
    Logger::log(LogLevel::INFO, "Starting grayscale conversion");
    if (!convertToGrayscaleNPP(gpuImage, grayGpuImage)) {
        Logger::log(LogLevel::ERROR, "Failed to convert image to grayscale. Exiting.");
        Logger::close();
        return 1;
    }
    printGpuMatInfo(grayGpuImage, "Grayscale image");

    Logger::log(LogLevel::INFO, "Starting denoising");
    if (!denoiseImageNPPMedianFilter(grayGpuImage, denoiseImageNPPMedianFilterImage, 1)) {
        Logger::log(LogLevel::ERROR, "Failed to denoise image. Exiting.");
        Logger::close();
        return 1;
    }
    printGpuMatInfo(denoiseImageNPPMedianFilterImage, "Denoised image");

    Logger::log(LogLevel::INFO, "Starting binarization");
    if (!adaptiveThresholdOpenCV(denoiseImageNPPMedianFilterImage, binaryImage, 3, 2.0)) {
        Logger::log(LogLevel::ERROR, "Failed to binarize image. Exiting.");
        Logger::close();
        return 1;
    }
    printGpuMatInfo(binaryImage, "Binary image");

    // Page Layout Analysis
    cv::cuda::GpuMat layoutAnalyzedImage = pageLayoutAnalysis(binaryImage);



    // Download the result to CPU
    cv::Mat cpuProcessedImage;
    if (!downloadImageFromGPU(layoutAnalyzedImage, cpuProcessedImage)) {
        Logger::log(LogLevel::ERROR, "Failed to download processed image from GPU. Exiting.");
        Logger::close();
        return 1;
    }

    // Save the result
    if (!saveImage(cpuProcessedImage, "output/processed_image.jpg")) {
        Logger::log(LogLevel::ERROR, "Failed to save processed image. Exiting.");
        Logger::close();
        return 1;
    }

    // if (!downloadImageFromGPU(binaryImage, cpuProcessedImage)) {
    //     return 1;
    // }
    // if (!saveImage(cpuProcessedImage, "output/processed_binary_image.jpg")) {
    //     return 1;
    // }

    // if (!downloadImageFromGPU(denoiseImageNPPMedianFilterImage, cpuProcessedImage)) {
    //     return 1;
    // }
    // if (!saveImage(cpuProcessedImage, "output/processed_denoised_median_image.jpg")) {
    //     return 1;
    // }

    // if (!denoiseImageNPP(grayGpuImage, denoiseImageNPPGaussianBlurImage, 3)) {
    //     return 1;
    // }

    // if (!downloadImageFromGPU(denoiseImageNPPGaussianBlurImage, cpuProcessedImage)) {
    //     return 1;
    // }
    // if (!saveImage(cpuProcessedImage, "output/processed_denoised_blurred_image.jpg")) {
    //     return 1;
    // }

    // if (!downloadImageFromGPU(grayGpuImage, cpuProcessedImage)) {
    //     return 1;
    // }

    // if (!saveImage(cpuProcessedImage, "output/processed_gray_image.jpg")) {
    //     return 1;
    // }
    Logger::log(LogLevel::INFO, "OCR Pipeline completed successfully");
    Logger::close();
    return 0;
}