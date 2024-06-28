#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include "checks.cuh"
#include "preprocessing.cuh"
#include "config.h"
#include "utils.cuh"

int main() {
    // Initialize logger
    Logger::init("ocr_log.txt", INFO);
    LOG(INFO, "OCR process started");

    // Load image
    cv::Mat h_colorImg = cv::imread(INPUT_IMAGE_PATH, cv::IMREAD_COLOR);
    LOG(DEBUG, "Attempting to load image from: " << INPUT_IMAGE_PATH);

    // Perform image checks
    if (!check_uploaded_img(h_colorImg)) {
        LOG(ERROR, "Image upload check failed");
        Logger::close();
        return -1;
    }
    LOG(INFO, "Image upload check passed");

    // Preprocess image
    cv::cuda::GpuMat d_colorImg, d_preprocessedImg;
    d_colorImg.upload(h_colorImg);
    LOG(INFO, "Image loaded and transferred to GPU memory successfully");

    if (!check_gpu_memory(h_colorImg)) {
        LOG(ERROR, "Insufficient GPU memory");
        Logger::close();
        return -1;
    }
    LOG(INFO, "GPU memory check passed");

    if (!is_image_quality_acceptable(d_colorImg)) {
        LOG(ERROR, "Image quality check failed");
        Logger::close();
        return -1;
    }
    LOG(INFO, "Image quality check passed");

    // Preprocess the image
    try {
        preprocessImage(d_colorImg, d_preprocessedImg);
    } catch (const std::exception& e) {
        LOG(ERROR, "Preprocessing failed: " << e.what());
        Logger::close();
        return -1;
    }

    // Download the preprocessed image from GPU to CPU
    cv::Mat h_preprocessedImg;
    d_preprocessedImg.download(h_preprocessedImg);

    // Display the original and preprocessed images
    cv::imshow("Original Image", h_colorImg);
    cv::imshow("Preprocessed Image", h_preprocessedImg);
    cv::waitKey(0);

    // Save the preprocessed image
    std::string outputPath = "preprocessed_" + std::string(INPUT_IMAGE_PATH);
    cv::imwrite(outputPath, h_preprocessedImg);
    LOG(INFO, "Preprocessed image saved to: " << outputPath);

    // Add more OCR pipeline steps here...

    LOG(INFO, "OCR process completed successfully");
    Logger::close();
    return 0;
}