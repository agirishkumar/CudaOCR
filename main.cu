#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "checks.cuh"
// #include "preprocessing.cuh"
// #include "segmentation.cuh"
#include "config.h"
#include "utils.cuh"

int main() {
    // Initialize logger
    Logger::init("ocr_log.txt", INFO);
    LOG(INFO, "OCR process started");

    // Load image
    cv::Mat h_colorImg = cv::imread(INPUT_IMAGE_PATH, cv::IMREAD_COLOR);
    LOG(DEBUG, "Attempting to load image from: " + std::string(INPUT_IMAGE_PATH));

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
        return -1;
    }
    LOG(INFO, "Image quality check passed");

    // Add more OCR pipeline steps here...

    LOG(INFO, "OCR process completed successfully");
    Logger::close();
    return 0;
}
