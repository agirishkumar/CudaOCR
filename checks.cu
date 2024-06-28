#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime.h>
#include "config.h"
#include "utils.cuh"



bool check_uploaded_img(const cv::Mat &image) {
    if (image.empty()) {
        LOG(ERROR, "Failed to load image");
        return false;
    }
    if (image.type() != CV_8UC3) {
        LOG(ERROR, "Unexpected image type. Expected CV_8UC3 (3-channel 8-bit color image)");
        return false;
    }
    if (image.cols <= 0 || image.rows <= 0) {
        LOG(ERROR, "Invalid image size");
        return false;
    }
    if (image.depth() != CV_8U) {
        LOG(ERROR, "Unexpected image depth. Expected 8-bit depth");
        return false;
    }
    LOG(DEBUG, "Image checks passed. Size: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));
    return true;
}

bool check_gpu_memory(const cv::Mat &image) {
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = image.total() * image.elemSize();
    if (free_mem < required_mem) {
        LOG(ERROR, "Insufficient GPU memory. Required: " + std::to_string(required_mem) + ", Available: " + std::to_string(free_mem));
        return false;
    }
    LOG(DEBUG, "GPU memory check passed. Free memory: " + std::to_string(free_mem));
    return true;
}

bool is_image_quality_acceptable(const cv::cuda::GpuMat &d_image) {
    if (d_image.cols < MIN_IMAGE_WIDTH || d_image.rows < MIN_IMAGE_HEIGHT) {
        LOG(ERROR, "Image resolution too low. Size: " + std::to_string(d_image.cols) + "x" + std::to_string(d_image.rows));
        return false;
    }

    cv::cuda::GpuMat d_gray;
    cv::cuda::cvtColor(d_image, d_gray, cv::COLOR_BGR2GRAY);
    
    cv::cuda::GpuMat d_laplacian;
    cv::Ptr<cv::cuda::Filter> laplacianFilter = cv::cuda::createLaplacianFilter(d_gray.type(), -1, 1);
    laplacianFilter->apply(d_gray, d_laplacian);
    
    cv::Mat h_laplacian;
    d_laplacian.download(h_laplacian);
    double laplacian_variance = cv::mean(h_laplacian.mul(h_laplacian))[0];
    
    if (laplacian_variance < BLUR_DETECTION_VARIANCE_THRESHOLD) {
        LOG(WARNING, "Image is too blurry. Variance: " + std::to_string(laplacian_variance));
        return false;
    }
    
    // Calculate mean and stddev
    cv::Mat h_gray;
    d_gray.download(h_gray);
    cv::Scalar mean, stddev;
    cv::meanStdDev(h_gray, mean, stddev);

    if (mean[0] < BRIGHTNESS_DETECTION_MEAN_LOWER_THRESHOLD || mean[0] > BRIGHTNESS_DETECTION_MEAN_UPPER_THRESHOLD) {
        LOG(WARNING, "Image brightness is not within acceptable range. Mean: " + std::to_string(mean[0]));
        return false;
    }
    
    if (stddev[0] < CONTRAST_DETECTION_STDDEV_THRESHOLD) {
        LOG(WARNING, "Image contrast is too low. Standard Deviation: " + std::to_string(stddev[0]));
        return false;
    }
    
    LOG(INFO, "Image quality check passed");
    return true;
}