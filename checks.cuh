#ifndef CHECKS_CUH
#define CHECKS_CUH

#include <opencv2/opencv.hpp>

// Function declaration for checking the uploaded image
bool check_uploaded_img(const cv::Mat &image);

// Function declaration for filtering out low-quality images
bool is_image_quality_acceptable(const cv::cuda::GpuMat &d_image);

// Function declaration for checking GPU memory availability
bool check_gpu_memory(const cv::Mat &image);

#endif // CHECKS_CUH
