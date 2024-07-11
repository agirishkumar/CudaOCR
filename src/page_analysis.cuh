#ifndef PAGE_ANALYSIS_CUH
#define PAGE_ANALYSIS_CUH

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

float detectSkewAngle(const cv::cuda::GpuMat& d_image);
cv::cuda::GpuMat correctSkew(const cv::cuda::GpuMat& d_image, float angle);
cv::cuda::GpuMat pageLayoutAnalysis(const cv::cuda::GpuMat& d_inputImage);

#endif 
