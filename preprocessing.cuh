#ifndef PREPROCESSING_CUH
#define PREPROCESSING_CUH

#include "config.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

// Function declarations for individual kernels
__global__ void grayscaleConversionKernel(const uchar3* input, unsigned char* output, int width, int height);
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height);
__global__ void medianFilterKernel(const unsigned char* input, unsigned char* output, int width, int height);
__global__ void histogramEqualizationKernel(unsigned char* image, int width, int height);
__global__ void adaptiveHistogramEqualizationKernel(unsigned char* image, int width, int height);

// Main preprocessing function declaration
void preprocessImage(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output);

#endif // PREPROCESSING_CUH
