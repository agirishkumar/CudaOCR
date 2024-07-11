#ifndef PREPROCESSING_CUH
#define PREPROCESSING_CUH

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>


bool convertToGrayscaleGPU(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage);
bool convertToGrayscaleNPP(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage);
bool denoiseImageNPPGaussianBlur(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int kernelSize = 3);
bool denoiseImageNPPMedianFilter(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int kernelSize );
bool adaptiveThresholdOpenCV(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int blockSize, double C);


#endif // PREPROCESSING_CUH
