#include "preprocessing.cuh"
#include <iostream>
#include <npp.h>
#include <nppi.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

bool convertToGrayscaleGPU(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage) {
    try {
        cv::cuda::cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error during GPU grayscale conversion: " << e.what() << std::endl;
        return false;
    }
}


bool convertToGrayscaleNPP(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage) {
    NppiSize oSizeROI = {static_cast<int>(inputImage.cols), static_cast<int>(inputImage.rows)};
    
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv::cuda::Stream::Null());
    
    outputImage.create(inputImage.size(), CV_8UC1);

    const Npp32f aCoeffs[3] = {0.299f, 0.587f, 0.114f};

    NppStatus status = nppiColorToGray_8u_C3C1R(
        inputImage.ptr<Npp8u>(),
        static_cast<int>(inputImage.step),
        outputImage.ptr<Npp8u>(),
        static_cast<int>(outputImage.step),
        oSizeROI,
        aCoeffs
    );

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        return false;
    }

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
        return false;
    }

    return true;
}

bool denoiseImageNPPGaussianBlur(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int kernelSize) {
    NppiSize oSizeROI = {static_cast<int>(inputImage.cols), static_cast<int>(inputImage.rows)};
    
    // Ensure kernel size is odd
    kernelSize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
    
    outputImage.create(inputImage.size(), inputImage.type());

    NppiMaskSize eMaskSize;
    switch(kernelSize) {
        case 3: eMaskSize = NPP_MASK_SIZE_3_X_3; break;
        case 5: eMaskSize = NPP_MASK_SIZE_5_X_5; break;
        case 7: eMaskSize = NPP_MASK_SIZE_7_X_7; break;
        default:
            std::cerr << "Unsupported kernel size. Using 3x3." << std::endl;
            eMaskSize = NPP_MASK_SIZE_3_X_3;
            break;
    }

    NppStatus status = nppiFilterGauss_8u_C1R(
        inputImage.ptr<Npp8u>(),
        static_cast<int>(inputImage.step),
        outputImage.ptr<Npp8u>(),
        static_cast<int>(outputImage.step),
        oSizeROI,
        eMaskSize
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: Gaussian filter failed with error " << status << std::endl;
        return false;
    }

    return true;
}

bool denoiseImageNPPMedianFilter(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int kernelSize) {
    NppiSize oSizeROI = {static_cast<int>(inputImage.cols), static_cast<int>(inputImage.rows)};
    
    // Ensure kernel size is odd
    kernelSize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
    
    outputImage.create(inputImage.size(), inputImage.type());

    NppiSize oMaskSize = {kernelSize, kernelSize};
    NppiPoint oAnchor = {kernelSize / 2, kernelSize / 2};

    // Create a temporary buffer for the median filter
    Npp8u* pBuffer;
    Npp32u bufferSize = 0;
    NppStatus status = nppiFilterMedianGetBufferSize_8u_C1R(oSizeROI, oMaskSize, &bufferSize);
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: Failed to get buffer size for median filter" << std::endl;
        return false;
    }

    cudaMalloc(&pBuffer, bufferSize);

    status = nppiFilterMedian_8u_C1R(
        inputImage.ptr<Npp8u>(),
        static_cast<int>(inputImage.step),
        outputImage.ptr<Npp8u>(),
        static_cast<int>(outputImage.step),
        oSizeROI,
        oMaskSize,
        oAnchor,
        pBuffer
    );

    cudaFree(pBuffer);

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: Median filter failed with error " << status << std::endl;
        return false;
    }

    return true;
}


bool adaptiveThresholdOpenCV(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage, int blockSize, double C) {
    try {
        // Ensure block size is odd
        blockSize = (blockSize % 2 == 0) ? blockSize + 1 : blockSize;

        // Create output image
        outputImage.create(inputImage.size(), CV_8UC1);

        // Create a mean filter
        cv::Ptr<cv::cuda::Filter> meanFilter = cv::cuda::createBoxFilter(
            CV_8UC1, CV_8UC1, cv::Size(blockSize, blockSize));

        // Apply mean filter to get the local mean
        cv::cuda::GpuMat localMean;
        meanFilter->apply(inputImage, localMean);

        // Subtract the constant C from the local mean
        cv::cuda::subtract(localMean, cv::Scalar(C), localMean);

        // Apply the threshold
        cv::cuda::compare(inputImage, localMean, outputImage, cv::CMP_GT);

        // Convert boolean mask to 0 and 255
        outputImage.convertTo(outputImage, CV_8UC1, 255.0);

        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return false;
    }
}