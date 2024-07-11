#include "page_analysis.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <cmath>

#include "page_analysis.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <cmath>

__global__ void houghTransform(const unsigned char* image, int width, int height, int* accumulator, int rho, int theta, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const unsigned char* row = (unsigned char*)((char*)image + y * pitch);
        if (row[x] > 0) {
            for (int t = 0; t < theta; ++t) {
                float rads = (t - 90) * M_PI / 180;
                int r = int(x * cos(rads) + y * sin(rads)) + rho;
                if (r >= 0 && r < rho * 2) {
                    atomicAdd(&accumulator[r * theta + t], 1);
                }
            }
        }
    }
}

float detectSkewAngle(const cv::cuda::GpuMat& d_image) {
    Logger::log(LogLevel::INFO, "Starting skew angle detection");

    int width = d_image.cols;
    int height = d_image.rows;
    int diag = std::sqrt(width * width + height * height);
    int rho = diag * 2;
    int theta = 180;

    Logger::log(LogLevel::DEBUG, "Image dimensions: " + std::to_string(width) + "x" + std::to_string(height) +
                ", Rho: " + std::to_string(rho) + ", Theta: " + std::to_string(theta));

    // Allocate host memory for accumulator
    std::vector<int> h_accumulator(rho * theta, 0);

    // Allocate device memory for accumulator
    int* d_accumulator;
    size_t accumulator_size = rho * theta * sizeof(int);
    cudaError_t cudaStatus = cudaMalloc(&d_accumulator, accumulator_size);
    if (cudaStatus != cudaSuccess) {
        Logger::log(LogLevel::ERROR, "CUDA malloc failed: " + std::string(cudaGetErrorString(cudaStatus)));
        return 0.0f;
    }

    cudaStatus = cudaMemset(d_accumulator, 0, accumulator_size);
    if (cudaStatus != cudaSuccess) {
        Logger::log(LogLevel::ERROR, "CUDA memset failed: " + std::string(cudaGetErrorString(cudaStatus)));
        cudaFree(d_accumulator);
        return 0.0f;
    }

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    Logger::log(LogLevel::INFO, "Launching Hough transform kernel");
    houghTransform<<<grid, block>>>(d_image.ptr<unsigned char>(), width, height, d_accumulator, rho, theta, d_image.step);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        Logger::log(LogLevel::ERROR, "CUDA error in Hough transform: " + std::string(cudaGetErrorString(cudaStatus)));
        cudaFree(d_accumulator);
        return 0.0f;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        Logger::log(LogLevel::ERROR, "CUDA synchronization error: " + std::string(cudaGetErrorString(cudaStatus)));
        cudaFree(d_accumulator);
        return 0.0f;
    }

    // Copy accumulator data back to host
    cudaStatus = cudaMemcpy(h_accumulator.data(), d_accumulator, accumulator_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        Logger::log(LogLevel::ERROR, "CUDA memcpy failed: " + std::string(cudaGetErrorString(cudaStatus)));
        cudaFree(d_accumulator);
        return 0.0f;
    }

    cudaFree(d_accumulator);

    Logger::log(LogLevel::DEBUG, "Finding maximum element in accumulator");
    auto max_element = std::max_element(h_accumulator.begin(), h_accumulator.end());
    int max_index = std::distance(h_accumulator.begin(), max_element);

    int max_theta = max_index % theta;
    float skew_angle = max_theta - 90;

    Logger::log(LogLevel::INFO, "Skew angle detected: " + std::to_string(skew_angle));
    return skew_angle;
}

cv::cuda::GpuMat correctSkew(const cv::cuda::GpuMat& d_image, float angle)
{
    Logger::log(LogLevel::INFO, "Starting skew correction, angle: " + std::to_string(angle));

    cv::Mat cpuImage;
    d_image.download(cpuImage);

    cv::Point2f center(cpuImage.cols / 2.0, cpuImage.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotatedImage;
    cv::warpAffine(cpuImage, rotatedImage, rotationMatrix, cpuImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::cuda::GpuMat d_rotatedImage;
    d_rotatedImage.upload(rotatedImage);

    Logger::log(LogLevel::INFO, "Skew correction completed");
    return d_rotatedImage;
}

cv::cuda::GpuMat pageLayoutAnalysis(const cv::cuda::GpuMat& d_inputImage)
{
    Logger::log(LogLevel::INFO, "Starting page layout analysis");
    printGpuMatInfo(d_inputImage, "Input image for page layout analysis");

    if (d_inputImage.empty() || d_inputImage.type() != CV_8UC1) {
        Logger::log(LogLevel::ERROR, "Invalid input image for page layout analysis");
        return d_inputImage;
    }

    float skewAngle = detectSkewAngle(d_inputImage);
    if (std::isnan(skewAngle) || std::isinf(skewAngle)) {
        Logger::log(LogLevel::ERROR, "Invalid skew angle detected. Skipping correction.");
        return d_inputImage;
    }

    cv::cuda::GpuMat d_correctedImage = correctSkew(d_inputImage, skewAngle);

    Logger::log(LogLevel::INFO, "Page layout analysis completed");
    return d_correctedImage;
}
