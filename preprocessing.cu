#include "preprocessing.cuh"
#include "config.h"
#include "utils.cuh"
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <thrust/swap.h>

// Constant memory for Gaussian kernel
__constant__ float gaussianKernel[MAX_KERNEL_SIZE];

// Helper device functions
__device__ void sort3(unsigned char& a, unsigned char& b, unsigned char& c) {
    if (a > b) thrust::swap(a, b);
    if (b > c) thrust::swap(b, c);
    if (a > b) thrust::swap(a, b);
}

// Grayscale conversion kernel
__global__ void grayscaleConversionKernel(const uchar4* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 rgb = input[idx];
        float gray;

        #if GRAYSCALE_METHOD == 0
            gray = (rgb.x + rgb.y + rgb.z) / 3.0f;
        #elif GRAYSCALE_METHOD == 1
            gray = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
        #elif GRAYSCALE_METHOD == 2
            gray = (max(max(rgb.x, rgb.y), rgb.z) + min(min(rgb.x, rgb.y), rgb.z)) / 2.0f;
        #endif

        output[idx] = static_cast<unsigned char>(gray);
    }
}

// Gaussian blur kernels
__global__ void gaussianBlurHorizontal(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_SIZE_Y][BLOCK_SIZE_X + GAUSSIAN_KERNEL_SIZE - 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sharedX = threadIdx.x + GAUSSIAN_KERNEL_SIZE / 2;

    // Load data into shared memory
    if (x < width && y < height) {
        sharedMem[threadIdx.y][sharedX] = input[y * width + x];
    }

    // Load left halo
    if (threadIdx.x < GAUSSIAN_KERNEL_SIZE / 2) {
        int srcX = max(0, x - GAUSSIAN_KERNEL_SIZE / 2);
        sharedMem[threadIdx.y][threadIdx.x] = input[y * width + srcX];
    }

    // Load right halo
    if (threadIdx.x >= blockDim.x - GAUSSIAN_KERNEL_SIZE / 2) {
        int srcX = min(width - 1, x + GAUSSIAN_KERNEL_SIZE / 2);
        sharedMem[threadIdx.y][sharedX + GAUSSIAN_KERNEL_SIZE / 2] = input[y * width + srcX];
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
            sum += sharedMem[threadIdx.y][threadIdx.x + i] * gaussianKernel[i];
        }
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

__global__ void gaussianBlurVertical(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_SIZE_Y + GAUSSIAN_KERNEL_SIZE - 1][BLOCK_SIZE_X];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sharedY = threadIdx.y + GAUSSIAN_KERNEL_SIZE / 2;

    // Load data into shared memory
    if (x < width && y < height) {
        sharedMem[sharedY][threadIdx.x] = input[y * width + x];
    }

    // Load top halo
    if (threadIdx.y < GAUSSIAN_KERNEL_SIZE / 2) {
        int srcY = max(0, y - GAUSSIAN_KERNEL_SIZE / 2);
        sharedMem[threadIdx.y][threadIdx.x] = input[srcY * width + x];
    }

    // Load bottom halo
    if (threadIdx.y >= blockDim.y - GAUSSIAN_KERNEL_SIZE / 2) {
        int srcY = min(height - 1, y + GAUSSIAN_KERNEL_SIZE / 2);
        sharedMem[sharedY + GAUSSIAN_KERNEL_SIZE / 2][threadIdx.x] = input[srcY * width + x];
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
            sum += sharedMem[threadIdx.y + i][threadIdx.x] * gaussianKernel[i];
        }
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// Median filter kernel
__global__ void medianFilterKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    // Load data into shared memory
    if (x < width && y < height) {
        sharedMem[sharedY][sharedX] = input[y * width + x];
    }

    // Load halo regions
    if (threadIdx.x == 0 && x > 0) {
        sharedMem[sharedY][0] = input[y * width + (x - 1)];
    }
    if (threadIdx.x == blockDim.x - 1 && x < width - 1) {
        sharedMem[sharedY][sharedX + 1] = input[y * width + (x + 1)];
    }
    if (threadIdx.y == 0 && y > 0) {
        sharedMem[0][sharedX] = input[(y - 1) * width + x];
    }
    if (threadIdx.y == blockDim.y - 1 && y < height - 1) {
        sharedMem[sharedY + 1][sharedX] = input[(y + 1) * width + x];
    }

    __syncthreads();

    if (x < width && y < height) {
        unsigned char window[9];
        int idx = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                window[idx++] = sharedMem[sharedY + dy][sharedX + dx];
            }
        }

        // Sort the window
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 9; j++) {
                if (window[i] > window[j]) {
                    unsigned char temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        output[y * width + x] = window[4];
    }
}

// Setup function for Gaussian kernel
void setupGaussianKernel(float sigma) {
    float h_kernel[MAX_KERNEL_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
        int x = i - GAUSSIAN_KERNEL_SIZE / 2;
        h_kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += h_kernel[i];
    }
    for (int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
        h_kernel[i] /= sum;
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gaussianKernel, h_kernel, sizeof(float) * GAUSSIAN_KERNEL_SIZE));
}

// Gaussian blur application function
void applyGaussianBlur(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {
    LOG(DEBUG, "Applying Gaussian blur");
    CudaTimer timer;

    setupGaussianKernel(GAUSSIAN_SIGMA);

    cv::cuda::GpuMat temp(input.size(), CV_8UC1);

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    timer.Start();

    // Horizontal pass
    gaussianBlurHorizontal<<<grid, block>>>(input.ptr<unsigned char>(), temp.ptr<unsigned char>(), input.cols, input.rows);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Vertical pass
    gaussianBlurVertical<<<grid, block>>>(temp.ptr<unsigned char>(), output.ptr<unsigned char>(), output.cols, output.rows);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer.Stop();
    LOG(DEBUG, "Gaussian blur time: " << timer.ElapsedMilliseconds() << " ms");
}

// Median filter application function
void applyMedianFilter(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {
    LOG(DEBUG, "Applying Median filter");
    CudaTimer timer;

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    timer.Start();
    medianFilterKernel<<<grid, block>>>(input.ptr<unsigned char>(), output.ptr<unsigned char>(), input.cols, input.rows);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.Stop();

    LOG(DEBUG, "Median filter time: " << timer.ElapsedMilliseconds() << " ms");
}

// Global Histogram Equalization
__global__ void computeHistogramKernel(const unsigned char* input, int width, int height, unsigned int* histogram) {
    __shared__ unsigned int sharedHistogram[HISTOGRAM_BINS];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < HISTOGRAM_BINS) {
        sharedHistogram[tid] = 0;
    }
    __syncthreads();

    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&sharedHistogram[input[idx]], 1);
    }

    __syncthreads();

    // Accumulate shared histogram to global histogram
    if (tid < HISTOGRAM_BINS) {
        atomicAdd(&histogram[tid], sharedHistogram[tid]);
    }
}

__global__ void histogramEqualizationKernel(unsigned char* input, int width, int height, const float* cdf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        input[idx] = static_cast<unsigned char>(255.0f * cdf[input[idx]]);
    }
}

void applyHistogramEqualization(cv::cuda::GpuMat& input) {
    LOG(DEBUG, "Applying global histogram equalization");
    CudaTimer timer;

    unsigned int* d_histogram = (unsigned int*)safeCudaMalloc(HISTOGRAM_BINS * sizeof(unsigned int));
    CUDA_SAFE_CALL(cudaMemset(d_histogram, 0, HISTOGRAM_BINS * sizeof(unsigned int)));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    timer.Start();

    computeHistogramKernel<<<grid, block>>>(input.ptr<unsigned char>(), input.cols, input.rows, d_histogram);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Compute CDF on CPU (can be optimized to GPU if needed)
    unsigned int h_histogram[HISTOGRAM_BINS];
    CUDA_SAFE_CALL(cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    float cdf[HISTOGRAM_BINS];
    unsigned int sum = 0;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        sum += h_histogram[i];
        cdf[i] = static_cast<float>(sum) / (input.cols * input.rows);
    }

    float* d_cdf = (float*)safeCudaMalloc(HISTOGRAM_BINS * sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(d_cdf, cdf, HISTOGRAM_BINS * sizeof(float), cudaMemcpyHostToDevice));

    histogramEqualizationKernel<<<grid, block>>>(input.ptr<unsigned char>(), input.cols, input.rows, d_cdf);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer.Stop();
    LOG(DEBUG, "Histogram equalization time: " << timer.ElapsedMilliseconds() << " ms");

    safeCudaFree(d_histogram);
    safeCudaFree(d_cdf);
}

// Contrast Limited Adaptive Histogram Equalization (CLAHE)
__global__ void computeLocalHistogramsKernel(const unsigned char* input, int width, int height, int tileSize, unsigned int* localHistograms) {
    __shared__ unsigned int sharedHistogram[HISTOGRAM_BINS];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < HISTOGRAM_BINS) {
        sharedHistogram[tid] = 0;
    }
    __syncthreads();

    if (x < width && y < height) {
        int tileX = blockIdx.x;
        int tileY = blockIdx.y;
        int tileIdx = tileY * gridDim.x + tileX;
        int localIdx = input[y * width + x];
        atomicAdd(&sharedHistogram[localIdx], 1);
    }

    __syncthreads();

    // Accumulate shared histogram to global histogram for this tile
    if (tid < HISTOGRAM_BINS) {
        atomicAdd(&localHistograms[blockIdx.y * gridDim.x * HISTOGRAM_BINS + blockIdx.x * HISTOGRAM_BINS + tid], sharedHistogram[tid]);
    }
}

__global__ void applyClipLimitKernel(unsigned int* localHistograms, int numTiles, int tileSize, float clipLimit) {
    int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tileIdx < numTiles) {
        unsigned int* histogram = &localHistograms[tileIdx * HISTOGRAM_BINS];
        unsigned int clippedSum = 0;
        unsigned int maxVal = static_cast<unsigned int>(clipLimit * tileSize * tileSize / HISTOGRAM_BINS);

        for (int i = 0; i < HISTOGRAM_BINS; ++i) {
            if (histogram[i] > maxVal) {
                clippedSum += histogram[i] - maxVal;
                histogram[i] = maxVal;
            }
        }

        unsigned int redistributeValue = clippedSum / HISTOGRAM_BINS;
        for (int i = 0; i < HISTOGRAM_BINS; ++i) {
            histogram[i] += redistributeValue;
        }
    }
}

__global__ void claheKernel(unsigned char* input, int width, int height, int tileSize, const unsigned int* localHistograms) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int tileX = x / tileSize;
        int tileY = y / tileSize;
        int tilesPerRow = (width + tileSize - 1) / tileSize;

        float fx = (float)(x % tileSize) / tileSize;
        float fy = (float)(y % tileSize) / tileSize;

        int tileIdx00 = tileY * tilesPerRow + tileX;
        int tileIdx01 = min(tileIdx00 + 1, (tilesPerRow * ((height + tileSize - 1) / tileSize)) - 1);
        int tileIdx10 = min(tileIdx00 + tilesPerRow, (tilesPerRow * ((height + tileSize - 1) / tileSize)) - 1);
        int tileIdx11 = min(tileIdx10 + 1, (tilesPerRow * ((height + tileSize - 1) / tileSize)) - 1);

        unsigned char pixelValue = input[y * width + x];

        float cdf00 = localHistograms[tileIdx00 * HISTOGRAM_BINS + pixelValue] / (float)(tileSize * tileSize);
        float cdf01 = localHistograms[tileIdx01 * HISTOGRAM_BINS + pixelValue] / (float)(tileSize * tileSize);
        float cdf10 = localHistograms[tileIdx10 * HISTOGRAM_BINS + pixelValue] / (float)(tileSize * tileSize);
        float cdf11 = localHistograms[tileIdx11 * HISTOGRAM_BINS + pixelValue] / (float)(tileSize * tileSize);

        float newValue = (1 - fx) * (1 - fy) * cdf00 +
                         fx * (1 - fy) * cdf01 +
                         (1 - fx) * fy * cdf10 +
                         fx * fy * cdf11;

        input[y * width + x] = static_cast<unsigned char>(255.0f * newValue);
    }
}

void applyAdaptiveHistogramEqualization(cv::cuda::GpuMat& input) {
    LOG(DEBUG, "Applying adaptive histogram equalization (CLAHE)");
    CudaTimer timer;

    int tileSize = CLAHE_TILE_SIZE;
    int tilesPerRow = (input.cols + tileSize - 1) / tileSize;
    int tilesPerCol = (input.rows + tileSize - 1) / tileSize;
    int numTiles = tilesPerRow * tilesPerCol;

    unsigned int* d_localHistograms = (unsigned int*)safeCudaMalloc(numTiles * HISTOGRAM_BINS * sizeof(unsigned int));
    CUDA_SAFE_CALL(cudaMemset(d_localHistograms, 0, numTiles * HISTOGRAM_BINS * sizeof(unsigned int)));

    dim3 blockHist(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridHist(tilesPerRow, tilesPerCol);

    timer.Start();

    computeLocalHistogramsKernel<<<gridHist, blockHist>>>(input.ptr<unsigned char>(), input.cols, input.rows, tileSize, d_localHistograms);
    CHECK_CUDA_ERROR(cudaGetLastError());

    applyClipLimitKernel<<<(numTiles + 255) / 256, 256>>>(d_localHistograms, numTiles, tileSize, CLAHE_CLIP_LIMIT);
    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 blockClahe(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridClahe((input.cols + blockClahe.x - 1) / blockClahe.x, (input.rows + blockClahe.y - 1) / blockClahe.y);

    claheKernel<<<gridClahe, blockClahe>>>(input.ptr<unsigned char>(), input.cols, input.rows, tileSize, d_localHistograms);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer.Stop();
    LOG(DEBUG, "CLAHE time: " << timer.ElapsedMilliseconds() << " ms");

    safeCudaFree(d_localHistograms);
}

// Main preprocessing function
void preprocessImage(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {
    LOG(INFO, "Starting image preprocessing");
    CudaTimer totalTimer;
    totalTimer.Start();

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    cv::cuda::GpuMat temp1, temp2;
    temp1.create(input.size(), CV_8UC1);
    temp2.create(input.size(), CV_8UC1);

    #if USE_GRAYSCALE_CONVERSION
        LOG(DEBUG, "Applying grayscale conversion");
        CudaTimer timer;
        timer.Start();

        grayscaleConversionKernel<<<grid, block>>>(input.ptr<uchar4>(), temp1.ptr<unsigned char>(), input.cols, input.rows);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        timer.Stop();
        LOG(DEBUG, "Grayscale conversion time: " << timer.ElapsedMilliseconds() << " ms");
    #endif

    #if USE_DENOISING
        #if DENOISING_METHOD == 0
            applyGaussianBlur(temp1, temp2);
        #else
            applyMedianFilter(temp1, temp2);
        #endif

        temp1.swap(temp2);
    #endif

    #if USE_CONTRAST_ENHANCEMENT
        #if CONTRAST_METHOD == 0
            applyHistogramEqualization(temp1);
        #else
            applyAdaptiveHistogramEqualization(temp1);
        #endif
    #endif

    temp1.copyTo(output);

    totalTimer.Stop();
    LOG(INFO, "Image preprocessing completed. Total time: " << totalTimer.ElapsedMilliseconds() << " ms");
}
