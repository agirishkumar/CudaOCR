#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "segmentation.h"
#include <iostream>
#include <cmath> 

// CUDA kernel for converting color image to grayscale
__global__ void colorToGrayscaleKernel(unsigned char* colorImg, unsigned char* grayImg, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int grayOffset = y * width + x;
        int colorOffset = grayOffset * 3;

        unsigned char r = colorImg[colorOffset + 2];
        unsigned char g = colorImg[colorOffset + 1];
        unsigned char b = colorImg[colorOffset];

        grayImg[grayOffset] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Wrapper function for calling the CUDA grayscale conversion kernel
void colorToGrayscale(cv::Mat& colorImg, cv::Mat& grayImg) {
    unsigned char *d_colorImg, *d_grayImg;

    int sizeColor = colorImg.step * colorImg.rows;
    int sizeGray = grayImg.step * grayImg.rows;

    cudaMalloc<unsigned char>(&d_colorImg, sizeColor);
    cudaMalloc<unsigned char>(&d_grayImg, sizeGray);

    cudaMemcpy(d_colorImg, colorImg.ptr(), sizeColor, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((colorImg.cols + blockSize.x - 1) / blockSize.x,
                  (colorImg.rows + blockSize.y - 1) / blockSize.y);

    colorToGrayscaleKernel<<<gridSize, blockSize>>>(d_colorImg, d_grayImg, colorImg.cols, colorImg.rows);

    cudaMemcpy(grayImg.ptr(), d_grayImg, sizeGray, cudaMemcpyDeviceToHost);

    cudaFree(d_colorImg);
    cudaFree(d_grayImg);
}

void setGaussianKernel(float *kernel, int radius, float sigma) {
    const int size = 2 * radius + 1;
    float sum = 0.0;
    float s = 2.0 * sigma * sigma;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float r = sqrtf(i * i + j * j);
            kernel[(i + radius) * size + (j + radius)] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += kernel[(i + radius) * size + (j + radius)];
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i * size + j] /= sum;
        }
    }
}


__global__ void gaussianBlurKernel(float* d_kernel, unsigned char* in, unsigned char* out, int width, int height, int radius) {
    // Calculate our pixel's location
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;

    // Thread block's starting index
    int startCol = blockIdx.x * blockDim.x;
    int startRow = blockIdx.y * blockDim.y;

    // Shared memory size, including the halo
    extern __shared__ unsigned char shared[];

    // Load pixels into shared memory including the halo
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            int sharedIdx = (threadIdx.y + radius + i) * (blockDim.x + 2 * radius) + (threadIdx.x + radius + j);
            int imgIdx = (row + i) * width + (col + j);
            if (imgIdx >= 0 && imgIdx < width * height && 
                (col + j) >= 0 && (col + j) < width && 
                (row + i) >= 0 && (row + i) < height) {
                shared[sharedIdx] = in[imgIdx];
            }
        }
    }
    __syncthreads();

    // Apply the kernel to the pixel
    float sum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int sharedIdx = (threadIdx.y + radius + y) * (blockDim.x + 2 * radius) + (threadIdx.x + radius + x);
            sum += shared[sharedIdx] * d_kernel[(y + radius) * (2 * radius + 1) + (x + radius)];
        }
    }

    // Write our result
    if (col < width && row < height) {
        out[idx] = static_cast<unsigned char>(sum);
    }
}



void applyGaussianBlur(cv::Mat& input, cv::Mat& output, int radius, float sigma) {
    int size = 2 * radius + 1;
    float *h_kernel = new float[size * size];
    float *d_kernel;

    setGaussianKernel(h_kernel, radius, sigma);
    cudaMalloc(&d_kernel, sizeof(float) * size * size);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, input.total());
    cudaMalloc(&d_output, output.total());
    cudaMemcpy(d_input, input.data, input.total(), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);
 

    // Calculate shared memory size including halo
    size_t sharedMemSize = (blockSize.x + 2 * radius) * (blockSize.y + 2 * radius) * sizeof(unsigned char);

    // Launch the kernel
    gaussianBlurKernel<<<gridSize, blockSize, sharedMemSize>>>(d_kernel, d_input, d_output, input.cols, input.rows, radius);

    cudaMemcpy(output.data, d_output, output.total(), cudaMemcpyDeviceToHost);

    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_kernel;
}


__global__ void histogramKernel(unsigned char* image, int width, int height, unsigned int* histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        atomicAdd(&histogram[image[index]], 1);
    }
}


__global__ void otsuThresholdKernel(unsigned int* histogram, int totalPixels, unsigned char* threshold) {
    __shared__ float meanB[256];
    __shared__ float weightB[256];
    __shared__ float betweenVar[256];

    int i = threadIdx.x;

    // Compute cumulative sums and weights
    if (i == 0) {
        meanB[i] = 0;
        weightB[i] = histogram[0];
    } else {
        weightB[i] = weightB[i - 1] + histogram[i];
        meanB[i] = meanB[i - 1] + i * histogram[i];
    }
    __syncthreads();

    // Compute the between class variance for each threshold
    float meanTotal = meanB[255];
    for (int t = 0; t < 256; t++) {
        float weightF = totalPixels - weightB[t];
        if (weightB[t] != 0 && weightF != 0) {
            float meanF = (meanTotal - meanB[t]) / weightF;
            float meanT = meanB[t] / weightB[t];
            betweenVar[t] = weightB[t] * weightF * (meanT - meanF) * (meanT - meanF);
        } else {
            betweenVar[t] = 0;
        }
    }
    __syncthreads();

    // Find the maximum between class variance
    if (i == 0) {
        float maxVal = 0;
        int maxIdx = 0;
        for (int t = 0; t < 256; t++) {
            if (betweenVar[t] > maxVal) {
                maxVal = betweenVar[t];
                maxIdx = t;
            }
        }
        *threshold = maxIdx;
    }
}




// CUDA kernel for applying a fixed threshold to create a binary image
// __global__ void binarizationKernel(unsigned char* grayImg, unsigned char* binaryImg, int width, int height, unsigned char threshold) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         int offset = y * width + x;
//         unsigned char pixelValue = grayImg[offset];
//         binaryImg[offset] = (pixelValue >= threshold) ? 255 : 0;
//     }
// }

// // Apply adaptive thresholding to the grayscale image
// void applyAdaptiveThreshold(cv::Mat& grayImg, cv::Mat& binaryImg) {
//     int blockSize = 7; // Size of a pixel neighborhood used to calculate a threshold value
//     int C = 2;          // Constant subtracted from the mean or weighted mean
//     cv::adaptiveThreshold(grayImg, binaryImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
// }

// void grayscaleToBinary(cv::Mat& grayImg, cv::Mat& binaryImg, unsigned char threshold) {
//     unsigned char *d_grayImg, *d_binaryImg;

//     int sizeGray = grayImg.step * grayImg.rows;
//     int sizeBinary = binaryImg.step * binaryImg.rows;

//     cudaMalloc<unsigned char>(&d_grayImg, sizeGray);
//     cudaMalloc<unsigned char>(&d_binaryImg, sizeBinary);

//     cudaMemcpy(d_grayImg, grayImg.ptr(), sizeGray, cudaMemcpyHostToDevice);

//     dim3 blockSize(16, 16);
//     dim3 gridSize((grayImg.cols + blockSize.x - 1) / blockSize.x,
//                   (grayImg.rows + blockSize.y - 1) / blockSize.y);

//     binarizationKernel<<<gridSize, blockSize>>>(d_grayImg, d_binaryImg, grayImg.cols, grayImg.rows, threshold);

//     cudaMemcpy(binaryImg.ptr(), d_binaryImg, sizeBinary, cudaMemcpyDeviceToHost);

//     cudaFree(d_grayImg);
//     cudaFree(d_binaryImg);
// }

// __global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= width || y >= height) return;

//     int filterSize = 3; // Example size of the median filter window
//     int edge = filterSize / 2;

//     // Prepare an array to hold the values of the window
//     unsigned char window[9]; // Adjust size if filter size changes

//     int count = 0;
//     for (int fx = -edge; fx <= edge; fx++) {
//         for (int fy = -edge; fy <= edge; fy++) {
//             int ix = x + fx;
//             int iy = y + fy;

//             // Check bounds
//             if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
//                 window[count] = input[iy * width + ix];
//                 count++;
//             }
//         }
//     }

//     // Sort to find median
//     for (int i = 0; i < count; i++) {
//         for (int j = i + 1; j < count; j++) {
//             if (window[i] > window[j]) {
//                 // Swap
//                 unsigned char temp = window[i];
//                 window[i] = window[j];
//                 window[j] = temp;
//             }
//         }
//     }

//     // Set the median value
//     output[y * width + x] = window[count / 2];
// }

// void applyMedianFilter(cv::Mat& input, cv::Mat& output) {
//     unsigned char *d_input, *d_output;

//     int size = input.step * input.rows;

//     cudaMalloc<unsigned char>(&d_input, size);
//     cudaMalloc<unsigned char>(&d_output, size);

//     cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);

//     dim3 blockSize(16, 16);
//     dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
//                   (input.rows + blockSize.y - 1) / blockSize.y);

//     medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows);

//     cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

//     cudaFree(d_input);
//     cudaFree(d_output);
// }


int main() {
    cv::Mat h_colorImg = cv::imread("sample.png", cv::IMREAD_COLOR);
    if (h_colorImg.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat h_grayImg(h_colorImg.rows, h_colorImg.cols, CV_8UC1);

    colorToGrayscale(h_colorImg, h_grayImg);

    // Set kernel radius and sigma for Gaussian blur
    int radius = 1; // Corresponds to a 3x3 kernel
    float sigma = 0.3 * ((radius * 2 + 1 - 1) * 0.5 - 1) + 0.8; // Using OpenCV's formula to calculate sigma from kernel size

    cv::Mat h_blurredImg(h_grayImg.size(), h_grayImg.type());
    applyGaussianBlur(h_grayImg, h_blurredImg, radius, sigma);

    // Prepare GPU buffers for histogram and thresholding
    unsigned int h_histogram[256] = {0};
    unsigned int* d_histogram;
    cudaMalloc(&d_histogram, sizeof(unsigned int) * 256);
    cudaMemset(d_histogram, 0, sizeof(unsigned int) * 256);

    // Calculate histogram
    dim3 histBlockSize(16, 16);
    dim3 histGridSize((h_blurredImg.cols + histBlockSize.x - 1) / histBlockSize.x,
                      (h_blurredImg.rows + histBlockSize.y - 1) / histBlockSize.y);
    histogramKernel<<<histGridSize, histBlockSize>>>(h_blurredImg.ptr<unsigned char>(), h_blurredImg.cols, h_blurredImg.rows, d_histogram);

    // Copy histogram back to host and inspect it
    cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost);
    // [Optional] Add code here to inspect the histogram.

    // Otsu's thresholding
    unsigned char h_threshold;
    unsigned char* d_threshold;
    cudaMalloc(&d_threshold, sizeof(unsigned char));
    otsuThresholdKernel<<<1, 256>>>(d_histogram, h_blurredImg.rows * h_blurredImg.cols, d_threshold);
    cudaMemcpy(&h_threshold, d_threshold, sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Apply the threshold
    cv::Mat h_binaryImg(h_blurredImg.size(), CV_8UC1);
    cv::threshold(h_blurredImg, h_binaryImg, h_threshold, 255, cv::THRESH_BINARY);

    // Display results
    cv::imshow("Original", h_colorImg);
    cv::imshow("Blurred", h_blurredImg);
    cv::imshow("Binary", h_binaryImg);

    cv::waitKey(0);

    // Cleanup
    cudaFree(d_histogram);
    cudaFree(d_threshold);
    return 0;
}
