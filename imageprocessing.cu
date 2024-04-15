#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "segmentation.h"
#include <iostream>

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

__global__ void binarizationKernel(unsigned char* grayImg, unsigned char* binaryImg, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = y * width + x;
        unsigned char pixelValue = grayImg[offset];
        binaryImg[offset] = (pixelValue >= threshold) ? 255 : 0;
    }
}

// Apply adaptive thresholding to the grayscale image
void applyAdaptiveThreshold(cv::Mat& grayImg, cv::Mat& binaryImg) {
    // Adaptive thresholding
    int blockSize = 11; // Size of a pixel neighborhood used to calculate a threshold value
    int C = 2;          // Constant subtracted from the mean or weighted mean

    cv::adaptiveThreshold(grayImg, binaryImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
}

void grayscaleToBinary(cv::Mat& grayImg, cv::Mat& binaryImg, unsigned char threshold) {
    unsigned char *d_grayImg, *d_binaryImg;

    int sizeGray = grayImg.step * grayImg.rows;
    int sizeBinary = binaryImg.step * binaryImg.rows;

    cudaMalloc<unsigned char>(&d_grayImg, sizeGray);
    cudaMalloc<unsigned char>(&d_binaryImg, sizeBinary);

    cudaMemcpy(d_grayImg, grayImg.ptr(), sizeGray, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((grayImg.cols + blockSize.x - 1) / blockSize.x,
                  (grayImg.rows + blockSize.y - 1) / blockSize.y);

    binarizationKernel<<<gridSize, blockSize>>>(d_grayImg, d_binaryImg, grayImg.cols, grayImg.rows, threshold);

    cudaMemcpy(binaryImg.ptr(), d_binaryImg, sizeBinary, cudaMemcpyDeviceToHost);

    cudaFree(d_grayImg);
    cudaFree(d_binaryImg);
}

__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int filterSize = 3; // Example size of the median filter window
    int edge = filterSize / 2;

    // Prepare an array to hold the values of the window
    unsigned char window[9]; // Adjust size if filter size changes

    int count = 0;
    for (int fx = -edge; fx <= edge; fx++) {
        for (int fy = -edge; fy <= edge; fy++) {
            int ix = x + fx;
            int iy = y + fy;

            // Check bounds
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                window[count] = input[iy * width + ix];
                count++;
            }
        }
    }

    // Sort to find median
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (window[i] > window[j]) {
                // Swap
                unsigned char temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }

    // Set the median value
    output[y * width + x] = window[count / 2];
}

void applyMedianFilter(cv::Mat& input, cv::Mat& output) {
    unsigned char *d_input, *d_output;

    int size = input.step * input.rows;

    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_output, size);

    cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows);

    cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    cv::Mat h_colorImg = cv::imread("sample.png", cv::IMREAD_COLOR);
    if (h_colorImg.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat h_grayImg(h_colorImg.rows, h_colorImg.cols, CV_8UC1);

    colorToGrayscale(h_colorImg, h_grayImg);

    // Pre-processing: Apply Median Filter on Grayscale Image
    cv::Mat h_denoisedImg(h_grayImg.rows, h_grayImg.cols, CV_8UC1);
    applyMedianFilter(h_grayImg, h_denoisedImg);

    // // Binarize the denoised image
    // unsigned char threshold = 128;  // later to be adjusted with adaptive threshold
    // cv::Mat h_binaryImg(h_denoisedImg.rows, h_grayImg.cols, CV_8UC1);
    // grayscaleToBinary(h_grayImg, h_binaryImg, threshold);

    // Apply adaptive thresholding
    cv::Mat h_binaryImg;
    applyAdaptiveThreshold(h_grayImg, h_binaryImg);

    // cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3)); // Adjust size as needed
    // cv::Mat dilatedImg;
    // cv::dilate(h_binaryImg, dilatedImg, morphKernel, cv::Point(-1, -1), 2); // Dilation

    // cv::Mat erodedImg;
    // cv::erode(dilatedImg, erodedImg, morphKernel, cv::Point(-1, -1), 5); // Erosion to restore size

    //line segmentation
    std::vector<cv::Rect> lines = segmentLines(h_binaryImg);

    // Compute horizontal projection
    cv::Mat horizontal_projection;
    cv::reduce(h_binaryImg, horizontal_projection, 1, cv::REDUCE_SUM, CV_32S);

    for (const auto& rect : lines) {
    std::cout << "Line found at x: " << rect.x << ", y: " << rect.y 
              << ", width: " << rect.width << ", height: " << rect.height << std::endl;
    }

    cv::Mat displayImg = h_binaryImg.clone();  // Create a copy of the binary image for displaying
    cv::cvtColor(displayImg, displayImg, cv::COLOR_GRAY2BGR);  // Convert to BGR for coloring

    for (const auto& rect : lines) {
        cv::rectangle(displayImg, rect, cv::Scalar(0, 255, 0), 2);  // Draw rectangle in green with thickness 2
    }

    cv::imshow("Original", h_colorImg);
    cv::imshow("Grayscale", h_grayImg);
    cv::imshow("Denoised", h_denoisedImg);
    cv::imshow("Binary", h_binaryImg);
    // cv::imshow("Dilated", dilatedImg);
    // cv::imshow("Eroded", erodedImg);
    cv::imshow("Detected Lines", displayImg);
    visualizeHorizontalProjection(h_binaryImg, horizontal_projection);
    
    cv::waitKey(0);

    return 0;
}

