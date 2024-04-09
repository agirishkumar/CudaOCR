#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>


__global__ void convertToGrayscaleKernel(uchar3* input, uchar* output, int width, int height){
 
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= width || y >= height) return;
  int idx = y * width + x;
  uchar3 rgb = input[idx];
  output[idx] = (uchar)(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
}


int main() {
    // Load the image
    cv::Mat img = cv::imread("sampledocimg.png");
    if (img.empty()) {
        std::cerr << "Error: Image not found.\n";
        return -1;
    }

    // Ensure the image is in the expected format
    cv::Mat img_rgb;
    if (img.channels() == 4) cv::cvtColor(img, img_rgb, cv::COLOR_BGRA2BGR);
    else img_rgb = img.clone();

    cv::cuda::GpuMat d_img_rgb, d_img_gray;
    d_img_rgb.upload(img_rgb);

    // Allocate memory for output on device
    d_img_gray.create(d_img_rgb.size(), CV_8UC1);

    // Calculate grid and block sizes
    dim3 blockSize(16, 16); 
    dim3 gridSize((img_rgb.cols + blockSize.x - 1) / blockSize.x,
                  (img_rgb.rows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    convertToGrayscaleKernel<<<gridSize, blockSize>>>(d_img_rgb.ptr<uchar3>(), d_img_gray.ptr<uchar>(), img_rgb.cols, img_rgb.rows);
    cudaDeviceSynchronize();

    // Download the result back to host
    cv::Mat result;
    d_img_gray.download(result);

    // Save the output image
    cv::imwrite("sample_gray.png", result);

    return 0;

}