#pragma once

// Image input and dimensions
#define INPUT_IMAGE_PATH "sample.png"        // Path to the input image
#define MIN_IMAGE_WIDTH 640                  // Minimum acceptable image width
#define MIN_IMAGE_HEIGHT 480                 // Minimum acceptable image height

// Preprocessing method selection
#define USE_GRAYSCALE_CONVERSION 1           // Enable (1) or disable (0) grayscale conversion
#define USE_DENOISING 1                      // Enable (1) or disable (0) denoising
#define USE_CONTRAST_ENHANCEMENT 1           // Enable (1) or disable (0) contrast enhancement

// Grayscale conversion methods
#define GRAYSCALE_METHOD 1                   // Grayscale conversion method: 0 (Average), 1 (Weighted), 2 (Desaturation)

// Denoising methods
#define DENOISING_METHOD 0                   // Denoising method: 0 (Gaussian Blur), 1 (Median Filter)
#define GAUSSIAN_KERNEL_SIZE 5               // Kernel size for Gaussian blur (must be odd)
#define MAX_KERNEL_SIZE 25                   // Maximum kernel size (should be odd and >= GAUSSIAN_KERNEL_SIZE)
#define GAUSSIAN_SIGMA 1.0f                  // Sigma value for Gaussian blur
#define MEDIAN_KERNEL_SIZE 3                 // Kernel size for Median filter (must be odd)

// Contrast enhancement methods
#define CONTRAST_METHOD 0                    // Contrast enhancement method: 0 (Histogram Equalization), 1 (Adaptive Histogram Equalization)
#define HISTOGRAM_BINS 256                   // Number of bins for histogram equalization
#define CLAHE_CLIP_LIMIT 4.0                 // Clip limit for CLAHE
#define CLAHE_TILE_SIZE 8                    // Tile size for CLAHE

// Image quality assessment thresholds
#define BLUR_DETECTION_VARIANCE_THRESHOLD 10.0   // Variance threshold for blur detection
#define CONTRAST_DETECTION_STDDEV_THRESHOLD 50.0 // Standard deviation threshold for contrast detection
#define BRIGHTNESS_DETECTION_MEAN_LOWER_THRESHOLD 50.0 // Lower threshold for brightness detection
#define BRIGHTNESS_DETECTION_MEAN_UPPER_THRESHOLD 300.0 // Upper threshold for brightness detection

// CUDA kernel configuration
#define BLOCK_SIZE_X 32                     // Block size in X dimension for CUDA kernels
#define BLOCK_SIZE_Y 32                     // Block size in Y dimension for CUDA kernels

// Debug mode
#define DEBUG_MODE 1                        // Enable (1) or disable (0) debug mode

// Ensure kernel sizes are valid (must be odd and within specified range)
#if (GAUSSIAN_KERNEL_SIZE % 2 == 0) || (GAUSSIAN_KERNEL_SIZE > MAX_KERNEL_SIZE) || (GAUSSIAN_KERNEL_SIZE < 1)
#error "GAUSSIAN_KERNEL_SIZE must be an odd number between 1 and MAX_KERNEL_SIZE"
#endif

#if (MEDIAN_KERNEL_SIZE % 2 == 0) || (MEDIAN_KERNEL_SIZE > MAX_KERNEL_SIZE) || (MEDIAN_KERNEL_SIZE < 1)
#error "MEDIAN_KERNEL_SIZE must be an odd number between 1 and MAX_KERNEL_SIZE"
#endif
