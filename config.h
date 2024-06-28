#pragma once

#define INPUT_IMAGE_PATH "sample.png"
#define GAUSSIAN_BLUR_RADIUS 1
#define GAUSSIAN_BLUR_SIGMA 0.3f
#define BLUR_DETECTION_VARIANCE_THRESHOLD 10.0 // Variance threshold for detecting blur in the image
#define CONTRAST_DETECTION_STDDEV_THRESHOLD 50.0 // Standard deviation threshold for detecting low contrast in the image
#define BRIGHTNESS_DETECTION_MEAN_LOWER_THRESHOLD 50.0 // Lower mean threshold for detecting low brightness in the image
#define BRIGHTNESS_DETECTION_MEAN_UPPER_THRESHOLD 300.0 // Upper mean threshold for detecting high brightness in the image
#define MIN_IMAGE_WIDTH 640
#define MIN_IMAGE_HEIGHT 480

// Other configuration parameters...