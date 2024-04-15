#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Rect> segmentLines(const cv::Mat& binaryImg);
void visualizeHorizontalProjection(const cv::Mat& binaryImg, const cv::Mat& projection) ;

#endif // SEGMENTATION_H
