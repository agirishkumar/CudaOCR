#include "segmentation.h"

std::vector<cv::Rect> segmentLines(const cv::Mat& binaryImg) {
    // std::vector<cv::Rect> lines;
    // std::vector<std::vector<cv::Point>> contours;
    // cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // for (size_t i = 0; i < contours.size(); i++) {
    //     cv::Rect bounding = cv::boundingRect(contours[i]);
    //     if (bounding.height > 10) { // Filter out too small contours that likely aren't lines
    //         lines.push_back(bounding);
    //     }
    // }

    // return lines;

    // horizontal projection

    std::vector<cv::Rect> lineRects;
    cv::Mat horizontal_projection;
    cv::reduce(binaryImg, horizontal_projection, 1, cv::REDUCE_AVG, CV_32S);

    bool inLine = false;
    int lineStart = 0;
    for (int y = 0; y < horizontal_projection.rows; y++) {
        if (horizontal_projection.at<int>(y, 0) > 0) {
            if (!inLine) {
                lineStart = y;
                inLine = true;
            }
        } else {
            if (inLine) {
                lineRects.push_back(cv::Rect(0, lineStart, binaryImg.cols, y - lineStart));
                inLine = false;
            }
        }
    }
    if (inLine) {
        lineRects.push_back(cv::Rect(0, lineStart, binaryImg.cols, horizontal_projection.rows - lineStart));
    }
    return lineRects;
}


// Function to visualize the horizontal projection on the binary image
void visualizeHorizontalProjection(const cv::Mat& binaryImg, const cv::Mat& projection) {
    int maxVal = 0;
    for (int i = 0; i < projection.rows; i++) {
        maxVal = std::max(maxVal, projection.at<int>(i, 0));
    }

    cv::Mat projectionVis = cv::Mat::zeros(cv::Size(binaryImg.cols, binaryImg.rows), CV_8UC3);
    for (int y = 0; y < binaryImg.rows; y++) {
        int value = static_cast<int>((static_cast<double>(projection.at<int>(y, 0)) / maxVal) * binaryImg.cols);
        cv::line(projectionVis, cv::Point(0, y), cv::Point(value, y), cv::Scalar(255, 255, 255), 1);
    }
    cv::imshow("Horizontal Projection", projectionVis);
    cv::waitKey(0);
}