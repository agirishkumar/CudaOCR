#include "segmentation.h"

void detectTextLines(cv::Mat& image) {
    // // Preprocess the image: convert to grayscale
    // cv::Mat gray;
    // cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // // Apply Gaussian blur
    // cv::Mat blur;
    // cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    
    // // Apply thresholding: convert grayscale to binary image
    // cv::Mat bw;
    // cv::threshold(blur, bw, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // Define kernel size for morphological operations
    cv::Size kernelSize(15, 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernelSize);

    // Close gaps between lines - Morphological operation
    cv::Mat closed;
    cv::morphologyEx(image, closed, cv::MORPH_CLOSE, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter and sort contours
    std::vector<cv::Rect> boundRects;
    for (const auto& contour : contours) {
        cv::Rect boundRect = cv::boundingRect(contour);
        if (boundRect.width / static_cast<double>(boundRect.height) >= 3.0) {
            boundRects.push_back(boundRect);
        }
    }
    
    // Sort the bounding rectangles by their y-coordinate
    std::sort(boundRects.begin(), boundRects.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.y < b.y;
    });
    
    // Draw bounding rectangles and recognize text using Tesseract
    for (const auto& boundRect : boundRects) {
        // Expand the bounding box slightly to account for padding
        cv::rectangle(image, boundRect, cv::Scalar(0, 255, 0), 2);
        
        // Crop the image to each bounding box (line of text)
        // Note: OCR part is omitted as it requires Tesseract's C++ API
        // cv::Mat lineImage = image(boundRect);
        // std::string lineText = ... // OCR processing goes here
        // std::cout << lineText << std::endl;
    }
    
    cv::imshow("Detected Lines", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    cv::imwrite("opencv_detect_text_lines.jpg", image);
}
