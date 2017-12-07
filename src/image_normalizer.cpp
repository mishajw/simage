#include <opencv2/opencv.hpp>
#include "image_normalizer.h"

namespace simage::image_normalizer {

// Normalize the image to have a mean 0 and standard deviation 1
void simple_normalize(const cv::Mat &input, cv::Mat &output);

void normalize(const cv::Mat &input, cv::Mat &output) {
    assert(&output != &input);

    // Convert to gray scale
    cv::Mat input_gray_scale;
    cv::cvtColor(input, input_gray_scale, CV_RGB2GRAY);

    // Normalize input
    cv::Mat input_normalized;
    simple_normalize(input_gray_scale, input_normalized);

    // Blur the image
    cv::Mat input_blurred;
    cv::GaussianBlur(input_normalized, input_blurred, cv::Size(5, 5), 0, 0);

    // Apply edge detection filter
    cv::Laplacian(input_blurred, output, CV_32F, 3);
}

double get_difference_score(const cv::Mat &a, const cv::Mat &b) {
    return cv::mean(cv::abs(a - b))[0];
}

void simple_normalize(const cv::Mat &input, cv::Mat &output) {
    cv::Mat mean, std_dev;
    cv::meanStdDev(input, mean, std_dev);
    input.convertTo(output, CV_32FC(input.channels()));
    output -= mean;
    output /= std_dev;
}

}
