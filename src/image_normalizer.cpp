#include <opencv2/opencv.hpp>
#include <random>
#include "image_normalizer.h"

namespace simage::image_normalizer {

// Normalize the image to have a mean 0 and standard deviation 1
void simple_normalize(const cv::Mat &input, cv::Mat &output);

std::function<ImageNormalizeParameters()> ImageNormalizeParameters::get_random_generator() {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> random_gaussian_distribution(1, 10);
    std::uniform_int_distribution<> random_laplacian_distribution(1, 5);

    return [=]() mutable -> ImageNormalizeParameters {
        int random_gaussian_blur_size = random_gaussian_distribution(generator) * 2 + 1;
        int random_laplacian_filter_size = random_laplacian_distribution(generator) * 2 + 1;

        return ImageNormalizeParameters(random_gaussian_blur_size, random_laplacian_filter_size);
    };
}

std::ostream& operator<< (std::ostream &stream, const ImageNormalizeParameters &parameters) {
    stream <<
            "ImageNormalizeParameters(gauss: " <<
            parameters.gaussian_blur_size <<
            ", laplacian: " <<
            parameters.laplacian_filter_size << ")";
}

void normalize(const cv::Mat &input, cv::Mat &output, const ImageNormalizeParameters &parameters) {
    assert(&output != &input);

    // Convert to gray scale
    cv::Mat input_gray_scale;
    cv::cvtColor(input, input_gray_scale, CV_RGB2GRAY);

    // Normalize input
    cv::Mat input_normalized;
    simple_normalize(input_gray_scale, input_normalized);

    // Blur the image
    cv::Mat input_blurred;
    cv::GaussianBlur(
            input_normalized,
            input_blurred,
            cv::Size(parameters.gaussian_blur_size, parameters.gaussian_blur_size),
            0,
            0);

    // Apply edge detection filter
    cv::Laplacian(input_blurred, output, CV_32F, parameters.laplacian_filter_size);
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
