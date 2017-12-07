#include <opencv2/opencv.hpp>
#include <random>
#include "image_normalizer.h"

namespace simage::image_normalizer {

// Normalize the image to have a mean 0 and standard deviation 1
void simple_normalize(const cv::Mat &input, cv::Mat &output);

// Evaluate the effectiveness of some parameters on image groups
void evaluate_parameters(std::vector<std::vector<cv::Mat>> image_groups, const ImageNormalizeParameters &parameters);

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

void normalize(const cv::Mat &input, cv::Mat &output, const ImageNormalizeParameters parameters) {
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

void test_normalize_parameters(const std::vector<std::vector<cv::Mat>> image_groups, uint32_t num_iterations) {
    const auto random_generator = ImageNormalizeParameters::get_random_generator();

    for (uint32_t i = 0; i < num_iterations; i++) {
        const auto random_parameters = random_generator();

        evaluate_parameters(image_groups, random_parameters);
    }
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

void evaluate_parameters(
        std::vector<std::vector<cv::Mat>> image_groups, const ImageNormalizeParameters &parameters) {
    // Create edge images
    std::vector<std::vector<cv::Mat>> edges_groups;
    for (const auto &image_group : image_groups) {
        std::vector<cv::Mat> edges_group;
        for (const auto &image : image_group) {
            cv::Mat edges;
            normalize(image, edges, parameters);
            edges_group.push_back(edges);
        }
        edges_groups.push_back(edges_group);
    }

    double intra_group_difference_total = 0;
    int intra_group_difference_count = 0;
    for (const auto &edges_group : edges_groups) {
        for (auto iter1 = edges_group.begin(); iter1 != edges_group.end(); iter1++) {
            for (auto iter2 = iter1 + 1; iter2 != edges_group.end(); iter2++) {
                intra_group_difference_total += get_difference_score(*iter1, *iter2);
                intra_group_difference_count++;
            }
        }
    }
    double intra_group_difference_average = intra_group_difference_total / intra_group_difference_count;

    double inter_group_difference_total = 0;
    int inter_group_difference_count = 0;
    for (auto iter1 = edges_groups.begin(); iter1 != edges_groups.end(); iter1++) {
        for (auto iter2 = iter1 + 1; iter2 != edges_groups.end(); iter2++) {
            const auto &group1 = *iter1;
            const auto &group2 = *iter2;

            for (const auto &image1 : group1) {
                for (const auto &image2 : group2) {
                    inter_group_difference_total += get_difference_score(image1, image2);
                    inter_group_difference_count++;
                }
            }
        }
    }
    double inter_group_difference_average = inter_group_difference_total / inter_group_difference_count;

    double cost = pow(intra_group_difference_average, 2) - inter_group_difference_average;

    std::cout <<
        "Cost: " << cost << "; " <<
        "Params: " << parameters << "; " <<
        "Intra-group difference: " << intra_group_difference_average << "; " <<
        "Inter-group difference: " << inter_group_difference_average << std::endl;
}

}
