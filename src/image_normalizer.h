#ifndef SIMAGE_IMAGE_NORMALIZER_H
#define SIMAGE_IMAGE_NORMALIZER_H

#include <opencv2/core/mat.hpp>

namespace simage::image_normalizer {

struct ImageNormalizeParameters {
    ImageNormalizeParameters(const int gaussian_blur_size, const int laplacian_filter_size) :
            gaussian_blur_size(gaussian_blur_size), laplacian_filter_size(laplacian_filter_size) {}

    const int gaussian_blur_size;
    const int laplacian_filter_size;

    static std::function<ImageNormalizeParameters()> get_random_generator();
};

// Get edges from an image `input`, and put in `output`
void normalize(const cv::Mat &input, cv::Mat &output, int gaussian_blur_size, int laplacian_filter_size);

// Find optimal parameters for `normalize()`
void test_normalize_parameters(const std::vector<std::vector<cv::Mat>> image_groups, uint32_t num_iterations);

// Get the pixel-wise difference between two images
double get_difference_score(const cv::Mat &a, const cv::Mat &b);

}

#endif //SIMAGE_IMAGE_NORMALIZER_H
