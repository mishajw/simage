#ifndef SIMAGE_IMAGE_NORMALIZER_H
#define SIMAGE_IMAGE_NORMALIZER_H

#include <functional>
#include <opencv2/core/mat.hpp>

namespace simage::image_normalizer {

struct ImageNormalizeParameters {
    ImageNormalizeParameters(const int gaussian_blur_size, const int laplacian_filter_size) :
            gaussian_blur_size(gaussian_blur_size), laplacian_filter_size(laplacian_filter_size) {}

    ImageNormalizeParameters() :
            gaussian_blur_size(0), laplacian_filter_size(0) {}

    const int gaussian_blur_size;
    const int laplacian_filter_size;

    static std::function<ImageNormalizeParameters()> get_random_generator();
};

std::ostream& operator<< (std::ostream &stream, const ImageNormalizeParameters &parameters);

// Get edges from an image `input`, and put in `output`
void normalize(const cv::Mat &input, cv::Mat &output, const ImageNormalizeParameters &parameters);

// Get the pixel-wise difference between two images
double get_difference_score(const cv::Mat &a, const cv::Mat &b);

}

#endif //SIMAGE_IMAGE_NORMALIZER_H
