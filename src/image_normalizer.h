#ifndef SIMAGE_IMAGE_NORMALIZER_H
#define SIMAGE_IMAGE_NORMALIZER_H

#include <opencv2/core/mat.hpp>

namespace simage::image_normalizer {

// Get edges from an image `input`, and put in `output`
void normalize(const cv::Mat &input, cv::Mat &output);

// Get the pixel-wise difference between two images
double get_difference_score(const cv::Mat &a, const cv::Mat &b);

}

#endif //SIMAGE_IMAGE_NORMALIZER_H
