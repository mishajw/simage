#ifndef SIMAGE_UTIL_H
#define SIMAGE_UTIL_H

#include <opencv2/core/mat.hpp>

namespace simage::util {

// Format images for display
std::vector<cv::Mat> for_display(const std::vector<cv::Mat> &inputs);

// Print the distribution of pixels in an image
void print_distribution(const cv::Mat &image);

}

#endif //SIMAGE_UTIL_H
