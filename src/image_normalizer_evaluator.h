#ifndef SIMAGE_IMAGE_NORMALIZER_EVALUATOR_H
#define SIMAGE_IMAGE_NORMALIZER_EVALUATOR_H

#include <unitypes.h>
#include <opencv2/core/mat.hpp>

namespace simage::image_normalizer::evaluator {

// Find optimal parameters for `normalize()`
void test_normalize_parameters(const std::vector<std::vector<cv::Mat>> image_groups, uint32_t num_iterations);

}

#endif //SIMAGE_IMAGE_NORMALIZER_EVALUATOR_H
