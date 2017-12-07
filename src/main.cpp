#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_normalizer.h"

int main(int argc, char **argv) {
    if (argc < 3 || argc % 2 != 1) {
        std::cerr << "Usage: " << argv[0] << " <image path pair>..." << std::endl;
        exit(1);
    }

    std::vector<std::vector<cv::Mat>> image_groups;

    for (size_t i = 1; i < argc; i += 2) {
        std::string image_path1 = argv[i];
        std::string image_path2 = argv[i + 1];

        cv::Mat image1 = cv::imread(image_path1, CV_LOAD_IMAGE_COLOR);
        cv::Mat image2 = cv::imread(image_path2, CV_LOAD_IMAGE_COLOR);
        assert(!image1.empty());
        assert(!image2.empty());

        image_groups.emplace_back(std::vector<cv::Mat>{image1, image2});
    }

    simage::image_normalizer::test_normalize_parameters(image_groups, 100);
}
