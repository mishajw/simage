#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_normalizer.h"
#include "util.h"


namespace simage {

void run(const cv::Mat &image1, const cv::Mat &image2) {
    std::cout << "Image 1: "; util::print_distribution(image1);
    std::cout << "Image 2: "; util::print_distribution(image2);

    // Do edge detection
    cv::Mat image_edges1, image_edges2;
    image_normalizer::normalize(image1, image_edges1);
    image_normalizer::normalize(image2, image_edges2);
    std::cout << "Edges 1: "; util::print_distribution(image_edges1);
    std::cout << "Edges 2: "; util::print_distribution(image_edges2);

    // Display the images
    std::vector<cv::Mat> display_images = util::for_display({image_edges1, image_edges2});
    assert(display_images.size() == 2);
    for (const auto &i : display_images) {
        cv::imshow("Edge Image", i);
        cv::waitKey(0);
    }

    // Calculate the difference
    double difference = image_normalizer::get_difference_score(image_edges1, image_edges2);
    std::cout << "Difference: " << difference << std::endl;
}

}


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image 1> <image 2>" << std::endl;
        exit(1);
    }

    std::string image_path1 = argv[1];
    std::string image_path2 = argv[2];

    cv::Mat image1 = cv::imread(image_path1, CV_LOAD_IMAGE_COLOR);
    cv::Mat image2 = cv::imread(image_path2, CV_LOAD_IMAGE_COLOR);

    assert(!image1.empty());
    assert(!image2.empty());

    simage::run(image1, image2);
}

