#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image 1> <image 2>" << std::endl;
        exit(1);
    }

    std::string image_path1 = argv[1];
    std::string image_path2 = argv[2];

    cv::Mat image1 = cv::imread(image_path1, CV_LOAD_IMAGE_COLOR);
    cv::Mat image2 = cv::imread(image_path2, CV_LOAD_IMAGE_COLOR);

    cv::imshow("Image 1", image1);
    cv::waitKey(0);

    cv::imshow("Image 2", image2);
    cv::waitKey(0);
}
