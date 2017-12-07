#include <iostream>
#include <opencv2/core.hpp>
#include "util.h"

namespace simage::util {

std::vector<cv::Mat> for_display(const std::vector<cv::Mat> &inputs) {
    // Get min and max pixel values of all images
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    for (const auto &i : inputs) {
        double i_min, i_max;
        cv::minMaxLoc(i, &i_max, &i_min);

        if (i_min < min) {
            min = i_min;
        }

        if (i_max > max) {
            max = i_max;
        }
    }

    // Put pixels between 0-255 in 8bit channels
    std::vector<cv::Mat> outputs;
    for (const auto &input : inputs) {
        cv::Mat output = input;
        output -= min;
        output /= (max - min) / 256;

        output.convertTo(output, CV_8UC(input.channels()));

        outputs.push_back(output);
    }

    return outputs;
}

void print_distribution(const cv::Mat &image) {
    cv::Mat mean, std_dev;
    double min, max;
    cv::meanStdDev(image, mean, std_dev);
    cv::minMaxLoc(image, &min, &max);
    std::cout <<
              "Mean: " << mean.at<double>(0, 0) << "; " <<
              "Std: " << std_dev.at<double>(0, 0) << "; " <<
              "Min: " << min << "; " <<
              "Max: " << max << std::endl;
}

}
