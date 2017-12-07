#include <iostream>
#include <opencv2/opencv.hpp>

// Get edges from an image `input`, and put in `output`
void apply_edge_detection(const cv::Mat &input, cv::Mat &output);

// Normalize the image to have a mean 0 and standard deviation 1
void normalize_image(const cv::Mat &input, cv::Mat &output);

// Get the pixel-wise difference between two images
double get_difference_score(const cv::Mat &a, const cv::Mat &b);

// Format images for display
std::vector<cv::Mat> for_display(const std::vector<cv::Mat> &inputs);

// Print the distribution of pixels in an image
void print_distribution(const cv::Mat &image);

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image 1> <image 2>" << std::endl;
        exit(1);
    }

    // Load in images
    std::string image_path1 = argv[1];
    std::string image_path2 = argv[2];
    cv::Mat image1 = cv::imread(image_path1, CV_LOAD_IMAGE_COLOR);
    cv::Mat image2 = cv::imread(image_path2, CV_LOAD_IMAGE_COLOR);
    assert(!image1.empty());
    assert(!image2.empty());
    std::cout << "Image 1: "; print_distribution(image1);
    std::cout << "Image 2: "; print_distribution(image2);

    // Do edge detection
    cv::Mat image_edges1, image_edges2;
    apply_edge_detection(image1, image_edges1);
    apply_edge_detection(image2, image_edges2);
    std::cout << "Edges 1: "; print_distribution(image_edges1);
    std::cout << "Edges 2: "; print_distribution(image_edges2);

    // Display the images
    std::vector<cv::Mat> display_images = for_display({image_edges1, image_edges2});
    assert(display_images.size() == 2);
    for (const auto &i : display_images) {
        cv::imshow("Edge Image", i);
        cv::waitKey(0);
    }

    // Calculate the difference
    double difference = get_difference_score(image_edges1, image_edges2);
    std::cout << "Difference: " << difference << std::endl;
}

void apply_edge_detection(const cv::Mat &input, cv::Mat &output) {
    assert(&output != &input);

    // Convert to gray scale
    cv::Mat input_gray_scale;
    cv::cvtColor(input, input_gray_scale, CV_RGB2GRAY);

    // Normalize input
    cv::Mat input_normalized;
    normalize_image(input_gray_scale, input_normalized);

    // Blur the image
    cv::Mat input_blurred;
    cv::GaussianBlur(input_normalized, input_blurred, cv::Size(5, 5), 0, 0);

    // Apply edge detection filter
    cv::Laplacian(input_blurred, output, CV_32F, 3);
}

void normalize_image(const cv::Mat &input, cv::Mat &output) {
    cv::Mat mean, std_dev;
    cv::meanStdDev(input, mean, std_dev);
    input.convertTo(output, CV_32FC(input.channels()));
    output -= mean;
    output /= std_dev;
}

double get_difference_score(const cv::Mat &a, const cv::Mat &b) {
    return cv::mean(cv::abs(a - b))[0];
}

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
