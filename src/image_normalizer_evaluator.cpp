#include <iostream>
#include <memory>
#include "image_normalizer_evaluator.h"
#include "image_normalizer.h"


namespace simage::image_normalizer::evaluator {

// Evaluate the effectiveness of some parameters on image groups
double evaluate_parameters(std::vector<std::vector<cv::Mat>> image_groups, const ImageNormalizeParameters &parameters);

void test_normalize_parameters(const std::vector<std::vector<cv::Mat>> image_groups, uint32_t num_iterations) {
    const auto random_generator = ImageNormalizeParameters::get_random_generator();

    double min_cost = std::numeric_limits<double>::max();
    std::unique_ptr<ImageNormalizeParameters> min_parameters;

    for (uint32_t i = 0; i < num_iterations; i++) {
        const auto random_parameters = random_generator();

        double cost = evaluate_parameters(image_groups, random_parameters);
        if (cost < min_cost) {
            min_cost = cost;
            min_parameters = std::make_unique<ImageNormalizeParameters>(random_parameters);
        }
    }

    std::cout << "Found minimum parameters " << *min_parameters << " with cost " << min_cost << std::endl;
}

double evaluate_parameters(
        std::vector<std::vector<cv::Mat>> image_groups, const ImageNormalizeParameters &parameters) {
    // Create edge images
    std::vector<std::vector<cv::Mat>> edges_groups;
    for (const auto &image_group : image_groups) {
        std::vector<cv::Mat> edges_group;
        for (const auto &image : image_group) {
            cv::Mat edges;
            image_normalizer::normalize(image, edges, parameters);
            edges_group.push_back(edges);
        }
        edges_groups.push_back(edges_group);
    }

    double intra_group_difference_total = 0;
    int intra_group_difference_count = 0;
    for (const auto &edges_group : edges_groups) {
        for (auto iter1 = edges_group.begin(); iter1 != edges_group.end(); iter1++) {
            for (auto iter2 = iter1 + 1; iter2 != edges_group.end(); iter2++) {
                intra_group_difference_total += get_difference_score(*iter1, *iter2);
                intra_group_difference_count++;
            }
        }
    }
    double intra_group_difference_average = intra_group_difference_total / intra_group_difference_count;

    double inter_group_difference_total = 0;
    int inter_group_difference_count = 0;
    for (auto iter1 = edges_groups.begin(); iter1 != edges_groups.end(); iter1++) {
        for (auto iter2 = iter1 + 1; iter2 != edges_groups.end(); iter2++) {
            const auto &group1 = *iter1;
            const auto &group2 = *iter2;

            for (const auto &image1 : group1) {
                for (const auto &image2 : group2) {
                    inter_group_difference_total += get_difference_score(image1, image2);
                    inter_group_difference_count++;
                }
            }
        }
    }
    double inter_group_difference_average = inter_group_difference_total / inter_group_difference_count;

    double cost = pow(intra_group_difference_average, 2) - inter_group_difference_average;

    std::cout <<
              "Cost: " << cost << "; " <<
              "Params: " << parameters << "; " <<
              "Intra-group difference: " << intra_group_difference_average << "; " <<
              "Inter-group difference: " << inter_group_difference_average << std::endl;

    return cost;
}

}
