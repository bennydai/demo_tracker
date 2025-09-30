#ifndef tracking_manager_H
#define tracking_manager_H

#include <unordered_map>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "box_types.h"
#include "data_utils.h"
#include "debug_utils.h"
#include "hungarian_solver.h"

class BoxTracker
{
public:
    // Constructor
    BoxTracker() {};

    // Check if map is empty
    bool isEmpty() const
    {
        return tracked_boxes_.empty();
    };

    // Check the size
    std::size_t getSize() const
    {
        return tracked_boxes_.size();
    };

    // Initialise
    void init(const std::vector<Box>& boxes, 
        const double& timestamp)
    {
        // Check that there is no overlapping boxes @TODO
        std::vector<Box> filtered_boxes = checkForOverlaps(boxes);

        for (const auto& box : filtered_boxes)
        {
            tracked_boxes_.emplace(id_counter++, BoxFilter(box, timestamp));
        }

        // Print number of initialisations
        std::cout << "Number of initialisations: " << getSize() << std::endl;
    };

    // Track
    void track(const std::vector<Box>& boxes, 
        const double& timestamp, const cv::Mat& image)
    {
        // Predict all the tracked boxes
        for (auto & [id, box_filter] : tracked_boxes_)
        {
            // Run kalman filter prediction
            box_filter.predict(timestamp);
        }

        // @TODO run hungarian algorithm to determine matches
        findBestMatches(boxes, image);

        std::cout << "Current ID counter: " << id_counter << std::endl;

        // Update all the boxes

    };

    // Get tracked boxes
    const std::unordered_map<int, BoxFilter>& getTrackedBoxes() const
    {
        return tracked_boxes_;
    }

private:
    // Kalman filter implementation details @todo
    // ...

    // Unordered map
    std::unordered_map<int, BoxFilter> tracked_boxes_;

    // ID counter
    int id_counter = 0;

    // Run the hungarian algorithm to determine matches
    void findBestMatches(const std::vector<Box>& boxes, const cv::Mat& image)
    {
        // @TODO implement
        // 1. Create cost matrix with cost values
        // 2. Apply Hungarian algorithm
        // 3. Return matches

        // @TODO might need to put this into a class
        HungarianSolver solver(tracked_boxes_, boxes);
        bool success = solver.solve();
    };

    // Overlap removal for initialising 
    std::vector<Box> checkForOverlaps(const std::vector<Box> &boxes)
    {
        // Checking for overlaps
        std::cout << "Checking for overlaps..." << std::endl;

        // Create a cost matrix
        Eigen::MatrixXd cost_matrix = 
            Eigen::MatrixXd::Zero(boxes.size(), boxes.size());

        // Print 

        // For each box, calculate iou
        for (int i = 0; i < boxes.size(); ++i)
        {
            for (int j = 0; j < boxes.size(); ++j)
            {
                // Skip self-comparison
                if (i == j) continue;

                double iou = calculateIoU(boxes[i], boxes[j], true);
                cost_matrix(i, j) = iou;
            }
        }

        // Grab the indices of overlaps
        std::unordered_set<std::pair<int, int>, PairHash> overlap_indices;
        for (int i = 0; i < cost_matrix.rows(); ++i)
        {
            for (int j = 0; j < cost_matrix.cols(); ++j)
            {
                // Overlap threshold
                if (cost_matrix(i, j) > 0.25)
                {
                    // Arrange indices in ascending order
                    if (i > j)
                    {
                        overlap_indices.insert({j, i});
                    }
                    else
                    {
                        overlap_indices.insert({i, j});
                    }
                }
            }
        }

        // Size of overlap indices
        std::cout << "Number of overlapping box pairs: " 
                  << overlap_indices.size() << std::endl;

        // Indices to remove
        std::unordered_set<int> indices_to_remove;

        // Keep the first box of each pair
        for (const auto& pair : overlap_indices)
        {
            indices_to_remove.insert(pair.second);
        }

        // Iterate over boxes to return
        std::vector<Box> filtered_boxes;
        for (int i = 0; i < boxes.size(); ++i)
        {
            // If it is not in the list to remove, keep it
            if (indices_to_remove.find(i) == indices_to_remove.end())
            {
                filtered_boxes.push_back(boxes[i]);
            }
        }

        // Print number of boxes removed
        std::cout << "Filtered boxes count: " << filtered_boxes.size() << std::endl;
        std::cout << "Original boxes count: " << boxes.size() << std::endl;

        return filtered_boxes;
    };
};

#endif