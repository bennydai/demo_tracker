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
        // Check that there is no overlapping boxes 
        std::vector<Box> filtered_boxes = checkForOverlaps(boxes);

        for (const auto& box : filtered_boxes)
        {
            tracked_boxes_.emplace(id_counter++, BoxFilter(box, timestamp));
        }

        // Print number of tracked boxes
        std::cout << "Number of tracked boxes: " << getSize() << std::endl;
    };

    // Track
    std::vector<Box> track(const std::vector<Box>& boxes, 
        const double& timestamp, const cv::Mat& image)
    {
        // Predict all the tracked boxes
        for (auto & [id, box_filter] : tracked_boxes_)
        {
            // Run kalman filter prediction
            box_filter.predict(timestamp);
        }

        // Run hungarian algorithm to determine matches
        DataAssociationResults results = findBestMatches(boxes, image);

        if (results._matches.size() > 0)
        {
            // Iterate through the matches and update
            for (auto & [tracker_id, box_idx] : results._matches)
            {
                // Fetch the filter
                BoxFilter& tracked_box_filter = tracked_boxes_.at(tracker_id);

                // Retrieve the new box measurement
                const Box& new_box = boxes[box_idx];

                // Run the update step of the Kalman filter
                tracked_box_filter.update(new_box);
            }
        }

        // Delete unconfident tracked boxes
        cullUnconfidentBoxes();

        // Initialise new boxes
        if (results._new_boxes_to_init.size() > 0)
        {
            // Create vector of boxes to init
            std::vector<Box> boxes_to_init;
            for (const auto& idx : results._new_boxes_to_init)
            {
                boxes_to_init.push_back(boxes[idx]);
            }

            // Initialise new boxes
            init(boxes_to_init, timestamp);
        }

        // Update all the boxes
        return results._processed_boxes;
    };

    // Get tracked boxes
    const std::unordered_map<int, BoxFilter>& getTrackedBoxes() const
    {
        return tracked_boxes_;
    }

private:
    // Unordered map
    std::unordered_map<int, BoxFilter> tracked_boxes_;

    // ID counter
    int id_counter = 0;

    // Run the hungarian algorithm to determine matches
    DataAssociationResults findBestMatches(
        const std::vector<Box>& boxes, const cv::Mat& image)
    {
        HungarianSolver solver(tracked_boxes_, boxes);

        DataAssociationResults results;

        bool success = solver.solve(results);

        // Check if success
        if (success)
        {
            return results;
        }

        return DataAssociationResults();
    };

    // Overlap removal for initialising 
    std::vector<Box> checkForOverlaps(const std::vector<Box> &boxes)
    {
        // Create a cost matrix
        Eigen::MatrixXd cost_matrix = 
            Eigen::MatrixXd::Zero(boxes.size(), boxes.size());

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

        return filtered_boxes;
    };

    // Cull unconfident boxes
    void cullUnconfidentBoxes()
    {
        std::vector<int> ids_to_remove;

        // Iterate through tracked boxes and remove unconfident ones
        for (const auto& [id, box_filter] : tracked_boxes_)
        {
            if (!box_filter.isConfident())
            {
                std::cout << "Culling Tracker ID: " << id << std::endl;
                ids_to_remove.push_back(id);
            }
        }

        // Remove culled boxes
        for (int id : ids_to_remove)
        {
            tracked_boxes_.erase(id);
        }
    };
};

#endif