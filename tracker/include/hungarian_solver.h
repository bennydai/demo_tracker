#ifndef HUNGARIAN_SOLVER_H
#define HUNGARIAN_SOLVER_H

#include <vector>
#include <unordered_set>
#include <set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "box_types.h"
#include "math_utils.h"

struct DataAssociationResults
{
    std::unordered_map<int, int> _matches;
    std::unordered_set<int> _untracked_boxes;
    std::unordered_set<int> _new_boxes_to_init;

    // Filtered boxes 
    std::vector<Box> _processed_boxes;

    // Constructor with tracked indices
    void update(const std::unordered_set<std::pair<int, int>, PairHash> &matches, 
        const std::vector<std::pair<int, int>> &tracked_indices, 
        const std::unordered_set<int> &untracked_boxes,
        const std::unordered_set<int> &new_boxes_to_init,
        const std::vector<Box> &processed_boxes)
    {
        // Reset all members
        _matches.clear();
        _untracked_boxes = untracked_boxes;
        _new_boxes_to_init = new_boxes_to_init;
        _processed_boxes = processed_boxes;

        // Populate matches map
        for (const auto & match : matches)
        {
            _matches[tracked_indices[match.first].second] = match.second;
        }

        // Print to double check
        std::cout << "DataAssociationResults initialised with:" << std::endl;
        std::cout << "Matches: " << std::endl;
        for (const auto & pair : _matches)
        {
            std::cout << "Tracked ID: " << pair.first;
            std::cout << " -> New Box Index: " << pair.second << std::endl;
        }
        std::cout << std::endl;
    };
};

class HungarianSolver
{
public:
    // Constructor
    HungarianSolver(const std::unordered_map<int, BoxFilter> &tracked_boxes,
        const std::vector<Box> &new_boxes) : tracked_boxes_(tracked_boxes),
        new_boxes_(new_boxes), num_tracked_(tracked_boxes.size()),
        num_new_(new_boxes.size()) {};

    // Solve @TODO need to pass in argument
    bool solve(DataAssociationResults &results)
    {
        // Check if we can apply hungarian algorithm
        if (!canApplyHungarian())
        {
            std::cout << "Cannot apply Hungarian algorithm." << std::endl;
            return false;
        }

        // Proceed with Hungarian algorithm
        std::cout << "Applying Hungarian algorithm..." << std::endl;

        // Create Eigen matrix of zeros
        Eigen::MatrixXd cost_matrix = 
            Eigen::MatrixXd::Ones(num_rows_, num_cols_) * 10.0;

        // Store the index mappings from unordered map and new boxes
        std::vector<std::pair<int, int>> tracked_indices;
        std::vector<int> new_indices(num_new_);
        std::iota(new_indices.begin(), new_indices.end(), 0);

        std::cout << "Print size of cost matrix: " 
            << cost_matrix.rows() << " x " << cost_matrix.cols() << std::endl;

        std::cout << "Number of tracked boxes: " 
            << num_tracked_ << std::endl;

        // Calculate cost matrix
        calculateCostMatrix(cost_matrix, tracked_indices);

        // Create sets to hold matches and unassigned indices
        std::unordered_set<std::pair<int, int>, PairHash> matches;
        std::unordered_set<int> unassigned_tracked_indices;
        std::unordered_set<int> unassigned_new_box_indices(
            new_indices.begin(), new_indices.end());
        std::unordered_set<int> new_boxes_to_init;

        // Step 1. Apply row reduction
        bool found_matches = applyRowReduction(
            cost_matrix, tracked_indices, new_indices, 
            matches, unassigned_tracked_indices,
            unassigned_new_box_indices);
        
        if (found_matches)
        {
            // Step 2. Apply column reduction to filter out undesired 
            // new boxes. 
            new_boxes_to_init = applyColReduction(tracked_indices, new_indices, 
                unassigned_new_box_indices);

            // Step 3. Check if the new boxes intersect with each other or not
            new_boxes_to_init = checkOverlaps(new_boxes_to_init);
        }

        // Copy boxes
        std::vector<Box> processed_boxes = new_boxes_;

        // Grab the indices of new boxes to be retained
        for (const auto &match : matches)
        {
            // Grab the new box index
            const int new_box_idx = match.second;

            // Set them true
            processed_boxes[new_box_idx].to_delete = true;
        }

        // Set the new boxes to init as true
        for (const auto& idx : new_boxes_to_init)
        {
            processed_boxes[idx].to_delete = true;
        }

        // Modify results
        results.update(matches, tracked_indices, 
            unassigned_tracked_indices, new_boxes_to_init, processed_boxes);

        return true;
    };

private:
    // Number of tracked boxes and new boxes
    const int num_tracked_;
    const int num_new_;

    // Static variables
    int num_rows_;
    int num_cols_;

    // Tracked boxes
    std::unordered_map<int, BoxFilter> tracked_boxes_;
    std::vector<Box> new_boxes_;

    // Function to check if hungarian algorithm can be applied
    bool canApplyHungarian()
    {
        if (num_tracked_ == 0 || num_new_ == 0)
        {
            num_rows_ = 0;
            num_cols_ = 0;
            return false;
        }
        num_rows_ = num_tracked_;
        num_cols_ = num_new_;

        return true;
    };

    // Function to calculate iou cost matrix
    void calculateCostMatrix(Eigen::MatrixXd &cost_matrix, 
        std::vector<std::pair<int, int>> &tracked_indices) const
    {
        // Iterate and calculate iou costs
        for (int i = 0; i < num_tracked_; ++i)
        {
            // Get tracked box
            auto it = tracked_boxes_.begin();
            std::advance(it, i);
            const BoxFilter& tracked_box_filter = it->second;
            const Box& tracked_box = tracked_box_filter.box;

            // Add tracked index to the vector
            tracked_indices.emplace_back(i, it->first);

            for (int j = 0; j < num_new_; ++j)
            {
                // Box
                const Box& new_box = new_boxes_[j];

                // Calculate IoU
                double iou = calculateIoU(tracked_box, new_box);

                // Fill in cost matrix
                cost_matrix(i, j) = iou;
            }
        }
    };

    // Function to apply row reduction
    bool applyRowReduction(Eigen::MatrixXd &cost_matrix, 
        const std::vector<std::pair<int, int>> &tracked_indices, 
        const std::vector<int> &new_indices, 
        std::unordered_set<std::pair<int, int>, PairHash> &matches, 
        std::unordered_set<int> &unassigned_tracked_indices,
        std::unordered_set<int> &unassigned_new_box_indices)
    {
        // Step 1. Reduce the rows
        for (int i = 0; i < cost_matrix.rows(); ++i)
        {
            double row_min = cost_matrix.row(i).minCoeff();
            cost_matrix.row(i) -= Eigen::VectorXd::Constant(
                cost_matrix.cols(), row_min);
        }

        // Check if there are enough zeros
        int unique_row_count = 0;
        int rows_to_match = cost_matrix.rows();
        std::set<int> rows_to_skip;
        std::set<int> valid_tracked_indices;

        // Insert valid tracked indices
        for (const auto& pair : tracked_indices)
        {
            valid_tracked_indices.insert(pair.first);
        }

        // Check if there is only one zero in each row
        for (int i = 0; i < cost_matrix.rows(); ++i)
        {
            int zero_count = (cost_matrix.row(i).array() == 0.0).count();
            
            // Ignore rows where columns are all zeros
            if (zero_count == cost_matrix.cols())
            {
                rows_to_match--;

                // Retain the cost matrix index
                rows_to_skip.insert(i);
            }
            else if (zero_count == 1)
            {
                unique_row_count++;
            }
        }

        // Check if unique_row_count is equivalent to the number of rows
        if (unique_row_count == rows_to_match)
        {
            std::cout << "All rows have a unique zero." << std::endl;
            std::cout << "Optimal assignment found." << std::endl;

            // Find the zero positions
            for (int i = 0; i < cost_matrix.rows(); ++i)
            {
                // Skip the row if it is in the skip set
                if (rows_to_skip.contains(i) && valid_tracked_indices.contains(i))
                {
                    // Push back if there is no assignment
                    unassigned_tracked_indices.insert(i);
                } 
                else if(valid_tracked_indices.contains(i))
                {
                    // Only consider valid tracked indices
                    Eigen::Index col_index;
                    double min_value = cost_matrix.row(i).minCoeff(&col_index);

                    matches.insert({tracked_indices[i].first, col_index});

                    // Remove from unassigned new box indices
                    unassigned_new_box_indices.erase(col_index);
                }
                else
                {
                    // Dummy row case
                    std::cout << "Row " << i;
                    std::cout << " is not a valid tracked index." << std::endl;
                }
            }

            for (const auto& match : matches)
            {
                std::cout << "Tracked ID: " << tracked_indices[match.first].second;
                std::cout << " @ row " << match.first;
                std::cout << " matched to New Box Index: " << match.second << std::endl;
            }

            // Check unassigned new indices
            std::cout << "Unassigned new box indices: [";
            for (const auto& idx : unassigned_new_box_indices)
            {
                std::cout << idx << " ";
            }
            std::cout << "]" << std::endl;

            // Print out unassigned tracked indices
            std::cout << "Unassigned tracked indices: [";
            for (const auto & idx : unassigned_tracked_indices)
            {
                std::cout << idx << " ";
                std::cout << " -> Tracker ID: " << tracked_indices[idx].second << " ";
            }
            std::cout << "]" << std::endl;

            return true;
        }

        return false;
    };

    std::unordered_set<int> applyColReduction(const std::vector<std::pair<int, int>> &tracked_indices, 
        const std::vector<int> &new_indices, 
        std::unordered_set<int> &unassigned_new_box_indices)
    {
        // Create a cost matrix between tracked indices and unassigned new box indices
        Eigen::MatrixXd cost_matrix = 
            Eigen::MatrixXd::Zero(tracked_indices.size(), 
            unassigned_new_box_indices.size());

        // @TODO calculate iou costs between existing boxes
        for (int i = 0; i < tracked_indices.size(); ++i)
        {
            // Get tracked box
            auto it = tracked_boxes_.begin();
            std::advance(it, i);
            const BoxFilter& tracked_box_filter = it->second;
            const Box& tracked_box = tracked_box_filter.box;

            int col_idx = 0;
            for (const auto& new_idx : unassigned_new_box_indices)
            {
                // Box
                const Box& new_box = new_boxes_[new_idx];

                // Calculate IoU
                double iou = calculateIoU(tracked_box, new_box, true);

                // Fill in cost matrix
                cost_matrix(i, col_idx) = iou;

                col_idx++;
            }
        }

        // Apply column reduction
        for (int i = 0; i < cost_matrix.cols(); ++i)
        {
            double col_min = cost_matrix.col(i).minCoeff();
            cost_matrix.col(i) -= Eigen::VectorXd::Constant(
                cost_matrix.rows(), col_min);
        }

        // Create a new set to return
        std::unordered_set<int> filtered_new_box_indices;

        // Check if the columns have intersections over 0.25
        for (int i = 0; i < cost_matrix.cols(); ++i)
        {
            // Grab the sum
            double col_sum = cost_matrix.col(i).sum();

            // Grab the box
            auto it = unassigned_new_box_indices.begin();
            std::advance(it, i);
            const int new_box_idx = *it;

            if (col_sum < 0.25)
            {
                filtered_new_box_indices.insert(new_box_idx);
            }
        }

        // Check if any new boxes are to be initialised
        std::cout << "New boxes to initialise after column reduction: [";
        for (const auto& idx : filtered_new_box_indices)
        {
            std::cout << idx << " ";
        }
        std::cout << "]" << std::endl;

        return filtered_new_box_indices;
    };

    // Function to ensure new boxes don't overlap with each other
    std::unordered_set<int> checkOverlaps(const std::unordered_set<int> &new_box_indices)
    {
        // Check if there are any new boxes
        if (new_box_indices.size() <= 1)
        {
            return new_box_indices;
        }

        // Create cost matrix 
        Eigen::MatrixXd cost_matrix = 
            Eigen::MatrixXd::Zero(new_box_indices.size(), new_box_indices.size());

        // Create a vector of std pair 
        std::unordered_map<int, int> index_pairs;

        // Iterate
        for (int i = 0; i < new_box_indices.size(); ++i)
        {
            auto it_i = new_box_indices.begin();
            std::advance(it_i, i);
            const Box& box_i = new_boxes_[*it_i];

            // Create index pair
            index_pairs.emplace(i, *it_i);

            for (int j = 0; j < new_box_indices.size(); ++j)
            {
                if (i == j) continue;

                auto it_j = new_box_indices.begin();
                std::advance(it_j, j);
                const Box& box_j = new_boxes_[*it_j];

                // Fill in cost matrix
                cost_matrix(i, j) = calculateIoU(box_i, box_j, true);
            }
        }

        std::cout << "Cost matrix for new box overlaps: " << std::endl;
        std::cout << cost_matrix << std::endl;

        // Create indices to remove 
        std::unordered_set<std::pair<int, int>, PairHash> overlap_indices;
        for (int i = 0; i < cost_matrix.rows(); ++i)
        {
            for (int j = 0; j < cost_matrix.cols(); ++j)
            {
                // Overlap threshold
                if (cost_matrix(i, j) > 0.0)
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

        // {Print out indices to remove
        std::cout << "Indices of new boxes to remove due to overlaps: [";
        for (const auto& idx : indices_to_remove)
        {
            std::cout << index_pairs[idx] << " ";
        }
        std::cout << "]" << std::endl;

        // Create unordered set
        std::vector<int> new_box_indices_filtered;

        // Iterate over new_box_indices to return
        for (int i = 0; i < new_box_indices.size(); ++i)
        {
            auto it = new_box_indices.begin();
            std::advance(it, i);
            const int idx = *it;

            // Check if idx is in indices_to_remove
            if (indices_to_remove.find(i) == indices_to_remove.end())
            {
                new_box_indices_filtered.push_back(idx);
            }
        }

        std::cout << "Filtered new box indices after overlap removal: [";
        for (const auto& test : new_box_indices_filtered)
        {
            std::cout << test << " ";
        }
        std::cout << "]" << std::endl;

        // Convert to set
        std::unordered_set<int> new_box_indices_filtered_set(
            new_box_indices_filtered.begin(), new_box_indices_filtered.end());

        return new_box_indices_filtered_set;
    };
};

#endif // HUNGARIAN_SOLVER_H