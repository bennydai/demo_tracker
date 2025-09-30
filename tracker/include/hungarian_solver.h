#ifndef HUNGARIAN_SOLVER_H
#define HUNGARIAN_SOLVER_H

#include <vector>
#include <unordered_set>
#include <set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "box_types.h"
#include "math_utils.h"

class HungarianSolver
{
public:
    // Constructor
    HungarianSolver(const std::unordered_map<int, BoxFilter> &tracked_boxes,
        const std::vector<Box> &new_boxes) : tracked_boxes_(tracked_boxes),
        new_boxes_(new_boxes), num_tracked_(tracked_boxes.size()),
        num_new_(new_boxes.size()) {};

    // Solve
    bool solve()
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

        // Step 1. Apply row reduction
        bool found_matches = applyRowReduction(
            cost_matrix, tracked_indices, new_indices, 
            matches, unassigned_tracked_indices,
            unassigned_new_box_indices);

        if (found_matches)
        {
            return true;
        }

        // // Step 2. Apply column reduction
        // found_matches = applyColReduction(
        //     cost_matrix, tracked_indices, new_indices, 
        //     matches, unassigned_tracked_indices,
        //     unassigned_new_box_indices);

        // @TODO need to think about it


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
        else if(num_tracked_ > num_new_)
        {
            num_rows_ = num_tracked_;
            num_cols_ = num_tracked_;
        }
        else
        {
            num_rows_ = num_new_;
            num_cols_ = num_new_;
        }

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

        // @TODO need to double check all the data structures

        // @TODO check if there is only one zero in each row
        for (int i = 0; i < cost_matrix.rows(); ++i)
        {
            int zero_count = (cost_matrix.row(i).array() == 0.0).count();
            
            // Ignore rows where columns are all zeros
            if (zero_count == cost_matrix.cols())
            {
                rows_to_match--;

                std::cout << "Skipping row " << i;
                std::cout << " as all elements are zero." << std::endl;
                std::cout << "Size of zero count: " << zero_count << std::endl;

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
            std::cout << "All rows have a unique zero. Optimal assignment found." << std::endl;
            std::cout << "Print out pairs: ";
            std::cout << std::endl;

            // Find the zero positions
            for (int i = 0; i < cost_matrix.rows(); ++i)
            {
                // Skip the row if it is in the skip set
                if (rows_to_skip.contains(i) && valid_tracked_indices.contains(i))
                {
                    // Push back
                    std::cout << "Tracked ID: " << tracked_indices[i].second;
                    std::cout << " @ row " << tracked_indices[i].first;
                    std::cout << " has no assignment." << std::endl;
                    unassigned_tracked_indices.insert(i);
                } 
                else if(valid_tracked_indices.contains(i))
                {
                    Eigen::Index col_index;
                    double min_value = cost_matrix.row(i).minCoeff(&col_index);

                    matches.insert({tracked_indices[i].first, col_index});

                    // Remove from unassigned new box indices
                    unassigned_new_box_indices.erase(col_index);
                }
                else
                {
                    std::cout << "Row " << i << " is not a valid tracked index." << std::endl;
                }
            }

            std::cout << "Size of cost matrix: ";
            std::cout << cost_matrix.rows() << " x " << cost_matrix.cols();
            std::cout << std::endl;
            std::cout << "Size of matches: " << matches.size() << std::endl;
            std::cout << "Size of tracked indices: " << tracked_indices.size() << std::endl;
            std::cout << "Valid tracked indices size: " << valid_tracked_indices.size() << std::endl;
            std::cout << "Cost matrix after row reduction: \n" << cost_matrix << std::endl;

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

    bool applyColReduction(Eigen::MatrixXd &cost_matrix, 
        const std::vector<int> &tracked_indices, 
        const std::vector<int> &new_indices, 
        std::unordered_set<std::pair<int, int>, PairHash> &matches, 
        std::unordered_set<int> &unassigned_tracked_indices,
        std::unordered_set<int> &unassigned_new_box_indices)
    {
        // Step 2. Reduce the columns
        for (int j = 0; j < cost_matrix.cols(); ++j)
        {
            double col_min = cost_matrix.col(j).minCoeff();
            cost_matrix.col(j) -= Eigen::VectorXd::Constant(
                cost_matrix.rows(), col_min);
        }

        // Check if there are enough zeros
        int unique_row_count = 0; 
        int rows_to_match = cost_matrix.rows();
        std::set<int> assigned_box_indices_to_skip;

        // @TODO check if there are enough zeros 
        std::cout << "Cost Matrix Print out for 2nd step: \n" << cost_matrix << std::endl;

        // @TODO need to implement the rest of the logic

        return false;

    };
};

#endif // HUNGARIAN_SOLVER_H