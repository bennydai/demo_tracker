#ifndef RENDERER_H
#define RENDERER_H

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <map>
#include <vector>
#include <string>

void drawBoxes(cv::Mat &image, const std::vector<Box>& boxes)
{
    // Draw the boxes
    for (const auto& box : boxes)
    {
        // Draw the box
        cv::rectangle(image, 
            box.getCvRect(), cv::Scalar(255, 0, 0), 2);
    }
};

void drawTrackedBoxes(cv::Mat &image, 
    const std::unordered_map<int, BoxFilter>& tracked_boxes)
{
    // Draw the boxes
    for (const auto& entry : tracked_boxes)
    {
        const int id = entry.first;
        const BoxFilter& box_filter = entry.second;

        // Grab the confidence 
        Eigen::Vector3i cov_ellipse = box_filter.getCovEllipse();

        // Draw the box
        cv::rectangle(image, 
            box_filter.box.getCvRect(), cv::Scalar(0, 255, 0), 1);

        // Draw the confidence bound
        cv::ellipse(image, 
            box_filter.box.getCenter(), cv::Size(cov_ellipse[0], cov_ellipse[1]),
            cov_ellipse[2], 0, 360, cv::Scalar(0, 255, 0), 1);

        // Draw the ID
        cv::putText(image, std::to_string(id), 
            box_filter.box.getTopLeft(), cv::FONT_HERSHEY_SIMPLEX, 
            0.5, cv::Scalar(0, 255, 0), 1);
    }
};

class Emulator
{
public:
    Emulator(const std::string &image_directory)
        : image_directory_(image_directory) {};

    // Load image and boxes
    void loadImageAndBox(const std::string &image_path, 
        const Box &box)
    {
        // Parse the image path
        std::string full_image_path = image_directory_ + "/" + image_path;

        // Check if image_path has been appended
        if (data_.find(full_image_path) == data_.end())
        {
            data_[full_image_path] = std::vector<Box>();
        }

        // Emplace
        data_[full_image_path].push_back(box);
    };

    // Load the timestamps and create a map
    void loadTimestamps(const std::vector<double> &timestamps)
    {
        // Grab all the entries from the existing data map
        std::vector<std::string> image_paths;

        // Grab the map
        for (const auto& entry : data_)
        {
            image_paths.push_back(entry.first);
        }

        // Map timestamps to image paths
        for (size_t i = 0; i < timestamps.size() && i < image_paths.size(); ++i)
        {
            timestamp_map_[image_paths[i]] = timestamps[i];
        }
    };

    // Print
    void print()
    {
        // Check lengths
        std::cout << "Number of entries: " << data_.size() << std::endl;
    };

    // Retrieve the std::map
    const std::map<std::string, std::vector<Box>>& getData() const
    {
        return data_;
    };

    // Retrieve the timestamp map
    const std::map<std::string, double>& getTimestampMap() const
    {
        return timestamp_map_;
    }

private:
    // Image directory
    const std::string image_directory_;

    // Map to read file and their boxes
    std::map<std::string, std::vector<Box>> data_;

    // Map to hold the timestamps
    std::map<std::string, double> timestamp_map_;
};

#endif