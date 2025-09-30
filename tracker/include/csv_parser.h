#ifndef csv_parser_H
#define csv_parser_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "box_types.h"
#include "renderer.h"

void parseCSV(const std::string &filename, 
    Emulator &emulator)
{
    // Read filename
    std::ifstream file(filename);

    // Check if file is open
    if (!file.is_open())
    {
        // Check
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }

    // Create line
    std::string line;

    // Skip the header
    std::getline(file, line);

    // Parse
    while (std::getline(file, line))
    {
        // String values
        std::string imagePath, cx, cy, width, height, confidence;
        std::stringstream ss(line);

        // Grab values
        bool imageBool = static_cast<bool>(std::getline(ss, imagePath, ','));
        bool cxBool = static_cast<bool>(std::getline(ss, cx, ','));
        bool cyBool = static_cast<bool>(std::getline(ss, cy, ','));
        bool widthBool = static_cast<bool>(std::getline(ss, width, ','));
        bool heightBool = static_cast<bool>(std::getline(ss, height, ','));
        bool confidenceBool = static_cast<bool>(std::getline(ss, confidence, ','));

        // Create box and push
        if (imageBool && cxBool && cyBool && widthBool && heightBool && confidenceBool)
        {
            // Grab the box
            Eigen::Vector2d center = Eigen::Vector2d(std::stod(cx), std::stod(cy));
            double widthValue = std::stod(width);
            double heightValue = std::stod(height);
            double confidenceValue = std::stod(confidence);

            Box box(center, widthValue, heightValue, confidenceValue);

            // Load image and paths
            emulator.loadImageAndBox(imagePath, box);
        }
    }

    // Print
    emulator.print();
};

std::vector<double> parseTimestamps(const std::string &filename)
{
    // Create timestamp
    std::vector<double> timestamps;

    // Read filename
    std::ifstream file(filename);

    // Check if file is open
    if (!file.is_open())
    {
        // Check
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }

    // Create line
    std::string line;

    // Skip the header
    std::getline(file, line);

    // Parse
    while (std::getline(file, line))
    {
        // String values
        std::string timestamp, imagePath;
        std::stringstream ss(line);

        // Grab values
        bool timeBool = static_cast<bool>(std::getline(ss, timestamp, ','));
        bool imageBool = static_cast<bool>(std::getline(ss, imagePath, ','));

        // Create box and push
        if (timeBool && imageBool)
        {
            // Convert timestamp to double
            double timeValue = std::stod(timestamp);

            timestamps.push_back(timeValue);
        }
    }

    return timestamps;
};

#endif // csv_parser_H