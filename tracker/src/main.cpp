#include <iostream>
#include "csv_parser.h"
#include "box_types.h"
#include "renderer.h"
#include "tracking_manager.h"

void emulateAndTrack(const std::map<std::string, std::vector<Box>>& data, 
    const std::map<std::string, double>& timestamp_image_map, 
    BoxTracker& tracker)
{
    // Create variables
    double current_timestamp = 0.0;
    double previous_timestamp = 0.0;

    // Iterate and loop
    for (const auto& entry : data)
    {
        const std::string& image_path = entry.first;
        const std::vector<Box>& boxes = entry.second;

        // Grab the timestamp
        auto timestamp_it = timestamp_image_map.find(image_path);
        if (timestamp_it != timestamp_image_map.end())
        {
            current_timestamp = timestamp_it->second;
        }

        // Read the image
        cv::Mat image = cv::imread(image_path);

        // Create new processed boxes
        std::vector<Box> processed_boxes;

        // Check if map is empty
        if (tracker.isEmpty())
        {
            // Run initialisation
            tracker.init(boxes, current_timestamp);
        }
        else
        {
            // Run tracking
            processed_boxes = tracker.track(boxes, current_timestamp, image);
        }

        // Grab the boxes and ids
        const std::unordered_map<int, BoxFilter> tracked_boxes = 
            tracker.getTrackedBoxes();

        // @TODO grab the indexes of the boxes that are ok etc.

        // Draw the tracked boxes
        drawTrackedBoxes(image, tracked_boxes);

        // draw boxes
        drawBoxes(image, processed_boxes);

        // Show the image
        cv::imshow("Phillip likes goth girls", image);

        // Calculate wait for
        int wait_time = (current_timestamp - previous_timestamp) * 1000;

        // Delay
        cv::waitKey(0);

        // Set the previous timestamp
        previous_timestamp = current_timestamp;
    }
};

int main(int argc, char *argv[]) 
{
    std::cout << "Phillip likes goth girls" << std::endl;

    // Create csv filepath string
    std::string baseDirectory;

    // Parse command line argument
    if (argc > 1) {
        baseDirectory = argv[1];
    }

    // Create emulator
    Emulator emulator(baseDirectory);

    // Parse the csv
    parseCSV(baseDirectory + "/coordinates.csv", emulator);

    // Parse timestamps
    std::vector<double> timestamps = parseTimestamps(
        baseDirectory + "/timestamps.csv");

    // Index timestamps
    emulator.loadTimestamps(timestamps);

    // Return the data
    const std::map<std::string, std::vector<Box>> data = emulator.getData();

    // Grab the map
    const std::map<std::string, double> timestamp_image_map = 
        emulator.getTimestampMap();

    // Create the tracking manager
    BoxTracker tracker;

    // Run emulate and tracking loop
    emulateAndTrack(data, timestamp_image_map, tracker);

    return 0;
}