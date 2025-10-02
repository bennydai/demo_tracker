#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "csv_parser.h"
#include "box_types.h"
#include "renderer.h"
#include "tracking_manager.h"

void emulateAndTrack(const std::map<std::string, std::vector<Box>>& data, 
    const std::map<std::string, double>& timestamp_image_map, 
    BoxTracker& tracker, const bool &debug_visualisation, 
    const double &visualisation_scale, const double &visualisation_slow_factor)
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

        // Calculate wait for
        const int wait_time = (current_timestamp - previous_timestamp) * 1000 * 
            visualisation_slow_factor;

        // Set the previous timestamp
        previous_timestamp = current_timestamp;

        // bifurcate dev mode vs normal mode etc.
        if (debug_visualisation)
        {
            // Draw the tracked boxes
            drawTrackedBoxesDebug(image, tracked_boxes);

            // Draw boxes
            drawBoxesDebug(image, processed_boxes);

            // Scale image
            if (visualisation_scale != 1.0)
            {
                cv::resize(image, image, cv::Size(), visualisation_scale, 
                    visualisation_scale, cv::INTER_CUBIC);
            }

            // Show the image
            cv::imshow("Phillip likes goth girls", image);

            // Delay
            cv::waitKey(0);
        }
        else
        {
            // Draw boxes
            cv::Mat output_image = drawBoxes(image, 
                processed_boxes, tracked_boxes);

            // Scale image
            if (visualisation_scale != 1.0)
            {
                cv::resize(output_image, output_image, cv::Size(), visualisation_scale,
                    visualisation_scale, cv::INTER_CUBIC);
            }

            // Show the image
            cv::imshow("Phillip likes goth girls", output_image);

            // Delay 
            cv::waitKey(wait_time);
        }
    }
};

int main(int argc, char *argv[]) 
{
    // Parse in configuration file
    std::string path_to_config = "../cfg/config.yaml";

    // Load configuration
    YAML::Node config = YAML::LoadFile(path_to_config);

    // Parse in base directory
    std::string baseDirectory = config["directory"].as<std::string>();

    // Parse in debug visualisation
    const bool debug_visualisation = 
        config["visualisation"]["debug"].as<bool>();
    const double visualisation_scale = 
        config["visualisation"]["scale"].as<double>();
    const double visualisation_slow_factor =
        config["visualisation"]["slow_factor"].as<double>();

    // Create filter params
    FilterParams params(
        config["filter_params"]["initial_P_pos_cov"].as<double>(),
        config["filter_params"]["initial_P_vel_cov"].as<double>(),
        config["filter_params"]["process_Q_pos_cov"].as<double>(),
        config["filter_params"]["process_Q_vel_cov"].as<double>(),
        config["filter_params"]["R_cov"].as<double>(),
        config["filter_params"]["cull_threshold"].as<double>(),
        config["filter_params"]["display_threshold"].as<double>(),
        config["filter_params"]["overlap_threshold"].as<double>()
    );

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
    BoxTracker tracker(params);

    // Run emulate and tracking loop
    emulateAndTrack(data, timestamp_image_map, tracker, 
        debug_visualisation, visualisation_scale, 
        visualisation_slow_factor);

    return 0;
}