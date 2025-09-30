#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include "box_types.h"

double calculateIoU(const Box& box1, const Box& box2, bool init = false, 
    bool debug = false)
{
    // Reference
    // https://www.v7labs.com/blog/intersection-over-union-guide

    // Calculate top left and bottom right points
    const cv::Point2f tl1 = box1.getTopLeft();
    const cv::Point2f br1 = box1.getBottomRight();
    const cv::Point2f tl2 = box2.getTopLeft();
    const cv::Point2f br2 = box2.getBottomRight();

    // Find the intersection rectangle
    const double a_x1 = std::max(tl1.x, tl2.x);
    const double a_y1 = std::max(tl1.y, tl2.y);
    const double a_x2 = std::min(br1.x, br2.x);
    const double a_y2 = std::min(br1.y, br2.y);

    // Check if there is no intersection
    if (a_x2 < a_x1 || a_y2 < a_y1)
    {
        if (init)
        {
            return 0.0; // Return negative value for initialization
        }
        return 10.0; // No overlap, return large cost
    }

    // Calculate intersection width and height
    const double intersection_width = std::max(0.0, a_x2 - a_x1);
    const double intersection_height = std::max(0.0, a_y2 - a_y1);

    // Calculate intersection area
    const double intersection_area = intersection_width * intersection_height;

    // Calculate union area
    const double box1_area = box1.width * box1.height;
    const double box2_area = box2.width * box2.height;

    const double union_area = box1_area + box2_area - intersection_area;

    if (debug)
    {
        std::cout << "Box 1: " << tl1 << " " << br1 << std::endl;
        std::cout << "Box 2: " << tl2 << " " << br2 << std::endl;
        std::cout << "Intersection: " << a_x1 << ", " << a_y1;
        std::cout << ", " << a_x2 << ", " << a_y2 << std::endl;
        std::cout << "Intersection width and height: ";
        std::cout << intersection_width << ", " << intersection_height << std::endl;
        std::cout << "Intersection area: " << intersection_area << std::endl;
        std::cout << "Union area: " << union_area << std::endl;
        std::cout << "Box 1 calculated width: " << br1.x - tl1.x << std::endl;
        std::cout << "Box 1 width: " << box1.width << std::endl;
        std::cout << "Box 2 calculated width: " << br2.x - tl2.x << std::endl;
        std::cout << "Box 2 width: " << box2.width << std::endl;
    }

    // Calculate IoU
    const double iou = intersection_area / union_area;

    if (init)
    {
        return iou;
    }

    return 1.0 - iou;
};

#endif // MATH_UTILS_H