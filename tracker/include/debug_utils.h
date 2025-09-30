#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

void writeMatrixToCSV(const std::string &filename, const Eigen::MatrixXd &matrix)
{
    // Define csv format
    const static Eigen::IOFormat CSVFormat(
        Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

    // Open the file
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();

        std::cout << "Matrix written to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error opening file " << filename << std::endl;
    }
};

#endif // DEBUG_UTILS_H