#ifndef BOX_TYPES_H
#define BOX_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv4/opencv2/core.hpp>

struct Box 
{
    Box(const Eigen::Vector2d& center, 
        const double &width, const double &height, 
        const double &confidence) : center(center), 
        width(width), height(height), 
        confidence(confidence) {};

    cv::Point2f getCenter() const 
    {
        return cv::Point2f(center[0], center[1]);
    };

    cv::Point2f getTopLeft() const 
    {
        return cv::Point2f(center[0] - width / 2.0, center[1] - height / 2.0);
    };

    cv::Point2f getBottomRight() const 
    {
        return cv::Point2f(center[0] + width / 2.0, center[1] + height / 2.0);
    };

    cv::Rect getCvRect() const
    {
        cv::Point top_left = getTopLeft();
        return cv::Rect(top_left, cv::Size(width, height));
    };

    // Print
    void print() const
    {
        std::cout << "Box Center: " << center.transpose() << std::endl;
        std::cout << "Box Width: " << width << std::endl;
        std::cout << "Box Height: " << height << std::endl;
    };

    // Center of the box
    Eigen::Vector2d center;
    double width;
    double height;
    double confidence;

    // Flag to set if the box will be deleted
    bool to_delete = false;
};

struct BoxFilter
{
    BoxFilter(const Box& box, const double& timestamp) : 
        box(box), ts(timestamp) {};

    // Parameters
    Box box;

    // Velocity parameters @TODO test F with various vel
    double vel_cx = 0;
    double vel_cy = 0;
    double vel_w = 10;
    double vel_h = 10;

    // Covariance Matrix
    Eigen::MatrixXd P_cov = Eigen::MatrixXd::Identity(8, 8) * 5.0;

    // Pertubation Matrix
    Eigen::MatrixXd Q_cov = Eigen::MatrixXd::Identity(8, 8) * 1.0;

    // Current timestamp
    double ts = 0;

    // Run prediction
    void predict(const double &current_ts)
    {
        // Calculate dt
        double dt = current_ts - ts;

        // Create state vector
        Eigen::VectorXd state(8);
        state[0] = box.center[0];
        state[1] = box.center[1];
        state[2] = box.width;
        state[3] = box.height;
        state[4] = vel_cx;
        state[5] = vel_cy;
        state[6] = vel_w;
        state[7] = vel_h;

        // Create the F matrix
        const Eigen::MatrixXd F = createF(dt);

        // Update the state
        state = F * state;

        // Update box positions based on constant velocities
        box.center[0] += vel_cx * dt;
        box.center[1] += vel_cy * dt;
        box.width += vel_w * dt;
        box.height += vel_h * dt;

        // Update the state covariance matrix
        P_cov = F * P_cov * F.transpose() + Q_cov;

        // Update ts
        ts = current_ts;
    };

    // Create state transition matrix
    Eigen::Matrix<double, 8, 8> createF(double dt) const 
    {
        // Assumed Motion equation is:
        // cx = cx0 + vcx * dt
        // cy = cy0 + vcy * dt
        // w = w0 + vw * dt
        // h = h0 + vh * dt

        // Therefore the state transition matrix is:
        // [1 0 0 0 dt 0  0  0
        //  0 1 0 0 0  dt 0 0
        //  0 0 1 0 0  0 dt 0
        //  0 0 0 1 0  0 0 dt
        //  0 0 0 0 1  0 0 0
        //  0 0 0 0 0  1 0 0
        //  0 0 0 0 0  0 1 0
        //  0 0 0 0 0  0 0 1]

        // Initialise
        Eigen::Matrix<double, 8, 8> F = 
            Eigen::Matrix<double, 8, 8>::Identity();

        // Fill in the state transition matrix
        F(0, 4) = dt;
        F(1, 5) = dt;
        F(2, 6) = dt;
        F(3, 7) = dt;

        return F;
    };

    // @TODO get covariance
    Eigen::Vector3i getCovEllipse(const int sigma_bound = 3) const
    {
        // Grab the positional covariance
        Eigen::Matrix2d pos_cov = P_cov.block<2, 2>(0, 0);

        // Calculate the eigenvalues and eigenvectors
        Eigen::EigenSolver<Eigen::MatrixXd> es(pos_cov);

        // Get the eigenvalues and eigenvectors
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        Eigen::MatrixXd eigenvectors = es.eigenvectors().real();

        // Calculate the minor axis
        double major_bound = sigma_bound * std::sqrt(eigenvalues(0));
        double minor_bound = sigma_bound * std::sqrt(eigenvalues(1));

        // Calculate the angle of rotation
        double angle = std::atan2(eigenvectors(1, 0), 
            eigenvectors(0, 0)) * 180.0 / CV_PI;

        // Cast values to int
        major_bound = std::round(major_bound);
        minor_bound = std::round(minor_bound);
        angle = std::round(angle);

        // Return as vector
        return Eigen::Vector3i(major_bound, minor_bound, angle);
    };
};
#endif 