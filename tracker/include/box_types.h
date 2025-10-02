#ifndef BOX_TYPES_H
#define BOX_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv4/opencv2/core.hpp>

struct FilterParams
{
    // Pass in filter parameters to reduce number of arguments
    FilterParams(
        const double& initial_P_pos_cov, const double& initial_P_vel_cov,
        const double& process_Q_pos_cov, const double& process_Q_vel_cov,
        const double& R_cov, const double& cull_threshold,
        const double& display_threshold, const double& overlap_threshold) :
        initial_P_pos_cov(initial_P_pos_cov),
        initial_P_vel_cov(initial_P_vel_cov),
        process_Q_pos_cov(process_Q_pos_cov),
        process_Q_vel_cov(process_Q_vel_cov),
        R_cov(R_cov),
        cull_threshold(cull_threshold),
        display_threshold(display_threshold),
        overlap_threshold(overlap_threshold) {};

    // Initial covariance for positions, width and height
    const double initial_P_pos_cov;
    // Initial covariance for velocities
    const double initial_P_vel_cov;

    // Process noise covariance for positions, width and height
    const double process_Q_pos_cov;
    // Process noise covariance for velocities
    const double process_Q_vel_cov;

    // Measurement noise covariance for positions, width and height
    const double R_cov;

    // Threshold for culling and displaying boxes
    const double cull_threshold;
    const double display_threshold;

    // Threshold for removing overlapping boxes
    const double overlap_threshold;
};

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

class BoxFilter
{
public:
    BoxFilter(const Box& box, const double& timestamp, 
        const FilterParams& params) : 
        box(box), ts_(timestamp), params_(params)
    {
        // Set initial covariance
        P_cov_.topLeftCorner(4, 4) *= params_.initial_P_pos_cov;
        P_cov_.bottomRightCorner(4, 4) *= params_.initial_P_vel_cov;

        // Set the process noise covariance
        Q_cov_.topLeftCorner(4, 4) *= params_.process_Q_pos_cov;
        Q_cov_.bottomRightCorner(4, 4) *= params_.process_Q_vel_cov;

        // Set the measurement noise covariance
        R_cov_ *= params_.R_cov;
    };

    // Parameters
    Box box;

    // Check if the tracked box is confident
    bool isConfident() const
    {
        // Check if the positional covariance is below threshold
        const Eigen::Matrix2d pos_cov = getCovariance();
        const double trace = pos_cov.trace();

        return trace < params_.cull_threshold;
    };

    // Check if the tracked box is within displayable range
    bool isDisplayable() const
    {
        // Check if the positional covariance is below threshold
        const Eigen::Matrix2d pos_cov = getCovariance();
        const double trace = pos_cov.trace();

        return trace < params_.display_threshold;
    };

    // Get positional covariance
    Eigen::Matrix2d getCovariance() const
    {
        return P_cov_.block<2, 2>(0, 0);
    };

    // Run prediction
    void predict(const double &current_ts)
    {
        // Calculate dt
        double dt = current_ts - ts_;

        // Create state vector
        Eigen::VectorXd state = getState();

        // Create the F matrix
        const Eigen::MatrixXd F = createF(dt);

        // Update the state
        state = F * state;

        // Apply state correction
        correct(state);

        // Update the state covariance matrix
        P_cov_ = F * P_cov_ * F.transpose() + Q_cov_;

        // Update ts
        ts_ = current_ts;
    };

    // Run update step
    void update(const Box& measurement)
    {
        // Create H Matrix
        const Eigen::Matrix<double, 4, 8> H = createH();

        // Calculate the Kalman Gain
        const Eigen::Matrix<double, 8, 4> K = createKalmanGain(H);

        // Update state vector
        Eigen::VectorXd state = getState();

        // Create measurement vector
        Eigen::Vector4d z;
        z << measurement.center[0], measurement.center[1],
                measurement.width, measurement.height;
        
        // Correct state
        state = state + K * (z - H * state);

        // Apply state correction
        correct(state);

        // Update covariance
        P_cov_ = (Eigen::MatrixXd::Identity(8, 8) - K * H) * P_cov_;
    };

    // Get covariance bound
    Eigen::Vector3i getCovEllipse(const int sigma_bound = 3) const
    {
        // Grab the positional covariance
        Eigen::Matrix2d pos_cov = P_cov_.block<2, 2>(0, 0);

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

private:
    // Velocities
    double vel_cx_ = 0;
    double vel_cy_ = 0;
    double vel_w_ = 0;
    double vel_h_ = 0;

    // Covariance Matrix
    Eigen::MatrixXd P_cov_ = Eigen::MatrixXd::Identity(8, 8);

    // Pertubation Matrix for prediction 
    Eigen::MatrixXd Q_cov_ = Eigen::MatrixXd::Identity(8, 8);

    // Pertubation Matrix for measurement
    Eigen::MatrixXd R_cov_ = Eigen::MatrixXd::Identity(4, 4);

    // Params
    FilterParams params_;

    // Current timestamp
    double ts_ = 0;

    // Create state vector
    Eigen::VectorXd getState() const
    {
        // Create state vector
        Eigen::VectorXd state(8);

        state[0] = box.center[0];
        state[1] = box.center[1];
        state[2] = box.width;
        state[3] = box.height;
        state[4] = vel_cx_;
        state[5] = vel_cy_;
        state[6] = vel_w_;
        state[7] = vel_h_;

        return state;
    };

    // Apply state correction 
    void correct(const Eigen::VectorXd &state)
    {
        // Update box positions from state vector
        box.center[0] = state[0];
        box.center[1] = state[1];
        box.width = state[2];
        box.height = state[3];

        // Update velocities
        vel_cx_ = state[4];
        vel_cy_ = state[5];
        vel_w_ = state[6];
        vel_h_ = state[7];
    };

    // Create H matrix
    Eigen::Matrix<double, 4, 8> createH() const
    {
        Eigen::Matrix<double, 4, 8> H = Eigen::Matrix<double, 4, 8>::Zero();

        // Fill in the H matrix
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 2) = 1;
        H(3, 3) = 1;

        return H;
    };

    // Create Kalman Gain Matrix
    Eigen::Matrix<double, 8, 4> createKalmanGain(
        const Eigen::Matrix<double, 4, 8> &H) const
    {
        // Calculate the innovation covariance S
        Eigen::Matrix4d S = (H * P_cov_ * H.transpose()) + R_cov_;

        // Calculate the Kalman Gain K
        Eigen::Matrix<double, 8, 4> K = P_cov_ * H.transpose() * S.inverse();

        return K;
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
};
#endif 