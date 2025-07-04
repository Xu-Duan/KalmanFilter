#include "KalmanFilter.hpp"
#include <iostream>

KalmanFilter::KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H, 
    const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q) {
    n = A.rows();
    m = H.rows();
    n_u = B.cols();
    this->A = A;
    this->B = B;
    this->H = H;
    this->R = R;
    this->Q = Q;
}

KalmanFilter::KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& H, 
    const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q) {
    n = A.rows();
    m = H.rows();
    n_u = 0;
    this->A = A;
    this->H = H;
    this->R = R;
    this->Q = Q;
}

void KalmanFilter::predict(Eigen::VectorXd u) {
    if (n_u == 0 && u.size() > 0) {
        std::cout << "ERROR: KalmanFilter::predict: No control matrix provided" << std::endl;
        return;
    }
    
    if (n_u > 0 && u.size() == 0) {
        u = Eigen::VectorXd::Zero(n_u);  // Fixed: should be VectorXd, not MatrixXd
    }
    if (n_u == 0) {
        x = A * x;
    } else {
        x = A * x + B * u;
    }
    
    P = A * P * A.transpose() + Q;
    predicted = true;
}

void KalmanFilter::update(const Eigen::VectorXd &z) {
    if (!predicted) {
        std::cout << "KalmanFilter::update: Predicted state not available" << std::endl;
        std::cout << "WARNING: Predicting state using zero control input" << std::endl;
        this->predict();
    }
    Eigen::VectorXd y = z - H * x;
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(n, n) - K * H) * P;
    predicted = false;
}