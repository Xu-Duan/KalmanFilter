#include "KalmanFilter.hpp"
#include <iostream>

KalmanFilter::KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd H, 
    Eigen::MatrixXd R, Eigen::MatrixXd Q) {
    n = A.rows();
    m = H.rows();
    n_u = B.cols();
    this->A = A;
    this->B = B;
    this->H = H;
    this->R = R;
    this->Q = Q;
}

KalmanFilter::KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd H, 
    Eigen::MatrixXd R, Eigen::MatrixXd Q) {
    n = A.rows();
    m = H.rows();
    n_u = 0;
    this->A = A;
    this->H = H;
    this->R = R;
    this->Q = Q;
}

KalmanFilter::~KalmanFilter() {
}


void KalmanFilter::init(Eigen::MatrixXd x, Eigen::MatrixXd P) {
    this->x = x;
    this->P = (P.size() == 0) ? Eigen::MatrixXd::Identity(n, n) : P;
}

Eigen::MatrixXd KalmanFilter::get_state() const {
    return this->x;
}

void KalmanFilter::predict(Eigen::MatrixXd u) {
    if (n_u == 0 && u.size() > 0) {
        std::cout << "ERROR: KalmanFilter::predict: No control matrix provided" << std::endl;
        return;
    }

    if (n_u > 0 && u.size() == 0) {
        u = Eigen::MatrixXd::Zero(n_u, 1);
    }
    if (n_u == 0) {
        x = A * x;
    } else {
        x = A * x + B * u;
    }
    
    P = A * P * A.transpose() + Q;
    predicted = true;
}

void KalmanFilter::update(Eigen::MatrixXd z) {
    if (!predicted) {
        std::cout << "KalmanFilter::update: Predicted state not available" << std::endl;
        std::cout << "WARNIG: Predicting state using zero control input" << std::endl;
        this->predict();
    }
    Eigen::MatrixXd y = z - H * x;
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(n, n) - K * H) * P;
    predicted = false;
}
