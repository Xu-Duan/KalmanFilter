#include "KalmanFilter.hpp"
#include <iostream>

int main() {
    int n = 2;
    int m = 1;
    Eigen::MatrixXd A(n, n);
    Eigen::MatrixXd H(m, n);
    Eigen::MatrixXd R(m, m);
    Eigen::MatrixXd Q(n, n);
    Eigen::MatrixXd P(n, n);
    A << 0.9952315, 0.9309552, -0.0093096, 0.8635746;
    H << 1, 0;
    R << 0.25;
    Q << 8.4741e-04,1.2257e-03,1.2257e-03,2.4557e-03; 
    P << 1, 0, 0, 0.01;
    KalmanFilter kf(A, H, R, Q);
    kf.init(Eigen::MatrixXd::Zero(n, 1), P);
    kf.predict();
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(m, 1);
    kf.update(z);
    std::cout << "State: " << kf.get_state() << std::endl;
    return 0;
}