// https://yugu.faculty.wvu.edu/files/d/fed008fa-14a3-48c4-b7b4-cb75b83ea7db/irl_wvu_online_ekf_vs_ukf_v1-0_06_28_2013.pdf

#include "UnscentedKalmanFilter.hpp"
#include "ExtendedKalmanFilter.hpp"
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <iostream>
#include <Eigen/Dense>

#define n 2
#define m 1

// Fix the variable declarations - remove type annotations and fix syntax
Eigen::VectorXd x_EKF = (Eigen::VectorXd(n) << 5.00, 2.00).finished();
Eigen::MatrixXd P_EKF = (Eigen::MatrixXd(n, n) << 1.036, 0.012, 0.012, 1.006).finished();
Eigen::VectorXd x_UKF = (Eigen::VectorXd(n) << 5.003, 2.00).finished();
Eigen::MatrixXd P_UKF = (Eigen::MatrixXd(n, n) << 1.0360182, 0.012, 0.012, 1.006).finished();

template<typename T>
T f(const T& x) {
    // Example state transition function
    T res = T::Zero(n);
    res(0) = x(0) * x(0) + x(1) * x(1); // Simple linear model
    res(1) = x(0) * x(1); // Simple linear model
    return res; // Simple linear model for demonstration
}

template<typename T>
T h(const T& x) {
    // Example measurement function
    Eigen::MatrixXd H(m, n);
    H << 1, 0; // Simple measurement model
    return H * x; // Direct measurement of the state
}

int main(){
    // Define the state transition matrix A and measurement matrix H
    Eigen::MatrixXd Q(n, n);
    Eigen::MatrixXd R(m, m);
    R << 1;
    Q << 1, 0, 0, 1;
    ExtendedKalmanFilter filter1 = ExtendedKalmanFilter(Q, R, f<autodiff::VectorXreal>, h<autodiff::VectorXreal>);
    UnscentedKalmanFilter filter2 = UnscentedKalmanFilter(Q, R, f<Eigen::VectorXd>, h<Eigen::VectorXd>);
    
    Eigen::VectorXd x(n); // Initial state
    x << 1, 2; // Initial state
    Eigen::MatrixXd P(n, n); // Initial covariance
    P << 1e-3 * 1, 0, 0, 1e-3 * 2;

    filter1.init(x, P);
    filter2.init(x, P);

    filter1.predict();
    filter2.predict();

    assert(filter1.get_state().isApprox(x_EKF, 1e-6));
    assert(filter2.get_state().isApprox(x_UKF, 1e-6));

    assert(filter1.get_cov().isApprox(P_EKF, 1e-4));
    assert(filter2.get_cov().isApprox(P_UKF, 1e-4));
    return 0;
}