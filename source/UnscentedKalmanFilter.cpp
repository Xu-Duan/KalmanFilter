#include "UnscentedKalmanFilter.hpp"
#include <iostream>

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f_func,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_func) {
    n = Q.rows();
    m = R.rows();
    this->R = R;
    this->Q = Q;
    f = f_func; // Store the nonlinear state transition function
    h = h_func; // Store the nonlinear measurement function
    // n_u will be > 0 because we passed a B matrix
}

// Constructor without control input - follows parent's pattern
UnscentedKalmanFilter::UnscentedKalmanFilter(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f_no_control,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_func){
    n = Q.rows();
    m = R.rows();
    this->R = R;
    this->Q = Q;
    h = h_func; // Store the nonlinear measurement function
    f = [f_no_control](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        return f_no_control(x);
    };
    n_u = 0; // No control input in this case
}

void UnscentedKalmanFilter::init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P){
    this->x = x;
    this->P = (P.size() == 0) ? Eigen::MatrixXd::Identity(n, n) : P;
    // n_u will be 0 because we didn't pass a B matrix

    // UKF parameters (typical values)
    alpha = 1e-3;
    beta = 2.0;
    kappa = 0.0;
    
    calculateWeights();
    // Generate sigma points
    generateSigmaPoints();

    predicted = false;
};

void UnscentedKalmanFilter::calculateWeights() {
    if (x.size() == 0) return; // Wait until state is initialized
    
    int n = x.size();
    lambda = alpha * alpha * (n + kappa) - n;
    
    int num_sigma_points = 2 * n + 1;
    Wm = Eigen::VectorXd(num_sigma_points);
    Wc = Eigen::VectorXd(num_sigma_points);
    
    // Weight for mean of first sigma point
    Wm(0) = lambda / (n + lambda);
    
    // Weight for covariance of first sigma point
    Wc(0) = lambda / (n + lambda) + (1 - alpha * alpha + beta);
    
    // Weights for remaining sigma points
    for (int i = 1; i < num_sigma_points; i++) {
        Wm(i) = 0.5 / (n + lambda);
        Wc(i) = 0.5 / (n + lambda);
    }
}

void UnscentedKalmanFilter::generateSigmaPoints() {
    int n = x.size();
    lambda = alpha * alpha * (n + kappa) - n;
    
    int num_sigma_points = 2 * n + 1;
    sigma_points = Eigen::MatrixXd(n, num_sigma_points);
    
    // Calculate square root of (n + lambda) * P
    Eigen::MatrixXd sqrt_matrix = calculateSquareRoot((n + lambda) * P);
    
    // First sigma point
    sigma_points.col(0) = x;
    
    // Remaining sigma points
    for (int i = 0; i < n; i++) {
        sigma_points.col(i + 1) = x + sqrt_matrix.col(i);
        sigma_points.col(i + 1 + n) = x - sqrt_matrix.col(i);
    }
}

Eigen::MatrixXd UnscentedKalmanFilter::calculateSquareRoot(const Eigen::MatrixXd& matrix) {
    // Using Cholesky decomposition for positive definite matrices
    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if (llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Matrix is not positive definite for Cholesky decomposition");
    }
    return llt.matrixL();
}

void UnscentedKalmanFilter::predict(Eigen::VectorXd u) {
    // Handle control input like parent class
    if (n_u == 0 && u.size() > 0) {
        std::cout << "ERROR: ExtendedKalmanFilter::predict: No control input expected" << std::endl;
        return;
    }

    if (n_u > 0 && u.size() == 0) {
        u = Eigen::VectorXd::Zero(n_u);
    }

    // Predict the state using the sigma points
    for (int i = 0; i < sigma_points.cols(); ++i) {
        sigma_points.col(i) = f(sigma_points.col(i), u);
    }
    
    // Calculate the predicted state mean
    x = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        x += Wm(i) * sigma_points.col(i);
    }
    
    // Calculate the predicted covariance
    P = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        Eigen::VectorXd delta_x = sigma_points.col(i) - x;
        P += Wc(i) * delta_x * delta_x.transpose();
    }
    P += Q; // Add process noise
    
    predicted = true; // Set predicted flag
}

void UnscentedKalmanFilter::update(const Eigen::VectorXd& z){
    if (!predicted) {
        throw std::runtime_error("Predict must be called before update.");
    }
    
    // Predict the measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(m);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        z_pred += Wm(i) * h(sigma_points.col(i));
    }
    
    // Calculate the innovation covariance
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        Eigen::VectorXd delta_z = h(sigma_points.col(i)) - z_pred;
        S += Wc(i) * delta_z * delta_z.transpose();
    }
    S += R; // Add measurement noise
    
    // Calculate the cross-covariance
    Eigen::MatrixXd Cxz = Eigen::MatrixXd::Zero(n, m);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        Eigen::VectorXd delta_x = sigma_points.col(i) - x;
        Eigen::VectorXd delta_z = h(sigma_points.col(i)) - z_pred;
        Cxz += Wc(i) * delta_x * delta_z.transpose();
    }
    
    // Kalman gain
    Eigen::MatrixXd K = Cxz * S.inverse();
    
    // Update state and covariance
    x += K * (z - z_pred);
    P -= K * S * K.transpose();
    
    predicted = false; // Reset predicted flag after update
}