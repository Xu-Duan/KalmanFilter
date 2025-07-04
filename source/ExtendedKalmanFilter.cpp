#include "ExtendedKalmanFilter.hpp"
#include <iostream>

// Constructor with control input - follows parent's pattern
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    std::function<VectorXreal(const VectorXreal&, const VectorXreal&)> f_func,
    std::function<VectorXreal(const VectorXreal&)> h_func){
    n = Q.rows();
    m = R.rows();
    this->R = R;
    this->Q = Q;
    f = f_func; // Store the nonlinear state transition function
    h = h_func; // Store the nonlinear measurement function
    // n_u will be > 0 because we passed a B matrix
}

// Constructor without control input - follows parent's pattern
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    std::function<VectorXreal(const VectorXreal&)> f_no_control,
    std::function<VectorXreal(const VectorXreal&)> h_func){
    n = Q.rows();
    m = R.rows();
    this->R = R;
    this->Q = Q;
    h = h_func; // Store the nonlinear measurement function
    f = [f_no_control](const VectorXreal& x, const VectorXreal& u) -> VectorXreal {
        return f_no_control(x);
    };
    // n_u will be 0 because we didn't pass a B matrix
}

void ExtendedKalmanFilter::predict(Eigen::VectorXd u) {
    // Handle control input like parent class
    if (n_u == 0 && u.size() > 0) {
        std::cout << "ERROR: ExtendedKalmanFilter::predict: No control input expected" << std::endl;
        return;
    }

    if (n_u > 0 && u.size() == 0) {
        u = Eigen::VectorXd::Zero(n_u);
    }

    // Nonlinear state prediction: x_pred = f(x, u)
    VectorXreal x_ad = to_autodiff(x);
    VectorXreal u_ad = (n_u > 0) ? to_autodiff(u) : VectorXreal();
    
    Eigen::VectorXd x_pred = from_autodiff(f(x_ad, u_ad));
    
    // Compute Jacobian F = ∂f/∂x at current state
    Eigen::MatrixXd F = compute_F_jacobian(x, u);
    
    // Update state estimate
    x = x_pred;
    
    // Update covariance: P = F * P * F^T + Q
    P = F * P * F.transpose() + Q;
    predicted = true;
}

void ExtendedKalmanFilter::update(const Eigen::VectorXd& z) {
    if (!predicted) {
        std::cout << "ExtendedKalmanFilter::update: Predicted state not available" << std::endl;
        std::cout << "WARNING: Predicting state using zero control input" << std::endl;
        Eigen::VectorXd zero_u;
        this->predict(zero_u);
    }

    // Nonlinear measurement prediction: z_pred = h(x)
    VectorXreal x_ad = to_autodiff(x);
    Eigen::VectorXd z_pred = from_autodiff(h(x_ad));
    
    // Compute Jacobian H = ∂h/∂x at current state
    Eigen::MatrixXd H_jac = compute_H_jacobian(x);
    
    // Standard Kalman update equations with computed Jacobians
    Eigen::VectorXd y = z - z_pred;
    Eigen::MatrixXd S = H_jac * P * H_jac.transpose() + R;
    Eigen::MatrixXd K = P * H_jac.transpose() * S.inverse();
    
    // Update state and covariance
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(n, n) - K * H_jac) * P;
    predicted = false;
}

Eigen::MatrixXd ExtendedKalmanFilter::compute_F_jacobian(const Eigen::VectorXd& x_val, const Eigen::VectorXd& u_val) {
    VectorXreal x_ad = to_autodiff(x_val);
    VectorXreal u_ad = (n_u > 0 && u_val.size() > 0) ? to_autodiff(u_val) : VectorXreal();
    
    // Create a lambda that captures u_ad and only depends on x
    auto f_x = [this, &u_ad](const VectorXreal& x) -> VectorXreal {
        return f(x, u_ad);
    };
    Eigen::MatrixXd f_x_jacobian;
    VectorXreal f_val;
    jacobian(f_x, wrt(x_ad), at(x_ad), f_val, f_x_jacobian);
    return f_x_jacobian;
}

Eigen::MatrixXd ExtendedKalmanFilter::compute_H_jacobian(const Eigen::VectorXd& x_val) {
    VectorXreal x_ad = to_autodiff(x_val);
    Eigen::MatrixXd H_jacobian;
    VectorXreal h_val;
    jacobian(h, wrt(x_ad), at(x_ad), h_val, H_jacobian);
    return H_jacobian;
}

VectorXreal ExtendedKalmanFilter::to_autodiff(const Eigen::VectorXd& v) {
    VectorXreal result(v.size());
    for (int i = 0; i < v.size(); ++i) {
        result[i] = v[i];
    }
    return result;
}

Eigen::VectorXd ExtendedKalmanFilter::from_autodiff(const VectorXreal& v) {
    Eigen::VectorXd result(v.size());
    for (int i = 0; i < v.size(); ++i) {
        result[i] = val(v[i]); // Extract the value from autodiff::real
    }
    return result;
}