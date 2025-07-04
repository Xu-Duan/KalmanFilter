#include "KalmanFilter.hpp"
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>

using namespace autodiff;

class ExtendedKalmanFilter : public KalmanFilter {
public:
    // Constructor with control input (mimics parent's pattern)
    ExtendedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<VectorXreal(const VectorXreal&, const VectorXreal&)> f_func,
                        std::function<VectorXreal(const VectorXreal&)> h_func);
    
    // Constructor without control input (mimics parent's pattern)
    ExtendedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<VectorXreal(const VectorXreal&)> f_func,
                        std::function<VectorXreal(const VectorXreal&)> h_func);

    void predict(Eigen::VectorXd u = Eigen::VectorXd()) override;
    void update(const Eigen::VectorXd& z) override;

private:
    // Single nonlinear function that handles both cases
    std::function<VectorXreal(const VectorXreal&, const VectorXreal&)> f; // state transition function
    std::function<VectorXreal(const VectorXreal&)> h; // measurement function
    
    // Helper methods to compute Jacobians using autodiff
    Eigen::MatrixXd compute_F_jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& u);
    Eigen::MatrixXd compute_H_jacobian(const Eigen::VectorXd& x);
    
    // Convert between Eigen::VectorXd and VectorXreal
    VectorXreal to_autodiff(const Eigen::VectorXd& v);
    Eigen::VectorXd from_autodiff(const VectorXreal& v);
};