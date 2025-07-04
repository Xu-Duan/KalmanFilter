#pragma once
#include "FilterBase.hpp"
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>

using namespace autodiff;

class ExtendedKalmanFilter : public FilterBase {
public:
    // Constructor with control input
    ExtendedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<VectorXreal(const VectorXreal&, const VectorXreal&)> f_func,
                        std::function<VectorXreal(const VectorXreal&)> h_func);
    
    // Constructor without control input
    ExtendedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<VectorXreal(const VectorXreal&)> f_func,
                        std::function<VectorXreal(const VectorXreal&)> h_func);

    //void init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P = Eigen::MatrixXd()) override;
    void predict(Eigen::VectorXd u = Eigen::VectorXd()) override;
    void update(const Eigen::VectorXd& z) override;
    //Eigen::VectorXd get_state() const override;

private:
    std::function<VectorXreal(const VectorXreal&, const VectorXreal&)> f;
    std::function<VectorXreal(const VectorXreal&)> h;
    
    // Helper methods remain the same
    Eigen::MatrixXd compute_F_jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& u);
    Eigen::MatrixXd compute_H_jacobian(const Eigen::VectorXd& x);
    VectorXreal to_autodiff(const Eigen::VectorXd& v);
    Eigen::VectorXd from_autodiff(const VectorXreal& v);
};