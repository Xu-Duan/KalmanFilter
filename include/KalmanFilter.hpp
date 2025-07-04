#pragma once
#include "FilterBase.hpp"

class KalmanFilter : public FilterBase {
public:
    KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, 
                 const Eigen::MatrixXd& H, const Eigen::MatrixXd& R, 
                 const Eigen::MatrixXd& Q);
    KalmanFilter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& H, 
                 const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q);

    //void init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P = Eigen::MatrixXd()) override;
    void predict(Eigen::VectorXd u = Eigen::VectorXd()) override;
    void update(const Eigen::VectorXd& z) override;
    //Eigen::VectorXd get_state() const override;

private:
    Eigen::MatrixXd A, B, H; // These matrices are meaningful for linear KF
};