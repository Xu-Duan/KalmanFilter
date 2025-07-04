#pragma once
#include <Eigen/Dense>

class FilterBase {
public:
    virtual ~FilterBase() = default;
    void init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P = Eigen::MatrixXd()){
        this->x = x;
        this->P = (P.size() == 0) ? Eigen::MatrixXd::Identity(n, n) : P;
        predicted = false;
    };
    virtual void predict(Eigen::VectorXd u = Eigen::VectorXd()) = 0;
    virtual void update(const Eigen::VectorXd& z) = 0;
    Eigen::VectorXd get_state(){
        return this->x;
    }

protected:
    int n, m, n_u;
    Eigen::VectorXd x;
    Eigen::MatrixXd P, Q, R;
    bool predicted = false;
};