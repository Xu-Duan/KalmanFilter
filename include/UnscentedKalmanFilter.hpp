#include "KalmanFilter.hpp"
#include <functional>

class UnscentedKalmanFilter : public FilterBase {
public:
    UnscentedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f_func,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_func);
    UnscentedKalmanFilter(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f_func,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_func);
    
    void predict(Eigen::VectorXd u = Eigen::VectorXd()) override;
    void update(const Eigen::VectorXd& z) override;
    void init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P = Eigen::MatrixXd()) override;
private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f; // state transition function
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h; // measurement function
    // UKF parameters
    double alpha;  // spread of sigma points
    double beta;   // parameter to incorporate prior knowledge
    double kappa;  // secondary scaling parameter
    double lambda; // composite scaling parameter
    
    // Weights for mean and covariance
    Eigen::VectorXd Wm; // weights for means
    Eigen::VectorXd Wc; // weights for covariance
    
    // Sigma points
    Eigen::MatrixXd sigma_points;
    
    // Helper functions
    void generateSigmaPoints();
    void calculateWeights();
    Eigen::MatrixXd calculateSquareRoot(const Eigen::MatrixXd& matrix);
};