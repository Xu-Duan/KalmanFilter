#include <eigen3/Eigen/Dense>

class KalmanFilter {
    public:
        KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd H, 
                    Eigen::MatrixXd R, Eigen::MatrixXd Q);
        KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd H, 
                    Eigen::MatrixXd R, Eigen::MatrixXd Q);
        ~KalmanFilter();

        void init(Eigen::MatrixXd x, Eigen::MatrixXd P = Eigen::MatrixXd());
        void predict(Eigen::MatrixXd u = Eigen::MatrixXd());
        void update(Eigen::MatrixXd z);
        Eigen::MatrixXd get_state() const;

    private:
        int n; // number of states
        int m; // number of measurements
        int n_u; // number of control inputs
        Eigen::MatrixXd x; // state vector
        Eigen::MatrixXd A; // state transition matrix
        Eigen::MatrixXd B; // control matrix
        Eigen::MatrixXd H; // measurement matrix
        Eigen::MatrixXd Q; // state noise covariance matrix
        Eigen::MatrixXd R; // measurement noise covariance matrix
        Eigen::MatrixXd P; // state covariance matrix

        bool predicted = false;
};