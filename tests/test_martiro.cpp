// test_martiro.cpp
// https://github.com/hmartiro/kalman-cpp/blob/master/kalman-test.cpp

#include <iostream>
#include "KalmanFilter.hpp"


Eigen::VectorXd exact_states[] = {
    (Eigen::VectorXd(3) << 1.04203, -0.327, -9.81).finished(),
    (Eigen::VectorXd(3) << 1.08708, 0.185739, -9.80886).finished(),
    (Eigen::VectorXd(3) << 1.21726, 1.1065, -9.80545).finished(),
    (Eigen::VectorXd(3) << 1.37662, 1.70821, -9.79885).finished(),
    (Eigen::VectorXd(3) << 1.56866, 2.20488, -9.78506).finished(),
    (Eigen::VectorXd(3) << 1.6825, 2.08472, -9.77798).finished(),
    (Eigen::VectorXd(3) << 1.8836, 2.33911, -9.74143).finished(),
    (Eigen::VectorXd(3) << 2.0217, 2.2488, -9.71669).finished(),
    (Eigen::VectorXd(3) << 2.11578, 1.99214, -9.70565).finished(),
    (Eigen::VectorXd(3) << 2.19069, 1.69607, -9.69904).finished(),
    (Eigen::VectorXd(3) << 2.3334, 1.63238, -9.61208).finished(),
    (Eigen::VectorXd(3) << 2.45705, 1.50983, -9.52416).finished(),
    (Eigen::VectorXd(3) << 2.50951, 1.1982, -9.52086).finished(),
    (Eigen::VectorXd(3) << 2.59713, 1.00888, -9.43237).finished(),
    (Eigen::VectorXd(3) << 2.68559, 0.839945, -9.31366).finished(),
    (Eigen::VectorXd(3) << 2.74987, 0.625611, -9.22392).finished(),
    (Eigen::VectorXd(3) << 2.80713, 0.415231, -9.12306).finished(),
    (Eigen::VectorXd(3) << 2.82256, 0.11541, -9.11823).finished(),
    (Eigen::VectorXd(3) << 2.86204, -0.0909276, -9.00104).finished(),
    (Eigen::VectorXd(3) << 2.914, -0.237813, -8.80868).finished(),
    (Eigen::VectorXd(3) << 2.89637, -0.558878, -8.84423).finished(),
    (Eigen::VectorXd(3) << 2.86967, -0.876857, -8.87484).finished(),
    (Eigen::VectorXd(3) << 2.8097, -1.26203, -8.99406).finished(),
    (Eigen::VectorXd(3) << 2.74668, -1.62325, -9.07626).finished(),
    (Eigen::VectorXd(3) << 2.71797, -1.85087, -8.97629).finished(),
    (Eigen::VectorXd(3) << 2.688, -2.05625, -8.85203).finished(),
    (Eigen::VectorXd(3) << 2.67404, -2.18991, -8.64065).finished(),
    (Eigen::VectorXd(3) << 2.62113, -2.41872, -8.56416).finished(),
    (Eigen::VectorXd(3) << 2.51844, -2.76889, -8.6464).finished(),
    (Eigen::VectorXd(3) << 2.3864, -3.17278, -8.79086).finished(),
    (Eigen::VectorXd(3) << 2.28523, -3.45258, -8.77464).finished(),
    (Eigen::VectorXd(3) << 2.13925, -3.83319, -8.88051).finished(),
    (Eigen::VectorXd(3) << 1.99756, -4.16841, -8.92665).finished(),
    (Eigen::VectorXd(3) << 1.85172, -4.48514, -8.94876).finished(),
    (Eigen::VectorXd(3) << 1.66611, -4.88236, -9.06039).finished(),
    (Eigen::VectorXd(3) << 1.57765, -4.98385, -8.83891).finished(),
    (Eigen::VectorXd(3) << 1.36503, -5.40203, -8.97247).finished(),
    (Eigen::VectorXd(3) << 1.16943, -5.74174, -9.01545).finished(),
    (Eigen::VectorXd(3) << 0.965763, -6.07385, -9.04817).finished(),
    (Eigen::VectorXd(3) << 0.746617, -6.41768, -9.09098).finished(),
    (Eigen::VectorXd(3) << 0.53849, -6.70629, -9.07666).finished(),
    (Eigen::VectorXd(3) << 0.322791, -6.98964, -9.058).finished(),
    (Eigen::VectorXd(3) << 0.0427047, -7.40495, -9.16592).finished(),
    (Eigen::VectorXd(3) << -0.219931, -7.7479, -9.20081).finished(),
    (Eigen::VectorXd(3) << -0.501726, -8.10938, -9.25087).finished()
};

bool vector_equal(Eigen::VectorXd a, Eigen::VectorXd b, double tolerance) {
    for (int i = 0; i < a.rows(); i++) {
        if (std::abs(a(i) - b(i)) > tolerance) {
            return false;
        }
    }
    return true;
}


int main() {
    int n = 3; // number of states
    int m = 1; // number of measurements
    double dt = 1.0/30; // time step
    Eigen::MatrixXd A(n, n);
    Eigen::MatrixXd H(m, n);
    Eigen::MatrixXd R(m, m);
    Eigen::MatrixXd Q(n, n);
    Eigen::MatrixXd P(n, n);
    A << 1, dt, 0, 0, 1, dt, 0, 0, 1;
    H << 1, 0, 0;
    R << 5;
    Q << 0.05, 0.05, .0, .05, .05, 0, 0, 0, 0;
    P << .1, .1, .1, .1, 10000, 10, .1, 10, 100;

    // Initialize the Kalman filter
    KalmanFilter kf(A, H, R, Q);
    Eigen::VectorXd x(n);
    // List of noisy position measurements (y)
    std::vector<double> measurements = {
        1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
        1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
        2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
        2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
        2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
        2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
        2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
        1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
        0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
    };
    x << measurements[0], 0, -9.81;
    kf.init(x, P);
    for (int i = 0; i < measurements.size(); i++) {
        kf.predict();
        Eigen::VectorXd z = Eigen::VectorXd::Zero(m);
        z(0) = measurements[i];
        kf.update(z);
        double tolerance = 1e-4;
        assert(vector_equal(kf.get_state(), exact_states[i], tolerance));
    }
    return 0;
}