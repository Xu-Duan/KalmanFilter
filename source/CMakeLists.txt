add_library(kalmanFilter KalmanFilter.cpp)

target_include_directories(kalmanFilter PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kalmanFilter PRIVATE Eigen3::Eigen)