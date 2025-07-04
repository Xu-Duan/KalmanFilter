add_library(kalmanFilter KalmanFilter.cpp 
                            ExtendedKalmanFilter.cpp)

target_include_directories(kalmanFilter PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(kalmanFilter PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_include_directories(kalmanFilter PUBLIC ${AUTODIFF_INCLUDE_DIRS})