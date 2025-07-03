cmake_minimum_required(VERSION 3.10)
project(kalmanFilter)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(autodiff REQUIRED)

enable_testing()
add_subdirectory(source)
add_subdirectory(tests)
