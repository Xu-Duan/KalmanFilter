add_executable(test_mannyray test_mannyray.cpp)
target_include_directories(test_mannyray PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_mannyray PRIVATE kalmanFilter)

add_executable(test_martiro test_martiro.cpp)
target_include_directories(test_martiro PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_martiro PRIVATE kalmanFilter)

add_executable(test_EKF test_EKF.cpp)
target_include_directories(test_EKF PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_EKF PRIVATE kalmanFilter)

add_test(NAME test_mannyray COMMAND test_mannyray)
add_test(NAME test_martiro COMMAND test_martiro)
add_test(NAME test_EKF COMMAND test_EKF)