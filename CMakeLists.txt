cmake_minimum_required(VERSION 3.16)

project(powerbuff LANGUAGES CXX)

# ============================================================
# C++ 编译标准
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# 查找第三方依赖
# ============================================================

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# ============================================================
# 项目公共头文件路径
# ============================================================

# 允许使用：
# #include "tasks/auto_buff/buff_detector.hpp"
# #include "tools/logger.hpp"

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# 构建项目内部库
# ============================================================

add_subdirectory(tools)
add_subdirectory(tasks/auto_buff)

# ============================================================
# auto_buff_test：能量机关测试程序
# ============================================================

add_executable(
    auto_buff_test
    tests/auto_buff_test.cpp
)

target_link_libraries(
    auto_buff_test
    PRIVATE
        auto_buff
        tools
        ${OpenCV_LIBS}
        openvino::runtime
        Eigen3::Eigen
        Ceres::ceres
        yaml-cpp
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# ============================================================
# fivepoints_infer：参考推理程序
# ============================================================

add_executable(
    fivepoints_infer
    infer/fivepoints_infer.cpp
)

target_include_directories(
    fivepoints_infer
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    fivepoints_infer
    PRIVATE
        openvino::runtime
        ${OpenCV_LIBS}
)
