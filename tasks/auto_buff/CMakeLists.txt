# ============================================================
# 能量机关算法静态库
# ============================================================

add_library(
    auto_buff STATIC
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    yolo11_buff.cpp
    fivepoints_buff.cpp
)

# ============================================================
# 头文件搜索路径
# ============================================================

target_include_directories(
    auto_buff
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# 算法库依赖
# ============================================================

target_link_libraries(
    auto_buff
    PUBLIC
        tools
        io
        openvino::runtime
        Eigen3::Eigen
        Ceres::ceres
        yaml-cpp
        ${OpenCV_LIBS}
)
