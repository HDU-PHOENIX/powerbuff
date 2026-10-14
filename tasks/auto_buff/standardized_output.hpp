#ifndef POWERBUFF_STANDARDIZED_OUTPUT_HPP
#define POWERBUFF_STANDARDIZED_OUTPUT_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

#include "fivepoints_buff.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff{
    class standardized_output{
        struct standard_Detection{
            std::vector<cv::Point2f>kpt;//前四个为角点，
            int label=-1;
        };//yolo11的object和fivepoints的detection都统一到这里来，以后其他的也统一到这里来
    };
}

#endif