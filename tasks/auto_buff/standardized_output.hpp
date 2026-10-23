#ifndef POWERBUFF_STANDARDIZED_OUTPUT_HPP
#define POWERBUFF_STANDARDIZED_OUTPUT_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "fivepoints_buff.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff{
    class standardized_output{
        public:
        std::variant<
        std::unique_ptr<YOLO11_BUFF>,
        std::unique_ptr<fivepoints_buff>
        > model_;

        struct standard_Detection{
            std::vector<cv::Point2f>kpt;//前四个为角点，以最上面的点为起点，逆时针；第五个是扇叶中心；第六个是r标点/辅助点
            int label;
            float prob;
        };//yolo11的object和fivepoints的detection都统一到这里来，以后其他的也统一到这里来

        std::vector<standard_Detection> getDetection(cv::Mat&image){
            if(image.empty()){
                return {};
            }

            if(auto* model=std::get_if<std::unique_ptr<YOLO11_BUFF>>(&model_)){
                cv::Mat img=image.clone();
                auto results=(*model)->get_multicandidateboxes(img);
                //auto results=(*model)->get_onecandidatebox(input);

                return convert(results);
            }else if(auto* model=std::get_if<std::unique_ptr<fivepoints_buff>>(&model_)){
                cv::Mat img=image.clone();
                auto results=(*model)->getlastdetection(img);
                
                return convert(results);
            }else{
                return {};
            }
        };

        standardized_output (const std::string & config){
            const auto yaml = YAML::LoadFile(config);
            const std::string model_path = yaml["model"].as<std::string>();

            const std::string filename =std::filesystem::path(model_path).filename().string();

            if (filename.rfind("model-0624", 0) == 0){
                model_ = std::make_unique<fivepoints_buff>(config);
            }
            else if (filename.rfind("yolo11_buff_int8", 0) == 0){
                model_ = std::make_unique<YOLO11_BUFF>(config);
            }
            else{
                throw std::runtime_error("无法根据模型文件名前缀识别模型类型: " + filename);
            }
                };
        private:

        std::vector<standard_Detection> convert(
            const std::vector<YOLO11_BUFF::Object> &results){
                std::vector<standard_Detection> detections;
                detections.reserve(results.size());

                for(const auto& result:results){
                    standard_Detection det;
                    det.kpt=result.kpt;
                    det.label=result.label;
                    det.prob=result.prob;

                    detections.push_back(det);
                }

                return detections;
            };

        std::vector<standard_Detection> convert(
            const std::vector<fivepoints_buff::Detection> &results){
                std::vector<standard_Detection> detections;
                detections.reserve(results.size());

                for(const auto& result:results){
                    standard_Detection det;
                    det.kpt.resize(6);

                    //角点对应的顺序暂定，逆时针
                    det.kpt[0]=result.keypoints[0];
                    det.kpt[1]=result.keypoints[1];
                    det.kpt[2]=result.keypoints[4];
                    det.kpt[3]=result.keypoints[3];
                    det.kpt[4]=(result.keypoints[0]+
                                result.keypoints[1]+
                                result.keypoints[4]+
                                result.keypoints[3])/4;//扇叶中心
                    det.kpt[5]=result.keypoints[2];//旋转中心

                    det.label=result.class_id;
                    det.prob=result.confidence;//还不确定是用confidence还是quality

                    detections.push_back(std::move(det));
                }
                return detections;
            };
        };
}

#endif