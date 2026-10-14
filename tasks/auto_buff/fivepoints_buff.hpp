#ifndef POWERBUFF_FIVEPOINTS_BUFF_HPP
#define POWERBUFF_FIVEPOINTS_BUFF_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace auto_buff{
    const int DEFAULT_KPT_DIM = 3;
    const float DEFAULT_CONF_THRESHOLD = 0.8f;
    const float DEFAULT_KPT_CONF_THRESHOLD = 0.8f;
    const float DEFAULT_NMS_DIST_THRESHOLD = 30.0f;
    const int DEFAULT_MIN_VALID_KPTS = 3;

    const std::vector<cv::Scalar> KPT_COLORS = {
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 0, 255)};

    const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
    const std::vector<std::string> VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"};

    
    
    class fivepoints_buff{
        public:
        
        struct Detection // FivePoints 模型的单个目标检测结果
        {
            int class_id = -1;
            float confidence = 0.0f;
            float quality = 0.0f;
            cv::Point2f center{0.0f, 0.0f};
            std::vector<cv::Point2f> keypoints;//好诡异阿，怎么center是2,交点0341（顺时针）的顺序阿？
            std::vector<float> kpt_confs;
        };
    
        
        struct TimingRecord  // 单张图像或单帧视频的处理耗时记录
        {
            std::string name;
            int detections = 0;
            float preprocess_time = 0.0f;
            float infer_time = 0.0f;
            float postprocess_time = 0.0f;
            float draw_time = 0.0f;
            float save_time = 0.0f;
            float total_time = 0.0f;
        };

        struct PerformanceStats  // 模型推理测试的整体性能统计信息
        {
            int total_images = 0;
            int total_detections = 0;
            int warmup_iterations = 0;
            float warmup_total_time_ms = 0.0f;
            std::vector<TimingRecord> records;

            void add_record(const TimingRecord &rec)
            {
                records.push_back(rec);
                total_images++;
                total_detections += rec.detections;
            }

            template <typename F>
            std::vector<float> extract(F &&getter) const
            {
                std::vector<float> res;
                res.reserve(records.size());
                for (const auto &r : records)
                    res.push_back(getter(r));
                return res;
            }

            static float mean(const std::vector<float> &v)
            {
                return v.empty() ? 0.0f : std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
            }

            static float stddev(const std::vector<float> &v)
            {
                if (v.size() < 2)
                    return 0.0f;
                float m = mean(v);
                float sum = 0.0f;
                for (float x : v)
                    sum += (x - m) * (x - m);
                return std::sqrt(sum / v.size());
            }

            static float min_val(const std::vector<float> &v)
            {
                return v.empty() ? 0.0f : *std::min_element(v.begin(), v.end());
            }

            static float max_val(const std::vector<float> &v)
            {
                return v.empty() ? 0.0f : *std::max_element(v.begin(), v.end());
            }

            static float percentile(std::vector<float> v, float p)
            {
                if (v.empty())
                    return 0.0f;
                std::sort(v.begin(), v.end());
                size_t idx = static_cast<size_t>((p / 100.0f) * (v.size() - 1));
                return v[idx];
            }
        };


            fivepoints_buff(const std::string &model_path, const std::string &device = "AUTO",
                            int num_classes_override = -1, int num_keypoints_override = -1,
                            float conf_threshold = DEFAULT_CONF_THRESHOLD,
                            float kpt_conf_threshold = DEFAULT_KPT_CONF_THRESHOLD,
                            float nms_dist_threshold = DEFAULT_NMS_DIST_THRESHOLD,
                            int min_valid_kpts = DEFAULT_MIN_VALID_KPTS)
                : conf_threshold_(conf_threshold),
                kpt_conf_threshold_(kpt_conf_threshold),
                nms_dist_threshold_(nms_dist_threshold),
                min_valid_kpts_(min_valid_kpts){}

            //对输入图像进行等比例缩放和填充，使其符合模型的输入尺寸，并记录缩放比例与填充量。
            void preprocess_letterbox(const cv::Mat &src, cv::Mat &dst, float &scale, int &pad_w, int &pad_h){}
            
            std::vector<Detection> infer(const cv::Mat &image, TimingRecord &rec){}//执行完整的模型推理流程，包括图像预处理、模型推理和检测结果后处理，并记录耗时。
            std::vector<Detection> postprocess(float scale, int pad_w, int pad_h, int orig_w, int orig_h){}//解析模型输出，将预测结果转换为原始图像坐标系下的检测结果。
            std::vector<Detection> nms(std::vector<Detection> &detections){}//通过非极大值抑制（NMS）去除重叠度较高的冗余检测框，保留较优的检测结果。
            void draw(cv::Mat &image, const std::vector<Detection> &detections) const{}//将检测结果绘制到输入图像上，用于可视化检测框、类别等信息。
            float warmup(const cv::Mat &image, int iterations = 10){}//使用输入图像重复执行指定次数的推理，以预热模型并测量预热阶段的平均耗时。

        private:
            std::vector<std::string> read_class_names_from_model(const std::shared_ptr<ov::Model> &model) const{}//从 OpenVINO 模型的元数据中读取类别名称，并返回类别名称列表。

            void resolve_head_dims(int num_classes_override, int num_keypoints_override){}//根据模型输出维度及用户指定的参数，确定检测头中的类别数量和关键点数量。
            void ensure_class_names(){}//检查类别名称是否已正确初始化，并在缺失时尝试补全或生成默认类别名称。

            ov::CompiledModel compiled_model_;
            ov::InferRequest infer_request_;
            int input_width_ = 640;
            int input_height_ = 640;
            int output_channels_ = 0;
            int num_anchors_ = 0;
            int num_classes_ = 0;
            int num_keypoints_ = 0;
            int kpt_dim_ = DEFAULT_KPT_DIM;
            float conf_threshold_ = DEFAULT_CONF_THRESHOLD;
            float kpt_conf_threshold_ = DEFAULT_KPT_CONF_THRESHOLD;
            float nms_dist_threshold_ = DEFAULT_NMS_DIST_THRESHOLD;
            int min_valid_kpts_ = DEFAULT_MIN_VALID_KPTS;
            std::vector<std::string> class_names_;
            bool output_layout_nca_ = true;
        };
}
#endif