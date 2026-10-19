#include "fivepoints_buff.hpp"

#include "tools/logger.hpp"

namespace fs = std::filesystem;

namespace auto_buff{

    fivepoints_buff::fivepoints_buff(const std::string&config){
        auto yaml = YAML::LoadFile(config);
        std::string model_path = yaml["model"].as<std::string>();
        std::string device = "AUTO";
        int num_classes_override = -1;
        int num_keypoints_override = -1;

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        class_names_ = read_class_names_from_model(model);
        const auto model_input_shape = model->input().get_shape();

        if (model_input_shape.size() != 4)
        {
            throw std::runtime_error("模型输入非4维 (NCHW)");
        }
        input_height_ = static_cast<int>(model_input_shape[2]);
        input_width_ = static_cast<int>(model_input_shape[3]);

        // PrePostProcessor 配置 (BGR u8 NHWC -> RGB f32 [0,1] NCHW)
        auto ppp = ov::preprocess::PrePostProcessor(model);
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0f);
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();

        compiled_model_ = core.compile_model(model, device,
                                             ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        infer_request_ = compiled_model_.create_infer_request();

        const auto output_shape = compiled_model_.output().get_shape();
        if (output_shape.size() != 3)
        {
            throw std::runtime_error("模型输出维度非3维 ([N,C,A] 或 [N,A,C])");
        }

        // 识别输出布局 [N,C,A] 或 [N,A,C]
        if (output_shape[1] <= 256 && output_shape[2] >= 100)
        {
            output_layout_nca_ = true;
            output_channels_ = static_cast<int>(output_shape[1]);
            num_anchors_ = static_cast<int>(output_shape[2]);
        }
        else
        {
            output_layout_nca_ = false;
            num_anchors_ = static_cast<int>(output_shape[1]);
            output_channels_ = static_cast<int>(output_shape[2]);
        }

        resolve_head_dims(num_classes_override, num_keypoints_override);
        ensure_class_names();
    };

    void fivepoints_buff::preprocess_letterbox(const cv::Mat &src, cv::Mat &dst, float &scale, int &pad_w, int &pad_h)
    {
        scale = std::min(static_cast<float>(input_width_) / src.cols,
                         static_cast<float>(input_height_) / src.rows);
        int new_w = static_cast<int>(std::round(src.cols * scale));
        int new_h = static_cast<int>(std::round(src.rows * scale));

        pad_w = (input_width_ - new_w) / 2;
        pad_h = (input_height_ - new_h) / 2;
        int right = input_width_ - new_w - pad_w;
        int bottom = input_height_ - new_h - pad_h;

        cv::Mat resized;
        if (new_w != src.cols || new_h != src.rows)
            cv::resize(src, resized, cv::Size(new_w, new_h));
        else
            resized = src;

        if (pad_w > 0 || pad_h > 0 || right > 0 || bottom > 0)
            cv::copyMakeBorder(resized, dst, pad_h, bottom, pad_w, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        else
            dst = resized;
    }

    std::vector<fivepoints_buff::Detection> fivepoints_buff::infer(const cv::Mat &image, TimingRecord &rec)
    {
        float scale = 1.0f;
        int pad_w = 0, pad_h = 0;
        cv::Mat preprocessed_img;

        // 1. 预处理
        auto t0 = std::chrono::high_resolution_clock::now();
        preprocess_letterbox(image, preprocessed_img, scale, pad_w, pad_h);
        ov::Tensor input_tensor(ov::element::u8,
                                {1, (size_t)input_height_, (size_t)input_width_, 3},
                                preprocessed_img.data);
        infer_request_.set_input_tensor(input_tensor);
        auto t1 = std::chrono::high_resolution_clock::now();
        rec.preprocess_time = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 2. 推理
        infer_request_.infer();
        auto t2 = std::chrono::high_resolution_clock::now();
        rec.infer_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // 3. 后处理
        auto results = postprocess(scale, pad_w, pad_h, image.cols, image.rows);
        auto t3 = std::chrono::high_resolution_clock::now();
        rec.postprocess_time = std::chrono::duration<float, std::milli>(t3 - t2).count();
        rec.detections = static_cast<int>(results.size());

        return results;
    }

    std::vector<fivepoints_buff::Detection> fivepoints_buff::postprocess(float scale, int pad_w, int pad_h, int orig_w, int orig_h)
    {
        std::vector<fivepoints_buff::Detection> detections;
        detections.reserve(64);
        const float *output_data = infer_request_.get_output_tensor().data<float>();

        auto get_val = [&](int c, int a) -> float {
            return output_layout_nca_ ? output_data[c * num_anchors_ + a]
                                      : output_data[a * output_channels_ + c];
        };

        for (int i = 0; i < num_anchors_; ++i)
        {
            int best_cls = -1;
            float max_conf = 0.0f;
            for (int c = 0; c < num_classes_; ++c)
            {
                float score = get_val(c, i);
                if (score > max_conf)
                {
                    max_conf = score;
                    best_cls = c;
                }
            }

            if (best_cls < 0 || max_conf < conf_threshold_)
                continue;

            fivepoints_buff::Detection det;
            det.class_id = best_cls;
            det.confidence = max_conf;
            det.keypoints.reserve(num_keypoints_);
            det.kpt_confs.reserve(num_keypoints_);

            int valid_kpts = 0;
            float kpt_conf_sum = 0.0f;
            cv::Point2f valid_center(0.0f, 0.0f);
            cv::Point2f all_center(0.0f, 0.0f);
            bool has_negative_kpt = false;

            for (int k = 0; k < num_keypoints_; ++k)
            {
                int base = num_classes_ + k * kpt_dim_;
                float x = (get_val(base, i) - pad_w) / scale;
                float y = (get_val(base + 1, i) - pad_h) / scale;
                float k_conf = get_val(base + 2, i);

                if (x < 0.0f || y < 0.0f)
                {
                    has_negative_kpt = true;
                    break;
                }

                x = std::clamp(x, 0.0f, static_cast<float>(orig_w - 1));
                y = std::clamp(y, 0.0f, static_cast<float>(orig_h - 1));

                cv::Point2f pt(x, y);
                det.keypoints.push_back(pt);
                det.kpt_confs.push_back(k_conf);
                all_center += pt;

                if (k_conf >= kpt_conf_threshold_)
                {
                    valid_kpts++;
                    kpt_conf_sum += k_conf;
                    valid_center += pt;
                }
            }

            if (has_negative_kpt || valid_kpts < std::min(min_valid_kpts_, num_keypoints_))
                continue;

            float mean_kpt_conf = valid_kpts > 0 ? (kpt_conf_sum / valid_kpts) : 0.0f;
            det.quality = det.confidence * mean_kpt_conf;
            det.center = (valid_kpts > 0) ? (valid_center * (1.0f / valid_kpts))
                                          : (all_center * (1.0f / num_keypoints_));

            detections.push_back(std::move(det));
        }

        return nms(detections);
    }

    std::vector<fivepoints_buff::Detection> fivepoints_buff::nms(std::vector<fivepoints_buff::Detection> &detections)
    {
        if (detections.empty())
            return {};

        std::sort(detections.begin(), detections.end(),
                  [](const fivepoints_buff::Detection &a, const fivepoints_buff::Detection &b) { return a.quality > b.quality; });

        std::vector<fivepoints_buff::Detection> result;
        std::vector<bool> suppressed(detections.size(), false);
        const float dist_thresh_sq = nms_dist_threshold_ * nms_dist_threshold_;

        for (size_t i = 0; i < detections.size(); ++i)
        {
            if (suppressed[i])
                continue;
            result.push_back(detections[i]);
            const auto &c_i = detections[i].center;

            for (size_t j = i + 1; j < detections.size(); ++j)
            {
                if (suppressed[j])
                    continue;
                float dx = c_i.x - detections[j].center.x;
                float dy = c_i.y - detections[j].center.y;
                if ((dx * dx + dy * dy) < dist_thresh_sq)
                    suppressed[j] = true;
            }
        }
        return result;
    }

    void fivepoints_buff::draw(cv::Mat &image, const std::vector<Detection> &detections) const
    {
        for (const auto &det : detections)
        {
            for (size_t k = 0; k < det.keypoints.size(); ++k)
            {
                if (det.kpt_confs[k] >= kpt_conf_threshold_)
                {
                    cv::Scalar color = KPT_COLORS[k % KPT_COLORS.size()];
                    cv::circle(image, det.keypoints[k], 3, color, -1);
                    cv::circle(image, det.keypoints[k], 3, cv::Scalar(255, 255, 255), 1);
                }
            }

            std::string cls_name = (det.class_id >= 0 && det.class_id < (int)class_names_.size())
                                       ? class_names_[det.class_id]
                                       : ("class" + std::to_string(det.class_id));
            std::string label = cls_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(image,
                          cv::Point(det.center.x - 2, det.center.y - text_size.height - 4),
                          cv::Point(det.center.x + text_size.width + 2, det.center.y + 2),
                          cv::Scalar(0, 0, 0), -1);
            cv::putText(image, label, cv::Point(det.center.x, det.center.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

    float fivepoints_buff::warmup(const cv::Mat &image, int iterations = 10)
    {
        if (iterations <= 0)
            return 0.0f;
        std::cout << "[Info] 预热 " << iterations << " 次...\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        TimingRecord rec;
        for (int i = 0; i < iterations; ++i)
            infer(image, rec);
        auto t2 = std::chrono::high_resolution_clock::now();
        float total_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
        std::cout << "[Info] 预热完成, 平均耗时: " << (total_ms / iterations) << " ms\n";
        return total_ms;
    }

    std::vector<std::string> fivepoints_buff::read_class_names_from_model(const std::shared_ptr<ov::Model> &model) const
    {
        try
        {
            if (model && model->has_rt_info("model_info", "labels"))
            {
                auto labels = model->get_rt_info<std::vector<std::string>>("model_info", "labels");
                for (auto &name : labels)
                    std::replace(name.begin(), name.end(), '_', ' ');
                return labels;
            }
        }
        catch (...)
        {
        }
        return {};
    }

    void fivepoints_buff::resolve_head_dims(int num_classes_override, int num_keypoints_override)
    {
        if (num_classes_override > 0 && num_keypoints_override > 0)
        {
            if (num_classes_override + num_keypoints_override * kpt_dim_ != output_channels_)
                throw std::runtime_error("参数 --nc 与 --nk 组合与模型输出通道不匹配");
            num_classes_ = num_classes_override;
            num_keypoints_ = num_keypoints_override;
            return;
        }

        if (num_classes_override > 0)
        {
            int remain = output_channels_ - num_classes_override;
            if (remain <= 0 || remain % kpt_dim_ != 0)
                throw std::runtime_error("参数 --nc 与输出通道不匹配");
            num_classes_ = num_classes_override;
            num_keypoints_ = remain / kpt_dim_;
            return;
        }

        if (num_keypoints_override > 0)
        {
            int nc = output_channels_ - num_keypoints_override * kpt_dim_;
            if (nc <= 0)
                throw std::runtime_error("参数 --nk 与输出通道不匹配");
            num_classes_ = nc;
            num_keypoints_ = num_keypoints_override;
            return;
        }

        if (!class_names_.empty())
        {
            int nc = static_cast<int>(class_names_.size());
            int remain = output_channels_ - nc;
            if (remain > 0 && remain % kpt_dim_ == 0)
            {
                num_classes_ = nc;
                num_keypoints_ = remain / kpt_dim_;
                return;
            }
        }

        // 默认匹配能量机关 3 类别 + 5 关键点 (3 + 5*3 = 18 通道)
        if (output_channels_ == 18)
        {
            num_classes_ = 3;
            num_keypoints_ = 5;
            return;
        }

        throw std::runtime_error("无法自动推断类别和关键点数，请使用 --nc 或 --nk 参数指定");
    }

    void fivepoints_buff::ensure_class_names()
    {
        if ((int)class_names_.size() == num_classes_)
            return;
        class_names_.clear();
        if (num_classes_ == 3)
        {
            class_names_ = {"inactive", "small_activated", "big_activated"};
        }
        else
        {
            class_names_.reserve(num_classes_);
            for (int i = 0; i < num_classes_; ++i)
                class_names_.push_back("class" + std::to_string(i));
        }
    }

    std::vector<fivepoints_buff::Detection>  fivepoints_buff::getlastdetection(cv::Mat& img){
        if (img.empty()) {
            tools::logger()->warn("Empty img!, camera drop!");
            return std::vector<fivepoints_buff::Detection> ();
        }
        const int64 start = cv::getTickCount();

        TimingRecord rec;
        std::vector<Detection> results;

        results=infer(img,rec);
        draw(img,results);
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        cv::putText(
        img, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
        cv::Scalar(255, 0, 0), 2, 8);

        return results;
    };

}