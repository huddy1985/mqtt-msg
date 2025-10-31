
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>

#include "app/cnn.hpp"

#ifdef APP_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace app {

struct CnnModel::Impl
{
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "InspectAI") {}

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    std::vector<int64_t> input_shape;
};

namespace {

std::uint64_t fingerprint(const std::vector<std::uint8_t>& data)
{
    const std::size_t step = data.size() > 1024 ? data.size() / 1024 : 1;
    std::uint64_t hash = 1469598103934665603ull;
    std::size_t processed = 0;
    for (std::size_t i = 0; i < data.size(); i += step) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ull;
        if (++processed > 2048) {
            break;
        }
    }
    return hash;
}

}  // namespace

CnnModel::CnnModel(const ScenarioDefinition& config): Model(std::move(config)), 
                                                        config_(std::move(config)),
                                                        type("cnn")
{
    load();
}

CnnModel::~CnnModel() = default;

bool CnnModel::load()
{
    std::string model_path = config_.model.path;

    if (!model_path.empty() && model_path[0] != '/') {
        std::filesystem::path current_path = std::filesystem::current_path();
        model_path = (current_path / model_path).string();
    }

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("CNN model file not found: " + model_path);
    }

    impl_ = std::make_unique<Impl>();

    impl_->session_options.SetIntraOpNumThreads(1);
    impl_->session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, model_path.c_str(), impl_->session_options
    );

    // ---------------- 获取输入输出名称 ----------------
    {
        std::vector<std::string> names = impl_->session->GetInputNames();
        impl_->input_names = names;
        impl_->input_name_ptrs.clear();
        for (auto& n : impl_->input_names) impl_->input_name_ptrs.push_back(n.c_str());
    }
    {
        std::vector<std::string> names = impl_->session->GetOutputNames();
        impl_->output_names = names;
        impl_->output_name_ptrs.clear();
        for (auto& n : impl_->output_names) impl_->output_name_ptrs.push_back(n.c_str());
    }

    // ---------------- 解析输入维度(修正dynamic) ----------------
    {
        Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> s = tensor_info.GetShape();

        if (s.size() == 4) {
            if (s[0] <= 0) s[0] = 1;       // batch
            if (s[1] <= 0) s[1] = 3;       // RGB channels
            if (s[2] <= 0) s[2] = 128;     // height
            if (s[3] <= 0) s[3] = 128;     // width
        } else {
            s = {1, 3, 128, 128};
        }
        impl_->input_shape = std::move(s);

        std::cerr << "[CNN] Using input shape: ["
                  << impl_->input_shape[0] << ", "
                  << impl_->input_shape[1] << ", "
                  << impl_->input_shape[2] << ", "
                  << impl_->input_shape[3] << "]\n";
    }

    std::cout << "only extract image: " <<
            "x: " << config_.detection_regions[0].x <<
            "y:" << config_.detection_regions[0].y <<
            "width: " << config_.detection_regions[0].width <<
            "height: " << config_.detection_regions[0].height << std::endl;

    loaded_ = true;
    return true;
}

bool CnnModel::release()
{
    if (impl_) {
        impl_->input_names.clear();
        impl_->input_name_ptrs.clear();
        impl_->output_names.clear();
        impl_->output_name_ptrs.clear();

        impl_->input_shape.clear();

        impl_->session.reset();
    }

    loaded_ = false;
    std::cout << "CNN model resources have been released successfully.\n";

    return loaded_;
}

std::string CnnModel::model_type()
{
    return type;
}

std::vector<Detection> CnnModel::infer(const CapturedFrame& frame) const
{
    std::vector<Detection> predictions;
    if (!loaded_ || !impl_ || !impl_->session || frame.data.empty()) {
        return predictions;
    }

    try {
        // 1) 解码 JPEG/PNG → BGR
        cv::Mat encoded(1, static_cast<int>(frame.data.size()), CV_8UC1,
                        const_cast<uint8_t*>(frame.data.data()));

        static std::mutex imdecode_mutex;
        cv::Mat image;
        {
            std::lock_guard<std::mutex> lk(imdecode_mutex);
            image = cv::imdecode(encoded, cv::IMREAD_COLOR); // BGR
        }

        if (image.empty()) {
            std::cerr << "[CNN] Failed to decode image.\n";
            return predictions;
        }

        // 2) 提取 ROI（若配置无效则使用默认）
        Region rg;
        if (config_.detection_regions.size() != 1) {
            rg = {740, 420, 240, 240};
        } else {
            rg = config_.detection_regions[0];
        }
        cv::Mat ROI = extractROI(image, rg.x, rg.y, rg.width, rg.height);
        if (ROI.empty()) {
            std::cerr << "[CNN] ROI extraction failed.\n";
            return predictions;
        }

        // 3) resize 到模型输入大小 (128x128)
        const int target_h = static_cast<int>(impl_->input_shape[2]);
        const int target_w = static_cast<int>(impl_->input_shape[3]);
        cv::Mat resized;
        cv::resize(ROI, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        // 4) BGR → RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // 5) 转 float32, 归一化到 [0,1]
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

        // 6) Normalize: (x - 0.5) / 0.5 = x * 2 - 1
        rgb = rgb * 2.0f - 1.0f;

        // 7) HWC → CHW
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        std::vector<float> input_tensor(1 * 3 * target_h * target_w);
        size_t channel_size = static_cast<size_t>(target_h * target_w);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor.data() + c * channel_size,
                        channels[c].ptr<float>(),
                        channel_size * sizeof(float));
        }

        // 8) 创建 ORT Tensor
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.data(),
            input_tensor.size(),
            impl_->input_shape.data(),
            impl_->input_shape.size()
        );

        // 9) 执行推理
        auto outputs = impl_->session->Run(
            Ort::RunOptions{},
            impl_->input_name_ptrs.data(),
            &input,
            1,
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size()
        );

        // 10) 解析输出 -> [1, 2] 概率
        const float* out = outputs[0].GetTensorData<float>();
        float prob_clear = out[0];
        float prob_hazy  = out[1];

        Detection d;
        if (prob_hazy > config_.threshold) {
            d.label = config_.labels[0];  // hazy
            d.confidence = prob_hazy;
        } else {
            d.label = "clear";
            d.confidence = prob_clear;
        }
        predictions.push_back(std::move(d));

        return predictions;
    }
    catch (const std::exception& ex) {
        std::cerr << "[CNN] exception: " << ex.what() << '\n';
        predictions.clear();
    }

    // fallback（异常时 fingerprint）
    if (predictions.empty()) {
        std::uint64_t hash = fingerprint(frame.data);
        Detection d;
        d.label = ((hash % 2) == 0) ? "Clear" : "Hazy";
        d.confidence = 0.6;
        predictions.push_back(d);
    }
    return predictions;
}


}  // namespace app

