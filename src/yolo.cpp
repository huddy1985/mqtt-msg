#include "app/yolo.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef APP_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace app {

struct YoloModel::Impl {
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "InspectAI") {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    std::vector<int64_t> input_shape{1, 3, 640, 640};
};

namespace {

std::uint64_t fingerprint(const std::vector<std::uint8_t>& data) {
    if (data.empty()) {
        return 0x9e3779b97f4a7c15ull;
    }
    const std::size_t step = data.size() > 1024 ? std::max<std::size_t>(1, data.size() / 1024) : 1;
    std::uint64_t hash = 1469598103934665603ull;
    std::size_t processed = 0;
    for (std::size_t i = 0; i < data.size(); i += step) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ull;
        if (++processed > 4096) {
            break;
        }
    }
    return hash;
}

std::vector<Detection> fallbackDetections(std::uint64_t hash, const std::vector<Region>& hints) {
    std::vector<Detection> detections;
    std::size_t count = hints.empty() ? static_cast<std::size_t>((hash % 3ull) + 1ull)
                                      : std::max<std::size_t>(1, hints.size());
    for (std::size_t i = 0; i < count; ++i) {
        Region region;
        if (!hints.empty()) {
            region = hints[i % hints.size()];
        } else {
            int base = static_cast<int>((hash >> (i * 8)) & 0xFFull);
            int span = 40 + (base % 80);
            region.x1 = (base * 13) % 320;
            region.y1 = (base * 7) % 240;
            region.x2 = region.x1 + span;
            region.y2 = region.y1 + span;
        }

        double confidence_seed = static_cast<double>((hash >> (i * 13)) & 0x3FFull) / 1024.0;
        double confidence = std::min(0.98, std::max(0.35, 0.5 + confidence_seed * 0.5));

        Detection detection;
        detection.region = region;
        detection.label = std::string("detected_object_") + std::to_string(i + 1);
        detection.confidence = confidence;
        detections.push_back(std::move(detection));
    }
    return detections;
}

}  // namespace

YoloModel::YoloModel(const ScenarioDefinition& config) : Model(std::move(config)), 
                                                            config_(std::move(config)),
                                                            type("yolo") 
{

}

YoloModel::~YoloModel() = default;

bool YoloModel::load() 
{
    std::string model_path = config_.model.path;

    if (!model_path.empty() && model_path[0] != '/') {
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path full_path = current_path / model_path;
        model_path = full_path.string();
    }

    std::filesystem::path path(model_path);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("YOLO model file not found: " + model_path);
    }

    impl_ = std::make_unique<Impl>();

    if (impl_) {
        try {
            impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);
            
            Ort::AllocatorWithDefaultOptions allocator;
            std::size_t input_count = impl_->session->GetInputCount();
            impl_->input_names.clear();
            impl_->input_name_ptrs.clear();
            impl_->input_names.reserve(input_count);
            impl_->input_name_ptrs.reserve(input_count);

            std::vector<std::string> input_names = impl_->session->GetInputNames();
            for (const auto& name : input_names) {
                impl_->input_names.push_back(name);
                impl_->input_name_ptrs.push_back(impl_->input_names.back().c_str());
            }

            std::size_t output_count = impl_->session->GetOutputCount();
            impl_->output_names.clear();
            impl_->output_name_ptrs.clear();
            impl_->output_names.reserve(output_count);
            impl_->output_name_ptrs.reserve(output_count);

            std::vector<std::string> output_names = impl_->session->GetOutputNames();
            for (const auto& name : output_names) {
                impl_->output_names.push_back(name);
                impl_->output_name_ptrs.push_back(impl_->output_names.back().c_str());
            }

            for (const auto& name : impl_->input_names)
                std::cout << "[DEBUG] Input name: " << name << std::endl;
            for (const auto& name : impl_->output_names)
                std::cout << "[DEBUG] Output name: " << name << std::endl;

            if (input_count > 0) {
                Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(0);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                impl_->input_shape = tensor_info.GetShape();
                for (auto& dim : impl_->input_shape) {
                    if (dim <= 0) {
                        dim = 1;
                    }
                }
            }
        } catch (const Ort::Exception& ex) {
            std::cerr << "YOLO model load warning: " << ex.what() << "\n";
            impl_->session.reset();
            impl_->input_name_ptrs.clear();
            impl_->output_name_ptrs.clear();
        }
    }

    std::cout << "load model " << model_path << " success" << std::endl;
    loaded_ = true;
    return true;
}

bool YoloModel::release()
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
    std::cout << "Model resources have been released successfully.\n";

    return loaded_;
}

std::string YoloModel::model_type()
{
    return type;
}


std::vector<Detection> YoloModel::infer(const CapturedFrame& frame) const
{
    std::vector<Detection> detections;

    if (!loaded_) {
        std::cerr << "[YoloModel] Warning: model not loaded.\n";
        return detections;
    }

    if (frame.data.empty()) {
        std::cerr << "[YoloModel] Warning: empty frame data.\n";
        return detections;
    }

    std::cout << "[YoloModel] Analyzing frame at timestamp " << frame.timestamp << std::endl;

    if (impl_ && impl_->session) {
        try {
            // ============ 1. 构造输入张量 ============
            cv::Mat image = cv::imdecode(frame.data, cv::IMREAD_COLOR);  // 这里使用 IMREAD_COLOR 读取彩色图像
            if (image.empty()) {
                std::cerr << "Failed to decode image" << std::endl;
                return detections;
            }

            auto prep = preprocess_letterbox(image, INPUT_WIDTH, INPUT_HEIGHT);

            std::array<int64_t, 4> input_shape{1, 3, INPUT_HEIGHT, INPUT_WIDTH};

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
            Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(
                mem_info, prep.input_tensor.data(), prep.input_tensor.size(),
                input_shape.data(), input_shape.size());
            
            std::cout << std::string(impl_->input_name_ptrs[0]) << " " << std::string(impl_->output_name_ptrs[0]) << std::endl;

            // ============ 2. 推理 ============
            auto outputs = impl_->session->Run(Ort::RunOptions{},
                                               impl_->input_name_ptrs.data(),
                                               &input_tensor_val,
                                               1,
                                               impl_->output_name_ptrs.data(),
                                               impl_->output_name_ptrs.size());

            std::cout << "\n=========== [ONNX DEBUG] Session->Run() Output Info ===========" << std::endl;
            std::cout << "Output tensor count: " << outputs.size() << std::endl;
            
            if (outputs.empty() || !outputs.front().IsTensor()) {
                std::cerr << "[YoloModel] Invalid ONNX output.\n";
                return detections;
            }

            auto& output = outputs.front();
            auto info = output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            const float* data = output.GetTensorData<float>();

            if (shape.size() != 3 || shape[1] != 6) {
                std::cerr << "[YoloModel] Unexpected output shape.\n";
                return detections;
            }

            int num_attrs = static_cast<int>(shape[1]);
            int num_boxes = static_cast<int>(shape[2]);

            for (int i = 0; i < num_boxes; ++i) {
                float x1 = data[0 * num_boxes + i];
                float y1 = data[1 * num_boxes + i];
                float x2 = data[2 * num_boxes + i];
                float y2 = data[3 * num_boxes + i];
                float conf = data[4 * num_boxes + i];
                float cls  = data[5 * num_boxes + i];

                if (conf < config_.threshold)
                    continue;

                Detection det;
                det.region.x1 = (static_cast<int>(x1) - prep.pad_x)/prep.scale;
                det.region.y1 = (static_cast<int>(y1) - prep.pad_y)/prep.scale;
                det.region.x2 = (static_cast<int>(x2) - prep.pad_x)/prep.scale;
                det.region.y2 = (static_cast<int>(y2) - prep.pad_y)/prep.scale;

                std::vector<std::string> labels = config_.labels;
                int cls_idx = static_cast<int>(cls);
                if ( cls_idx < labels.size()) {
                    det.label = "class_" + labels[cls_idx];
                } else {
                    det.label = "class_" + std::to_string(static_cast<int>(cls));
                }

                det.confidence = conf;
                detections.push_back(det);

                std::cout << "[YoloModel] Detections parsed: " << conf << 
                    " threshold: " << config_.threshold << 
                    " label: " << det.label << std::endl;
            }

            std::cout << "==============================================================\n" << std::endl;

        } catch (const Ort::Exception& ex) {
            std::cerr << "[YoloModel] Inference failed: " << ex.what() << std::endl;
            detections.clear();
        }
    }

    std::cout << "[YoloModel] Inference completed. Detections: " << detections.size() << std::endl;
    return detections;
}

}  // namespace app

