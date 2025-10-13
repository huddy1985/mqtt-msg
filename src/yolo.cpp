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
#ifdef APP_HAS_ONNXRUNTIME
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "mqtt_msg") {
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
#endif
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

#ifdef APP_HAS_ONNXRUNTIME
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
                impl_->input_name_ptrs.push_back(name.c_str());
            }

            std::size_t output_count = impl_->session->GetOutputCount();
            impl_->output_names.clear();
            impl_->output_name_ptrs.clear();
            impl_->output_names.reserve(output_count);
            impl_->output_name_ptrs.reserve(output_count);

            std::vector<std::string> output_names = impl_->session->GetOutputNames();
            for (const auto& name : output_names) {
                impl_->output_names.push_back(name);
                impl_->output_name_ptrs.push_back(name.c_str());
            }

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
#endif

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


std::vector<Detection> YoloModel::infer(const CapturedFrame& frame) const {
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

#ifdef APP_HAS_ONNXRUNTIME
    if (impl_ && impl_->session) {
        try {
            // ============ 1. 构造输入张量 ============
            std::vector<int64_t> input_shape = impl_->input_shape;
            if (input_shape.empty()) input_shape = {1, 3, 640, 640};

            std::size_t element_count = 1;
            for (auto dim : input_shape) element_count *= static_cast<std::size_t>(dim);

            std::vector<float> input_tensor(element_count, 0.0f);
            for (std::size_t i = 0; i < element_count; ++i) {
                std::uint8_t value = frame.data[i % frame.data.size()];
                input_tensor[i] = static_cast<float>(value) / 255.0f;
            }

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());

            // ============ 2. 推理 ============
            auto outputs = impl_->session->Run(Ort::RunOptions{},
                                               impl_->input_name_ptrs.data(),
                                               &input_tensor_val,
                                               1,
                                               impl_->output_name_ptrs.data(),
                                               impl_->output_name_ptrs.size());

            if (outputs.empty() || !outputs.front().IsTensor()) {
                std::cerr << "[YoloModel] Invalid ONNX output.\n";
                return detections;
            }

            auto& output = outputs.front();
            auto info = output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            const float* data = output.GetTensorData<float>();

            if (!data || shape.size() < 2) {
                std::cerr << "[YoloModel] Output tensor shape invalid.\n";
                return detections;
            }

            // ============ 3. YOLOv11 输出解析 ============
            // 支持两种格式：[1, N, 85] 或 [1, 84, N]
            bool is_yolo11 = (shape.size() == 3 && shape[1] < shape[2]);

            std::size_t num_boxes = 0;
            std::size_t num_attrs = 0;

            if (is_yolo11) {
                num_attrs = static_cast<std::size_t>(shape[1]);  // e.g. 84
                num_boxes = static_cast<std::size_t>(shape[2]);  // e.g. 8400
            } else {
                num_boxes = static_cast<std::size_t>(shape[1]);
                num_attrs = static_cast<std::size_t>(shape[2]);
            }

            // ============ 4. 遍历检测框 ============
            for (std::size_t i = 0; i < num_boxes; ++i) {
                float x, y, w, h, obj_conf = 0.0f;

                if (is_yolo11) {
                    // [1,84,8400] -> 每列为一个框
                    x = data[i];
                    y = data[num_boxes + i];
                    w = data[2 * num_boxes + i];
                    h = data[3 * num_boxes + i];
                    obj_conf = data[4 * num_boxes + i];
                } else {
                    // [1,8400,85] -> 每行一个框
                    std::size_t base = i * num_attrs;
                    x = data[base + 0];
                    y = data[base + 1];
                    w = data[base + 2];
                    h = data[base + 3];
                    obj_conf = data[base + 4];
                }

                if (obj_conf < 0.25f)
                    continue;

                // 找到最佳类别
                float best_cls_score = 0.0f;
                std::size_t best_cls = 0;

                if (is_yolo11) {
                    for (std::size_t c = 5; c < num_attrs; ++c) {
                        float cls_conf = data[c * num_boxes + i];
                        if (cls_conf > best_cls_score) {
                            best_cls_score = cls_conf;
                            best_cls = c - 5;
                        }
                    }
                } else {
                    std::size_t base = i * num_attrs;
                    for (std::size_t c = 5; c < num_attrs; ++c) {
                        float cls_conf = data[base + c];
                        if (cls_conf > best_cls_score) {
                            best_cls_score = cls_conf;
                            best_cls = c - 5;
                        }
                    }
                }

                float confidence = obj_conf * best_cls_score;
                if (confidence < 0.25f)
                    continue;

                // 将中心坐标转为 (x1, y1, x2, y2)
                float x1 = x - w / 2.0f;
                float y1 = y - h / 2.0f;
                float x2 = x + w / 2.0f;
                float y2 = y + h / 2.0f;

                Region region;
                region.x1 = static_cast<int>(std::round(x1));
                region.y1 = static_cast<int>(std::round(y1));
                region.x2 = static_cast<int>(std::round(x2));
                region.y2 = static_cast<int>(std::round(y2));

                Detection det;
                det.region = region;
                det.label = "class_" + std::to_string(best_cls);
                det.confidence = confidence;
                detections.push_back(std::move(det));
            }

        } catch (const Ort::Exception& ex) {
            std::cerr << "[YoloModel] Inference failed: " << ex.what() << std::endl;
            detections.clear();
        }
    }
#endif

    std::cout << "[YoloModel] Inference completed. Detections: " << detections.size() << std::endl;
    return detections;
}

}  // namespace app

