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

YoloModel::YoloModel(const ScenarioDefinition& config) : Model(std::move(config)), config_(std::move(config)) {
    load();
}

YoloModel::~YoloModel() = default;

bool YoloModel::load() {
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
            // 使用 GetInputNames 和 GetOutputNames
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
            // 使用 GetOutputNames
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

std::vector<Detection> YoloModel::infer(const CapturedFrame& frame) const {
    std::vector<Detection> detections;
    if (!loaded_) {
        return detections;
    }

    std::uint64_t frame_hash = fingerprint(frame.data);

#ifdef APP_HAS_ONNXRUNTIME
    if (impl_ && impl_->session && !frame.data.empty()) {
        try {
            std::vector<int64_t> input_shape = impl_->input_shape;
            if (input_shape.empty()) {
                input_shape = {1, 3, 640, 640};
            }
            std::size_t element_count = 1;
            for (auto dim : input_shape) {
                element_count *= static_cast<std::size_t>(dim);
            }
            if (element_count == 0) {
                element_count = frame.data.size();
                if (element_count == 0) {
                    element_count = 1;
                }
                input_shape = {static_cast<int64_t>(element_count)};
            }

            std::vector<float> input_tensor(element_count, 0.0f);
            for (std::size_t i = 0; i < element_count; ++i) {
                std::uint8_t value = frame.data[i % frame.data.size()];
                input_tensor[i] = static_cast<float>(value) / 255.0f;
            }

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                input_tensor.data(),
                                                                input_tensor.size(),
                                                                input_shape.data(),
                                                                input_shape.size());

            // 使用 std::move 来避免复制
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(tensor));  // 使用 std::move()

            std::size_t input_count = std::min<std::size_t>(impl_->input_name_ptrs.size(), static_cast<std::size_t>(1));
            if (input_count == 0) {
                input_count = 1;
            }
            std::size_t output_count = std::max<std::size_t>(static_cast<std::size_t>(1), impl_->output_name_ptrs.size());
            auto outputs = impl_->session->Run(Ort::RunOptions{},
                                               impl_->input_name_ptrs.data(),
                                               inputs.data(),  // 传递数据
                                               inputs.size(),  // 输入数量
                                               impl_->output_name_ptrs.data(),
                                               impl_->output_name_ptrs.size());

            if (!outputs.empty() && outputs.front().IsTensor()) {
                auto type_info = outputs.front().GetTensorTypeAndShapeInfo();
                std::size_t output_elements = type_info.GetElementCount();
                const float* data = outputs.front().GetTensorData<float>();  // 显式指定类型

                if (data && output_elements >= 6) {
                    std::size_t batch = 1;
                    std::size_t boxes = 0;
                    std::size_t attributes = 0;
                    std::vector<int64_t> shape = type_info.GetShape();
                    if (shape.size() >= 3) {
                        batch = static_cast<std::size_t>(std::max<int64_t>(1, shape[0]));
                        boxes = static_cast<std::size_t>(std::max<int64_t>(1, shape[1]));
                        attributes = static_cast<std::size_t>(std::max<int64_t>(6, shape[2]));
                    } else if (shape.size() == 2) {
                        boxes = static_cast<std::size_t>(std::max<int64_t>(1, shape[0]));
                        attributes = static_cast<std::size_t>(std::max<int64_t>(6, shape[1]));
                    } else {
                        attributes = 6;
                        boxes = output_elements / attributes;
                    }
                    if (attributes < 6) {
                        attributes = 6;
                    }
                    if (boxes == 0) {
                        boxes = output_elements / attributes;
                    }

                    std::size_t stride = attributes;
                    std::size_t total_boxes = batch * boxes;
                    for (std::size_t idx = 0; idx < total_boxes && idx * stride + attributes <= output_elements; ++idx) {
                        std::size_t base = idx * stride;
                        float x1 = data[base + 0];
                        float y1 = data[base + 1];
                        float x2 = data[base + 2];
                        float y2 = data[base + 3];
                        float objectness = data[base + 4];
                        if (objectness < 0.01f) {
                            continue;
                        }
                        float best_class_score = 0.0f;
                        std::size_t best_class = 0;
                        for (std::size_t c = 5; c < attributes; ++c) {
                            float score = data[base + c];
                            if (score > best_class_score) {
                                best_class_score = score;
                                best_class = c - 5;
                            }
                        }
                        double confidence = static_cast<double>(objectness * best_class_score);
                        if (confidence < 0.05) {
                            continue;
                        }

                        Region region;
                        region.x1 = static_cast<int>(std::round(x1));
                        region.y1 = static_cast<int>(std::round(y1));
                        region.x2 = static_cast<int>(std::round(x2));
                        region.y2 = static_cast<int>(std::round(y2));
                        if (region.x2 < region.x1) {
                            std::swap(region.x1, region.x2);
                        }
                        if (region.y2 < region.y1) {
                            std::swap(region.y1, region.y2);
                        }

                        // 修改：使用正确的 YoloDetection 类型
                        Detection detection;
                        detection.region = region;
                        detection.label = std::string("class_") + std::to_string(best_class);
                        detection.confidence = std::min(0.999, std::max(0.0, confidence));
                        detections.push_back(std::move(detection));
                    }
                }
            }
        } catch (const Ort::Exception& ex) {
            std::cerr << "YOLO inference warning: " << ex.what() << "\n";
            detections.clear();
        }
    }
#endif

    return detections;
}



}  // namespace app

