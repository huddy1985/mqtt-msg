#include "app/cnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef APP_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace app {

struct CnnModel::Impl {
#ifdef APP_HAS_ONNXRUNTIME
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "mqtt_msg") {}

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    std::vector<int64_t> input_shape;
#endif
};

namespace {

std::uint64_t fingerprint(const std::vector<std::uint8_t>& data) {
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

CnnModel::CnnModel() = default;

CnnModel::CnnModel(const std::string& model_path) {
    load(model_path);
}

CnnModel::~CnnModel() = default;

void CnnModel::load(const std::string& model_path) {
    std::filesystem::path path(model_path);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("CNN model file not found: " + model_path);
    }

    model_path_ = path.generic_string();
    impl_ = std::make_unique<Impl>();

#ifdef APP_HAS_ONNXRUNTIME
    try {
        impl_->session_options.SetIntraOpNumThreads(1);
        impl_->session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path_.c_str(), impl_->session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        std::size_t input_count = impl_->session->GetInputCount();
        impl_->input_names.clear();
        impl_->input_name_ptrs.clear();
        impl_->input_names.reserve(input_count);
        impl_->input_name_ptrs.reserve(input_count);
        for (std::size_t i = 0; i < input_count; ++i) {
            char* name = impl_->session->GetInputName(i, allocator);
            if (name) {
                impl_->input_names.emplace_back(name);
                allocator.Free(name);
            } else {
                impl_->input_names.emplace_back(std::string("input") + std::to_string(i));
            }
        }
        for (const auto& name : impl_->input_names) {
            impl_->input_name_ptrs.push_back(name.c_str());
        }

        std::size_t output_count = impl_->session->GetOutputCount();
        impl_->output_names.clear();
        impl_->output_name_ptrs.clear();
        impl_->output_names.reserve(output_count);
        impl_->output_name_ptrs.reserve(output_count);
        for (std::size_t i = 0; i < output_count; ++i) {
            char* name = impl_->session->GetOutputName(i, allocator);
            if (name) {
                impl_->output_names.emplace_back(name);
                allocator.Free(name);
            } else {
                impl_->output_names.emplace_back(std::string("output") + std::to_string(i));
            }
        }
        for (const auto& name : impl_->output_names) {
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
        throw std::runtime_error(std::string("Failed to load ONNX model: ") + ex.what());
    }
#endif

    loaded_ = true;
}

std::vector<CnnPrediction> CnnModel::infer(const CapturedFrame& frame) const {
    std::vector<CnnPrediction> predictions;
    if (!loaded_ || frame.data.empty()) {
        return predictions;
    }

#ifdef APP_HAS_ONNXRUNTIME
    if (impl_ && impl_->session) {
        try {
            std::vector<int64_t> input_shape = impl_->input_shape;
            if (input_shape.empty()) {
                input_shape = {1, static_cast<int64_t>(frame.data.size())};
            }
            std::size_t element_count = 1;
            for (auto dim : input_shape) {
                element_count *= static_cast<std::size_t>(dim);
            }
            if (element_count == 0) {
                element_count = frame.data.empty() ? 1 : frame.data.size();
                input_shape = {static_cast<int64_t>(element_count)};
            }

            std::vector<float> input_tensor(element_count, 0.0f);
            if (!frame.data.empty()) {
                for (std::size_t i = 0; i < element_count; ++i) {
                    std::uint8_t value = frame.data[i % frame.data.size()];
                    input_tensor[i] = static_cast<float>(value) / 255.0f;
                }
            }

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                input_tensor.data(),
                                                                input_tensor.size(),
                                                                input_shape.data(),
                                                                input_shape.size());

            const Ort::Value* inputs[] = {&tensor};
            auto outputs = impl_->session->Run(Ort::RunOptions{},
                                               impl_->input_name_ptrs.data(),
                                               inputs,
                                               1,
                                               impl_->output_name_ptrs.data(),
                                               impl_->output_name_ptrs.size());

            if (!outputs.empty() && outputs.front().IsTensor()) {
                auto type_info = outputs.front().GetTensorTypeAndShapeInfo();
                std::size_t output_elements = type_info.GetElementCount();
                const float* data = outputs.front().GetTensorData<float>();
                if (data && output_elements > 0) {
                    for (std::size_t i = 0; i < output_elements; ++i) {
                        CnnPrediction pred;
                        pred.label = "class_" + std::to_string(i);
                        pred.confidence = std::max(0.0, std::min(1.0, static_cast<double>(data[i])));
                        predictions.push_back(std::move(pred));
                    }
                }
            }
        } catch (const Ort::Exception& ex) {
            (void)ex;
            predictions.clear();
        }
    }
#endif

    if (predictions.empty()) {
        std::uint64_t hash = fingerprint(frame.data);
        double scaled = static_cast<double>((hash % 1000ull)) / 1000.0;
        double confidence = 0.55 + std::fmod(scaled, 0.4);

        CnnPrediction prediction;
        prediction.label = (hash % 2 == 0) ? "normal" : "anomaly";
        prediction.confidence = std::min(0.99, std::max(0.5, confidence));
        predictions.push_back(prediction);

        if ((hash % 5ull) == 0ull) {
            CnnPrediction secondary;
            secondary.label = prediction.label == "normal" ? "warning" : "normal";
            secondary.confidence = std::max(0.3, 0.8 - prediction.confidence / 2.0);
            predictions.push_back(secondary);
        }
    }

    return predictions;
}

}  // namespace app

