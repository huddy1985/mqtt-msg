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
            region.x = (base * 13) % 320;
            region.y = (base * 7) % 240;
            region.width = region.x + span;
            region.height = region.y + span;
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
            cv::Mat image = cv::imdecode(frame.data, cv::IMREAD_COLOR);

            if (image.empty()) {
                std::cerr << "Failed to decode image" << std::endl;
                return detections;
            }

            // Region rects = config_.detection_regions;

            #ifdef _DEBUG_

            #endif

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

            std::vector<int64_t> output_shape;

            float* output_data = outputs[0].GetTensorMutableData<float>();
            auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
            output_shape = tensor_info.GetShape();

            size_t total_len = 1;
            for (auto s : output_shape) total_len *= s;

            std::vector<float> output(output_data, output_data + total_len);
            std::cout << "[INFO] Inference done. Output shape = [";
            for (auto s : output_shape) std::cout << s << " ";
            std::cout << "]" << std::endl;

            int64_t dim1 = output_shape[1];
            int64_t dim2 = output_shape[2];
            bool transposed = false;
            int num_det = 0, num_attrs = 0;

            if (dim1 > dim2) {
                num_det = dim1;
                num_attrs = dim2;
            } else {
                num_det = dim2;
                num_attrs = dim1;
                transposed = true;
            }

            std::vector<float> detections(num_det * num_attrs);

            if (transposed) {
                for (int i = 0; i < num_attrs; i++)
                    for (int j = 0; j < num_det; j++)
                        detections[j * num_attrs + i] = output[i * num_det + j];
            } else {
                detections = output;
            }

            // ---------- sigmoid 激活 ----------
            auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            for (float &v : detections) v = sigmoid(v);

            int num_classes = std::max(0, num_attrs - 5);
            std::cout << "[INFO] Detected boxes: " << num_det
                    << ", num_classes=" << num_classes << std::endl;

            std::vector<cv::Rect2f> boxes;
            std::vector<float> scores;
            std::vector<int> class_ids;

            for (int i = 0; i < num_det; i++) {
                float cx = detections[i * num_attrs + 0];
                float cy = detections[i * num_attrs + 1];
                float w  = detections[i * num_attrs + 2];
                float h  = detections[i * num_attrs + 3];
                float conf = detections[i * num_attrs + 4];

                int cls_id = 0;
                if (num_classes > 0) {
                    float obj_conf = conf;
                    float max_prob = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float prob = detections[i * num_attrs + 5 + c];
                        if (prob > max_prob) {
                            max_prob = prob;
                            cls_id = c;
                        }
                    }
                    conf = obj_conf * max_prob;
                }

                if (conf > 0.5f) {
                    // ===== 修复：相对坐标转像素坐标 =====
                    float x1 = (cx - w / 2.0f) * 640.0f;
                    float y1 = (cy - h / 2.0f) * 640.0f;
                    float x2 = (cx + w / 2.0f) * 640.0f;
                    float y2 = (cy + h / 2.0f) * 640.0f;

                    // 映射回原图尺寸
                    float scale_x = static_cast<float>(image.cols) / 640.0f;
                    float scale_y = static_cast<float>(image.rows) / 640.0f;
                    x1 *= scale_x; x2 *= scale_x;
                    y1 *= scale_y; y2 *= scale_y;

                    x1 = std::clamp(x1, 0.0f, (float)image.cols - 1);
                    y1 = std::clamp(y1, 0.0f, (float)image.rows - 1);
                    x2 = std::clamp(x2, 0.0f, (float)image.cols - 1);
                    y2 = std::clamp(y2, 0.0f, (float)image.rows - 1);

                    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    scores.push_back(conf);
                    class_ids.push_back(cls_id);
                }
            }

            auto keep = NMS(boxes, scores, 0.45f);


            for (int idx : keep) {
                Detection det;

                cv::Rect2f box = boxes[idx];
                float conf = scores[idx];
                int cls_id = class_ids[idx];

                std::vector<std::string> labels = config_.labels;
                
                if ( cls_id < labels.size()) {
                    det.label = "class_" + labels[cls_id];
                } else {
                    det.label = "class_" + std::to_string(cls_id);
                }
                
                std::string label = det.label;

                std::cout << "Detected: " << label
                        << " conf=" << conf
                        << " box=(" << box.x << "," << box.y << ","
                        << box.x + box.width << "," << box.y + box.height << ")\n";

                cv::rectangle(image, box, {0,255,0}, 2);
                cv::putText(image, label + " " + std::to_string(conf),
                            cv::Point((int)box.x, (int)box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
            }

            cv::imwrite("/tmp/result.jpg", image);















            
            /* if (outputs.empty() || !outputs.front().IsTensor()) {
                std::cerr << "[YoloModel] Invalid ONNX output.\n";
                return detections;
            }

            auto& output = outputs.front();
            auto info = output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            const float* data = output.GetTensorData<float>();

            std::cout << "shape size: " << shape.size() << std::endl;
            for (auto it = shape.begin(); it != shape.end(); it++) {
                std::cout << " " << *it;
            }

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
                det.region.x = (x1 - prep.pad_x)/prep.scale;
                det.region.y = (y1 - prep.pad_y)/prep.scale;
                det.region.width = (x2 - prep.pad_x)/prep.scale;
                det.region.height = (y2 - prep.pad_y)/prep.scale;

                det.region.x = std::max(det.region.x, 0);  // 确保坐标在 [0, 原始宽度]
                det.region.y = std::max(det.region.y, 0);  // 同上
                det.region.width = std::min(det.region.width, static_cast<int>(image.cols)); // 防止越界
                det.region.height = std::min(det.region.height, static_cast<int>(image.rows)); // 防止越界

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

            std::cout << "==============================================================\n" << std::endl; */

        } catch (const Ort::Exception& ex) {
            std::cerr << "[YoloModel] Inference failed: " << ex.what() << std::endl;
            detections.clear();
        }
    }

    std::cout << "[YoloModel] Inference completed. Detections: " << detections.size() << std::endl;
    return detections;
}

}  // namespace app

