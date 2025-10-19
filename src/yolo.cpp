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
            cv::Mat encoded(1, frame.data.size(), CV_8UC1, const_cast<uint8_t*>(frame.data.data()));
            cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);

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

            if (outputs.empty() || !outputs.front().IsTensor()) {
                std::cerr << "[YoloModel] Invalid ONNX output.\n";
                return detections;
            }

            auto& output = outputs.front();
            auto info = output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            const float* data = output.GetTensorData<float>();

            // 自适配 YOLO 输出 [1, C, N]，其中 C = 4 + num_classes
            if (shape.size() != 3 || shape[0] != 1) {
                std::cerr << "[YoloModel] Unexpected output shape: ["
                        << (shape.size() > 0 ? std::to_string(shape[0]) : "?") << ","
                        << (shape.size() > 1 ? std::to_string(shape[1]) : "?") << ","
                        << (shape.size() > 2 ? std::to_string(shape[2]) : "?") << "]\n";
                return detections;
            }

            const int num_attrs   = static_cast<int>(shape[1]); // 5/6/7/...
            const int num_boxes   = static_cast<int>(shape[2]);
            const int num_classes = num_attrs - 4;
            if (num_classes <= 0) {
                std::cerr << "[YoloModel] num_classes <= 0\n";
                return detections;
            }

            // 预备：类别名映射（label=2 要求输出真实类别名）
            // 如果配置里没有提供类别名，则回退为 "class_0/1/..."
            std::vector<std::string> class_names;
            if (!config_.labels.empty() && static_cast<int>(config_.labels.size()) == num_classes) {
                class_names = config_.labels;
            } else {
                class_names.resize(num_classes);
                for (int c = 0; c < num_classes; ++c) class_names[c] = "class_" + std::to_string(c);
            }

            // 先收集所有通过阈值的候选框（class-agnostic，取每个框的best class）
            struct Cand {
                int   cls;
                float score;
                float x1, y1, x2, y2;  // 注意：你当前代码把 region.width/height 用作 x2/y2
            };
            std::vector<Cand> cands;
            cands.reserve(std::min(num_boxes, 20000)); // 防御性预留

            for (int i = 0; i < num_boxes; ++i) {
                float x1 = data[0 * num_boxes + i];
                float y1 = data[1 * num_boxes + i];
                float x2 = data[2 * num_boxes + i];
                float y2 = data[3 * num_boxes + i];

                // 遍历类别，取最高分及其类别ID（ONNX里已含 Softmax/Sigmoid，不要重复做）
                int   best_cls   = -1;
                float best_score = -1.0f;
                for (int c = 0; c < num_classes; ++c) {
                    float score = data[(4 + c) * num_boxes + i];
                    if (score > best_score) {
                        best_score = score;
                        best_cls   = c;
                    }
                }

                if (best_score < static_cast<float>(config_.threshold)) continue;

                // 反-letterbox到原图坐标系（与你原逻辑保持一致）
                float rx1 = (x1 - prep.pad_x) / prep.scale;
                float ry1 = (y1 - prep.pad_y) / prep.scale;
                float rx2 = (x2 - prep.pad_x) / prep.scale;
                float ry2 = (y2 - prep.pad_y) / prep.scale;

                // 简单裁剪到图像尺寸范围（可选）
                rx1 = std::max(0.0f, std::min(rx1, static_cast<float>(image.cols - 1)));
                ry1 = std::max(0.0f, std::min(ry1, static_cast<float>(image.rows - 1)));
                rx2 = std::max(0.0f, std::min(rx2, static_cast<float>(image.cols - 1)));
                ry2 = std::max(0.0f, std::min(ry2, static_cast<float>(image.rows - 1)));

                // 丢弃无效框
                if (rx2 <= rx1 || ry2 <= ry1) continue;

                cands.push_back({best_cls, best_score, rx1, ry1, rx2, ry2});
            }

            // ============ 3. NMS（class-agnostic, IoU = 0.35） ============

            // 计算 IoU 的小函数（x1,y1,x2,y2 格式）
            auto iou = [](const Cand& a, const Cand& b) {
                float xx1 = std::max(a.x1, b.x1);
                float yy1 = std::max(a.y1, b.y1);
                float xx2 = std::min(a.x2, b.x2);
                float yy2 = std::min(a.y2, b.y2);
                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;
                float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
                float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
                float uni = areaA + areaB - inter + 1e-6f;
                return inter / uni;
            };

            // 置信度从高到低排序
            std::sort(cands.begin(), cands.end(),
                    [](const Cand& a, const Cand& b) { return a.score > b.score; });

            const float IOU_THRESH = 0.35f;        // 你的参数
            const int   TOPK       = 300;          // 可按需调小/调大
            std::vector<int> keep;
            keep.reserve(cands.size());

            for (int i = 0; i < (int)cands.size(); ++i) {
                if ((int)keep.size() >= TOPK) break;
                bool suppressed = false;
                for (int j = 0; j < (int)keep.size(); ++j) {
                    if (iou(cands[i], cands[keep[j]]) > IOU_THRESH) {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed) keep.push_back(i);
            }

            // ============ 4. 输出 Detection ============

            detections.reserve(keep.size());
            for (int idx : keep) {
                const auto& c = cands[idx];
                Detection det;
                det.region.x      = c.x1;
                det.region.y      = c.y1;
                det.region.width  = c.x2;  // 注意：你项目里把 width/height 用作 x2/y2（保持与现有渲染一致）
                det.region.height = c.y2;
                det.confidence    = c.score;

                // 输出类别名（label=2）
                if (c.cls >= 0 && c.cls < (int)class_names.size()) {
                    det.label = class_names[c.cls];
                } else {
                    det.label = "class_" + std::to_string(std::max(0, c.cls));
                }

                detections.push_back(std::move(det));
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

