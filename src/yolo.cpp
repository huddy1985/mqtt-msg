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

                #ifdef _DEBUG_
                std::cout << "input batch: " << impl_->input_shape[0] << ", channel: " << impl_->input_shape[1] << std::endl;
                std::cout << impl_->input_shape[2] << "*" << impl_->input_shape[3] << std::endl;
                #endif

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
            // 输入 shape: [N, C, H, W]
            const int real_height = static_cast<int>(impl_->input_shape[2]); // H
            const int real_width  = static_cast<int>(impl_->input_shape[3]); // W

            // ============ 1. 构造输入张量 ============
            cv::Mat encoded(1, static_cast<int>(frame.data.size()), CV_8UC1,
                            const_cast<uint8_t*>(frame.data.data()));
            cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR); // BGR
            if (image.empty()) {
                std::cerr << "Failed to decode image" << std::endl;
                return detections;
            }

            #ifdef _DEBUG_
            std::cout << "input batch: " << impl_->input_shape[0]
                      << ", channel: "   << impl_->input_shape[1] << std::endl;
            std::cout << real_height << "x" << real_width << std::endl;
            #endif

            // 预处理注意参数顺序: (input_w, input_h)
            auto prep = preprocess_letterbox(image, /*input_w=*/real_width, /*input_h=*/real_height);

            std::array<int64_t, 4> input_shape{1, 3, real_height, real_width};
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
            
            Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(
                mem_info, prep.input_tensor.data(), prep.input_tensor.size(),
                input_shape.data(), input_shape.size()
            );

            // ============ 2. 推理 ============
            auto outputs = impl_->session->Run(
                Ort::RunOptions{},
                impl_->input_name_ptrs.data(), &input_tensor_val, 1,
                impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size()
            );

            if (outputs.empty() || !outputs.front().IsTensor()) {
                std::cerr << "[YoloModel] Invalid ONNX output.\n";
                return detections;
            }

            auto& output = outputs.front();
            auto info = output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape(); // 期望 [1, C, N]
            if (shape.size() != 3 || shape[0] != 1) {
                std::cerr << "[YoloModel] Unexpected output shape: ["
                          << (shape.size() > 0 ? std::to_string(shape[0]) : "?") << ","
                          << (shape.size() > 1 ? std::to_string(shape[1]) : "?") << ","
                          << (shape.size() > 2 ? std::to_string(shape[2]) : "?") << "]\n";
                return detections;
            }

            // --- 关键：固定按 [1, C, N] 读 ---
            const int C = static_cast<int>(shape[1]); // 7
            const int N = static_cast<int>(shape[2]); // 8400 或 136000
            const float* data = output.GetTensorData<float>();

            auto get_at = [&](int attr_idx, int i_box)->float {
                // 严格使用 [1, C, N] 布局
                return data[attr_idx * N + i_box];
            };

            // ============ 3. 明确参数 ============
            // -> 无 objectness: C = 4 + num_classes
            const bool has_obj   = false;         // *** 固定为无 obj ***
            const int  offset_cls = 4;
            const int  num_classes = C - 4;       // 7 - 4 = 3

            // 类别名动态：匹配就用配置，否则回退 class_i
            std::vector<std::string> class_names;
            if ((int)config_.labels.size() == num_classes) {
                class_names = config_.labels;
            } else {
                class_names.resize(num_classes);
                for (int c = 0; c < num_classes; ++c)
                    class_names[c] = "class_" + std::to_string(c);
            }

            struct Cand { 
                int cls; 
                float score; 
                float x1,y1,x2,y2; 
            };

            std::vector<Cand> cands; 
            cands.reserve(std::min(N, 20000));

            // ============ 4. 解析框（固定 cxcywh → xyxy） ============
            for (int i = 0; i < N; ++i) {
                float cx = get_at(0, i);
                float cy = get_at(1, i);
                float w  = get_at(2, i);
                float h  = get_at(3, i);

                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;

                // 分数（无 obj）：直接取类别概率最大值
                int   best_cls = -1;
                float best_prob = -1.0f;
                for (int c = 0; c < num_classes; ++c) {
                    float p = get_at(offset_cls + c, i); // 已含 Sigmoid/Softmax 概率
                    if (p > best_prob) { 
                        best_prob = p; 
                        best_cls = c; 
                    }
                }
                
                float best_score = best_prob; // **不要乘 obj**

                #ifdef DEBUG_
                std::cout << "prob: " << best_prob << "; thresh: " << static_cast<float>(config_.threshold) << std::endl;
                #endif

                if (best_score < static_cast<float>(config_.threshold))
                    continue;

                // 反 letterbox
                float rx1 = (x1 - prep.pad_x) / prep.scale;
                float ry1 = (y1 - prep.pad_y) / prep.scale;
                float rx2 = (x2 - prep.pad_x) / prep.scale;
                float ry2 = (y2 - prep.pad_y) / prep.scale;

                rx1 = std::clamp(rx1, 0.f, (float)image.cols - 1);
                ry1 = std::clamp(ry1, 0.f, (float)image.rows - 1);
                rx2 = std::clamp(rx2, 0.f, (float)image.cols - 1);
                ry2 = std::clamp(ry2, 0.f, (float)image.rows - 1);

                if (rx2 <= rx1 || ry2 <= ry1) 
                    continue;

                cands.push_back({best_cls, best_score, rx1, ry1, rx2, ry2});
            }

            // ============ 5. NMS（class-agnostic, IoU = 0.35） ============
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

            const float IOU_THRESH = 0.35f;
            const int   TOPK       = 300;
            std::vector<int> keep; keep.reserve(cands.size());

            for (int i = 0; i < (int)cands.size(); ++i) {
                if ((int)keep.size() >= TOPK) break;
                bool suppressed = false;
                for (int j = 0; j < (int)keep.size(); ++j) {
                    if (iou(cands[i], cands[keep[j]]) > IOU_THRESH) {
                        suppressed = true; break;
                    }
                }
                if (!suppressed) keep.push_back(i);
            }

            // ============ 6. 输出 ============
            #ifdef _DEBUG_
            cv::Mat vis = image.clone();
            #endif

            detections.reserve(keep.size());
            for (int idx : keep) {
                const auto& c = cands[idx];
                Detection det;

                det.region.x      = (int)std::round(c.x1);
                det.region.y      = (int)std::round(c.y1);
                det.region.width  = (int)std::round(c.x2 - c.x1);
                det.region.height = (int)std::round(c.y2 - c.y1);
                det.confidence    = c.score;

                if (c.cls >= 0 && c.cls < (int)class_names.size())
                    det.label = class_names[c.cls];
                else
                    det.label = "class_" + std::to_string(std::max(0, c.cls));

                #ifdef _DEBUG_
                std::cout << "label: " << det.label << std::endl;
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s %.2f", det.label.c_str(), c.score);
                int base=0;
                auto t = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);

                // 文本背景
                cv::rectangle(
                    vis,
                    cv::Rect(cv::Point(det.region.x, std::max(0, det.region.y - t.height - 6)),
                             cv::Size(t.width + 6, t.height + 6)),
                    cv::Scalar(0, 255, 0),
                    cv::FILLED
                );
                cv::putText(vis, buf, cv::Point(det.region.x + 3, det.region.y - 3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);

                // 主框
                cv::rectangle(vis,
                              cv::Rect(det.region.x, det.region.y, det.region.width, det.region.height),
                              cv::Scalar(0,255,0), 2);
                #endif

                detections.push_back(std::move(det));
            }

            #ifdef _DEBUG_
            std::ostringstream oss;
            oss << "/tmp/debug_frame_" << frame.timestamp << ".jpg";
            cv::imwrite(oss.str(), vis);
            std::cout << "[DEBUG] Saved debug frame: " << oss.str()
                      << "  (" << vis.cols << "x" << vis.rows << ")" << std::endl;
            #endif

        } catch (const Ort::Exception& ex) {
            std::cerr << "[YoloModel] Inference failed: " << ex.what() << std::endl;
            detections.clear();
        }
    }

    std::cout << "[YoloModel] Inference completed. Detections: " << detections.size() << std::endl;
    std::cout << "==============================================================\n" << std::endl;
    return detections;
}


}  // namespace app

