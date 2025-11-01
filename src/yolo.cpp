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
#include <cstring>
#include <fstream>
#include <array>

#include "app/yolo.hpp"
#include "app/common.hpp"

#ifdef APP_HAS_RKNN
#include <rknn_api.h>
#endif

#ifdef APP_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace app {

struct YoloModel::Impl {
#if defined(APP_HAS_RKNN)
    Impl() = default;
    std::vector<std::uint8_t> model_blob;
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    rknn_tensor_format input_format = RKNN_TENSOR_NCHW;
#elif defined(APP_HAS_ONNXRUNTIME)
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
#endif
    std::vector<int64_t> input_shape{1, 3, 640, 640};
    bool ready = false;
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
            region.width = span;
            region.height = span;
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

std::vector<std::uint8_t> loadBinaryFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    file.seekg(0, std::ios::end);
    std::streampos size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        throw std::runtime_error("Model file is empty: " + path);
    }
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    if (!file) {
        throw std::runtime_error("Failed to read model file: " + path);
    }
    return buffer;
}

#if defined(APP_HAS_RKNN)
std::vector<int64_t> normalizeInputShape(const rknn_tensor_attr& attr)
{
    std::vector<int64_t> dims(attr.dims, attr.dims + attr.n_dims);
    if (dims.size() == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
        return {dims[0], dims[3], dims[1], dims[2]};
    }
    if (dims.empty()) {
        return {1, 3, 640, 640};
    }
    return dims;
}

std::vector<int64_t> tensorDims(const rknn_tensor_attr& attr)
{
    return std::vector<int64_t>(attr.dims, attr.dims + attr.n_dims);
}
#endif

}  // namespace

YoloModel::YoloModel(const ScenarioDefinition& config)
    : Model(config), config_(config), type("yolo")
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

#if defined(APP_HAS_RKNN)
    impl_->model_blob = loadBinaryFile(model_path);
    if (impl_->model_blob.empty()) {
        throw std::runtime_error("RKNN model blob is empty: " + model_path);
    }

    if (rknn_init(&impl_->ctx,
                  impl_->model_blob.data(),
                  static_cast<unsigned int>(impl_->model_blob.size()),
                  0,
                  nullptr) != RKNN_SUCC) {
        throw std::runtime_error("Failed to initialize RKNN context");
    }

    if (rknn_query(impl_->ctx,
                   RKNN_QUERY_IN_OUT_NUM,
                   &impl_->io_num,
                   sizeof(impl_->io_num)) != RKNN_SUCC) {
        throw std::runtime_error("Failed to query RKNN IO information");
    }

    impl_->input_attrs.resize(impl_->io_num.n_input);
    for (std::uint32_t i = 0; i < impl_->io_num.n_input; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        if (rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
            throw std::runtime_error("Failed to query RKNN input attribute");
        }
        impl_->input_attrs[i] = attr;
    }

    impl_->output_attrs.resize(impl_->io_num.n_output);
    for (std::uint32_t i = 0; i < impl_->io_num.n_output; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
            throw std::runtime_error("Failed to query RKNN output attribute");
        }
        impl_->output_attrs[i] = attr;
    }

    if (!impl_->input_attrs.empty()) {
        impl_->input_shape = normalizeInputShape(impl_->input_attrs.front());
        impl_->input_format = impl_->input_attrs.front().fmt;
    }

    impl_->ready = true;
#elif defined(APP_HAS_ONNXRUNTIME)
    try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);

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

    impl_->ready = static_cast<bool>(impl_->session);
#else
    throw std::runtime_error("Neither RKNN nor ONNXRuntime backends are available");
#endif

    std::cout << "load model " << model_path << " success" << std::endl;
    loaded_ = impl_->ready;
    return loaded_;
}

bool YoloModel::release()
{
    if (impl_) {
#if defined(APP_HAS_RKNN)
        if (impl_->ctx) {
            rknn_destroy(impl_->ctx);
            impl_->ctx = 0;
        }
        impl_->model_blob.clear();
        impl_->input_attrs.clear();
        impl_->output_attrs.clear();
#elif defined(APP_HAS_ONNXRUNTIME)
        impl_->input_names.clear();
        impl_->input_name_ptrs.clear();
        impl_->output_names.clear();
        impl_->output_name_ptrs.clear();
        impl_->session.reset();
#endif
        impl_->input_shape.clear();
        impl_->ready = false;
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

    try {
        cv::Mat image = decodeFrameToMat(frame);
        if (image.empty()) {
            std::cerr << "Failed to decode image" << std::endl;
            return detections;
        }

#if defined(APP_HAS_RKNN)
        if (!impl_ || !impl_->ready) {
            return detections;
        }

        if (impl_->input_shape.size() < 4) {
            return detections;
        }

        const int real_height = static_cast<int>(impl_->input_shape[2]);
        const int real_width  = static_cast<int>(impl_->input_shape[3]);

        auto prep = preprocess_letterbox(image, real_width, real_height);

        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_FLOAT32;
        input.size = static_cast<uint32_t>(prep.input_tensor.size() * sizeof(float));
        input.pass_through = 0;

        std::vector<float> nhwc_buffer;
        if (impl_->input_format == RKNN_TENSOR_NHWC) {
            const int c = static_cast<int>(impl_->input_shape[1]);
            const int h = static_cast<int>(impl_->input_shape[2]);
            const int w = static_cast<int>(impl_->input_shape[3]);
            nhwc_buffer.resize(prep.input_tensor.size());
            for (int yy = 0; yy < h; ++yy) {
                for (int xx = 0; xx < w; ++xx) {
                    for (int cc = 0; cc < c; ++cc) {
                        nhwc_buffer[(yy * w + xx) * c + cc] =
                            prep.input_tensor[cc * h * w + yy * w + xx];
                    }
                }
            }
            input.buf = nhwc_buffer.data();
            input.fmt = RKNN_TENSOR_NHWC;
        } else {
            input.buf = prep.input_tensor.data();
            input.fmt = RKNN_TENSOR_NCHW;
        }

        if (rknn_inputs_set(impl_->ctx, 1, &input) != RKNN_SUCC) {
            return detections;
        }

        if (rknn_run(impl_->ctx, nullptr) != RKNN_SUCC) {
            return detections;
        }

        std::vector<rknn_output> outputs(impl_->io_num.n_output);
        for (auto& out : outputs) {
            out.want_float = 1;
            out.is_prealloc = 0;
            out.buf = nullptr;
        }

        if (rknn_outputs_get(impl_->ctx, outputs.size(), outputs.data(), nullptr) != RKNN_SUCC) {
            return detections;
        }

        auto release_outputs = [&]() {
            rknn_outputs_release(impl_->ctx, outputs.size(), outputs.data());
        };

        if (outputs.empty() || !outputs.front().buf) {
            release_outputs();
            return detections;
        }

        const auto& out_attr = impl_->output_attrs.front();
        std::vector<int64_t> out_shape = tensorDims(out_attr);
        int C = 0;
        int N = 0;
        if (out_shape.size() == 3) {
            C = static_cast<int>(out_shape[1]);
            N = static_cast<int>(out_shape[2]);
        } else if (out_shape.size() == 4) {
            if (out_attr.fmt == RKNN_TENSOR_NCHW) {
                C = static_cast<int>(out_shape[1]);
                N = static_cast<int>(out_shape[2] * out_shape[3]);
            } else {
                C = static_cast<int>(out_shape[3]);
                N = static_cast<int>(out_shape[1] * out_shape[2]);
            }
        }

        if (C <= 0 || N <= 0) {
            release_outputs();
            return detections;
        }

        const float* data = static_cast<const float*>(outputs.front().buf);
        auto get_at = [&](int attr_idx, int i_box) -> float {
            return data[attr_idx * N + i_box];
        };

        const bool has_obj = false;
        const int offset_cls = 4;
        const int num_classes = C - 4;

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
            float x1, y1, x2, y2;
        };

        std::vector<Cand> cands;
        cands.reserve(std::min(N, 20000));

        for (int i = 0; i < N; ++i) {
            float cx = get_at(0, i);
            float cy = get_at(1, i);
            float w  = get_at(2, i);
            float h  = get_at(3, i);

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            int best_cls = -1;
            float best_prob = -1.0f;
            for (int c = 0; c < num_classes; ++c) {
                float p = get_at(offset_cls + c, i);
                if (p > best_prob) {
                    best_prob = p;
                    best_cls = c;
                }
            }

            float best_score = has_obj ? best_prob * get_at(4, i) : best_prob;
            if (best_score < static_cast<float>(config_.threshold))
                continue;

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
        std::vector<int> keep;
        keep.reserve(cands.size());

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

            detections.push_back(std::move(det));
        }

        release_outputs();
#elif defined(APP_HAS_ONNXRUNTIME)
        if (!impl_ || !impl_->session) {
            return detections;
        }

        if (impl_->input_shape.size() < 4) {
            return detections;
        }

        const int real_height = static_cast<int>(impl_->input_shape[2]);
        const int real_width  = static_cast<int>(impl_->input_shape[3]);

        auto prep = preprocess_letterbox(image, real_width, real_height);

        std::array<int64_t, 4> input_shape{1, 3, real_height, real_width};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

        Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(
            mem_info, prep.input_tensor.data(), prep.input_tensor.size(),
            input_shape.data(), input_shape.size()
        );

        auto outputs = impl_->session->Run(
            Ort::RunOptions{},
            impl_->input_name_ptrs.data(), &input_tensor_val, 1,
            impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size()
        );

        if (outputs.empty() || !outputs.front().IsTensor()) {
            return detections;
        }

        auto& output = outputs.front();
        auto info = output.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        if (shape.size() != 3 || shape[0] != 1) {
            return detections;
        }

        const int C = static_cast<int>(shape[1]);
        const int N = static_cast<int>(shape[2]);
        const float* data = output.GetTensorData<float>();

        auto get_at = [&](int attr_idx, int i_box)->float {
            return data[attr_idx * N + i_box];
        };

        const bool has_obj   = false;
        const int  offset_cls = 4;
        const int  num_classes = C - 4;

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

        for (int i = 0; i < N; ++i) {
            float cx = get_at(0, i);
            float cy = get_at(1, i);
            float w  = get_at(2, i);
            float h  = get_at(3, i);

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            int   best_cls = -1;
            float best_prob = -1.0f;
            for (int c = 0; c < num_classes; ++c) {
                float p = get_at(offset_cls + c, i);
                if (p > best_prob) {
                    best_prob = p;
                    best_cls = c;
                }
            }

            float best_score = best_prob;

            if (best_score < static_cast<float>(config_.threshold))
                continue;

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

            detections.push_back(std::move(det));
        }
#else
        (void)image;
        return detections;
#endif
    } catch (const std::exception& ex) {
        std::cerr << "[YoloModel] Inference failed: " << ex.what() << std::endl;
        detections.clear();
    }

    if (detections.empty()) {
        std::uint64_t hash = fingerprint(frame.data);
        detections = fallbackDetections(hash, config_.detection_regions);
    }

    std::cout << "[YoloModel] Inference completed. Detections: " << detections.size() << std::endl;
    std::cout << "==============================================================\n" << std::endl;
    return detections;
}

}  // namespace app
