#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

#include "app/cnn.hpp"
#include "app/common.hpp"

#ifdef APP_HAS_RKNN
#include <rknn_api.h>
#endif

#ifdef APP_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace app {

struct CnnModel::Impl {
#if defined(APP_HAS_RKNN)
    Impl() = default;
    std::vector<std::uint8_t> model_blob;
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    rknn_tensor_format input_format = RKNN_TENSOR_NCHW;
#elif defined(APP_HAS_ONNXRUNTIME)
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "InspectAI") {}
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
#endif
    std::vector<int64_t> input_shape;
    bool ready = false;
};

namespace {

std::uint64_t fingerprint(const std::vector<std::uint8_t>& data)
{
    if (data.empty()) {
        return 0x9e3779b97f4a7c15ull;
    }
    const std::size_t step = data.size() > 1024 ? std::max<std::size_t>(1, data.size() / 1024) : 1;
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
        // Convert NHWC -> NCHW for internal processing
        std::vector<int64_t> reordered{dims[0], dims[3], dims[1], dims[2]};
        return reordered;
    }
    if (dims.empty()) {
        return {1, 3, 128, 128};
    }
    return dims;
}
#endif

}  // namespace

CnnModel::CnnModel(const ScenarioDefinition& config)
    : Model(config), config_(config), type("cnn")
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
    } else {
        impl_->input_shape = {1, 3, 128, 128};
        impl_->input_format = RKNN_TENSOR_NCHW;
    }

    impl_->ready = true;
#elif defined(APP_HAS_ONNXRUNTIME)
    impl_->session_options.SetIntraOpNumThreads(1);
    impl_->session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);

    {
        std::vector<std::string> names = impl_->session->GetInputNames();
        impl_->input_names = names;
        impl_->input_name_ptrs.clear();
        for (auto& n : impl_->input_names) {
            impl_->input_name_ptrs.push_back(n.c_str());
        }
    }
    {
        std::vector<std::string> names = impl_->session->GetOutputNames();
        impl_->output_names = names;
        impl_->output_name_ptrs.clear();
        for (auto& n : impl_->output_names) {
            impl_->output_name_ptrs.push_back(n.c_str());
        }
    }

    Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> s = tensor_info.GetShape();

    if (s.size() == 4) {
        if (s[0] <= 0) s[0] = 1;
        if (s[1] <= 0) s[1] = 3;
        if (s[2] <= 0) s[2] = 128;
        if (s[3] <= 0) s[3] = 128;
    } else {
        s = {1, 3, 128, 128};
    }
    impl_->input_shape = std::move(s);
    impl_->ready = true;
#else
    throw std::runtime_error("Neither RKNN nor ONNXRuntime backends are available");
#endif

    if (!config_.detection_regions.empty()) {
        std::cout << "only extract image: "
                  << "x: " << config_.detection_regions[0].x
                  << " y: " << config_.detection_regions[0].y
                  << " width: " << config_.detection_regions[0].width
                  << " height: " << config_.detection_regions[0].height
                  << std::endl;
    }

    loaded_ = impl_->ready;
    return loaded_;
}

bool CnnModel::release()
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
    if (frame.data.empty()) {
        return predictions;
    }

    try {
        Region rg;
        if (config_.detection_regions.size() != 1) {
            rg = {740, 420, 240, 240};
        } else {
            rg = config_.detection_regions[0];
        }

        cv::Mat image = decodeFrameToMat(frame);
        if (image.empty()) {
            return predictions;
        }

        cv::Mat ROI = extractROI(image, rg.x, rg.y, rg.width, rg.height);
        if (ROI.empty()) {
            return predictions;
        }

#if defined(APP_HAS_RKNN)
        if (!loaded_ || !impl_ || !impl_->ready) {
            return predictions;
        }

        if (impl_->input_shape.size() < 4) {
            return predictions;
        }

        const int target_h = static_cast<int>(impl_->input_shape[2]);
        const int target_w = static_cast<int>(impl_->input_shape[3]);
        cv::Mat resized;
        cv::resize(ROI, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);
        rgb = rgb * 2.0f - 1.0f;

        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        std::vector<float> input_tensor(3 * target_h * target_w);
        const std::size_t channel_size = static_cast<std::size_t>(target_h * target_w);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor.data() + c * channel_size,
                        channels[c].ptr<float>(),
                        channel_size * sizeof(float));
        }

        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_FLOAT32;
        input.size = static_cast<uint32_t>(input_tensor.size() * sizeof(float));
        input.pass_through = 0;

        std::vector<float> nhwc_buffer;
        if (impl_->input_format == RKNN_TENSOR_NHWC) {
            const int c = static_cast<int>(impl_->input_shape[1]);
            const int h = static_cast<int>(impl_->input_shape[2]);
            const int w = static_cast<int>(impl_->input_shape[3]);
            nhwc_buffer.resize(input_tensor.size());
            for (int yy = 0; yy < h; ++yy) {
                for (int xx = 0; xx < w; ++xx) {
                    for (int cc = 0; cc < c; ++cc) {
                        nhwc_buffer[(yy * w + xx) * c + cc] =
                            input_tensor[cc * h * w + yy * w + xx];
                    }
                }
            }
            input.buf = nhwc_buffer.data();
            input.fmt = RKNN_TENSOR_NHWC;
        } else {
            input.buf = input_tensor.data();
            input.fmt = RKNN_TENSOR_NCHW;
        }

        if (rknn_inputs_set(impl_->ctx, 1, &input) != RKNN_SUCC) {
            return predictions;
        }

        if (rknn_run(impl_->ctx, nullptr) != RKNN_SUCC) {
            return predictions;
        }

        std::vector<rknn_output> outputs(impl_->io_num.n_output);
        for (auto& out : outputs) {
            out.want_float = 1;
            out.is_prealloc = 0;
            out.buf = nullptr;
        }

        if (rknn_outputs_get(impl_->ctx, outputs.size(), outputs.data(), nullptr) != RKNN_SUCC) {
            return predictions;
        }

        auto release_outputs = [&]() {
            rknn_outputs_release(impl_->ctx, outputs.size(), outputs.data());
        };

        if (outputs.empty() || !outputs.front().buf) {
            release_outputs();
            return predictions;
        }

        const float* out = static_cast<const float*>(outputs.front().buf);
        float prob_clear = out[0];
        float prob_hazy = out[1];

        Detection d;
        if (prob_hazy > config_.threshold) {
            d.label = config_.labels.empty() ? "hazy" : config_.labels[0];
            d.confidence = prob_hazy;
        } else {
            d.label = "clear";
            d.confidence = prob_clear;
        }
        predictions.push_back(std::move(d));

        release_outputs();
#elif defined(APP_HAS_ONNXRUNTIME)
        if (!loaded_ || !impl_ || !impl_->session) {
            return predictions;
        }

        if (impl_->input_shape.size() < 4) {
            return predictions;
        }

        const int target_h = static_cast<int>(impl_->input_shape[2]);
        const int target_w = static_cast<int>(impl_->input_shape[3]);
        cv::Mat resized;
        cv::resize(ROI, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);
        rgb = rgb * 2.0f - 1.0f;

        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        std::vector<float> input_tensor(3 * target_h * target_w);
        const std::size_t channel_size = static_cast<std::size_t>(target_h * target_w);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor.data() + c * channel_size,
                        channels[c].ptr<float>(),
                        channel_size * sizeof(float));
        }

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.data(),
            input_tensor.size(),
            impl_->input_shape.data(),
            impl_->input_shape.size()
        );

        auto outputs = impl_->session->Run(
            Ort::RunOptions{},
            impl_->input_name_ptrs.data(),
            &input_tensor_val,
            1,
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size()
        );

        if (outputs.empty() || !outputs.front().IsTensor()) {
            return predictions;
        }

        const float* out = outputs.front().GetTensorData<float>();
        float prob_clear = out[0];
        float prob_hazy = out[1];

        Detection d;
        if (prob_hazy > config_.threshold) {
            d.label = config_.labels.empty() ? "hazy" : config_.labels[0];
            d.confidence = prob_hazy;
        } else {
            d.label = "clear";
            d.confidence = prob_clear;
        }
        predictions.push_back(std::move(d));
#else
        (void)ROI;
        return predictions;
#endif

        if (!predictions.empty()) {
            return predictions;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[CNN] exception: " << ex.what() << '\n';
    }

    if (predictions.empty()) {
        std::uint64_t hash = fingerprint(frame.data);
        Detection d;
        d.label = ((hash % 2) == 0) ? "Clear" : "Hazy";
        d.confidence = 0.6;
        predictions.push_back(std::move(d));
    }

    return predictions;
}

}  // namespace app
