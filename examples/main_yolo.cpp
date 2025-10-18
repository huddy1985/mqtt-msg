#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

// ========== 读取 YAML 中类别名 ==========
std::vector<std::string> loadClassNames(const std::string& yaml_path) {
    std::vector<std::string> names;
    try {
        YAML::Node data = YAML::LoadFile(yaml_path);
        if (data["names"]) {
            for (auto name : data["names"])
                names.push_back(name.as<std::string>());
        } else {
            std::cerr << "[WARN] 'names' not found in YAML.\n";
        }
    } catch (std::exception& e) {
        std::cerr << "[ERROR] Failed to parse YAML: " << e.what() << std::endl;
    }
    return names;
}

// ========== 计算IoU（用于NMS） ==========
float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float interArea = (a & b).area();
    float unionArea = a.area() + b.area() - interArea;
    return interArea / unionArea;
}

// ========== 执行NMS ==========
std::vector<int> NMS(const std::vector<cv::Rect2f>& boxes,
                     const std::vector<float>& scores,
                     float iouThreshold = 0.45f) {
    std::vector<int> indices;
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        indices.push_back(idx);
        for (size_t j = i + 1; j < order.size(); j++) {
            int idx2 = order[j];
            if (IoU(boxes[idx], boxes[idx2]) > iouThreshold)
                suppressed[idx2] = true;
        }
    }
    return indices;
}

// ========== YOLOv11 ONNX 模型类 ==========
class YoloONNX {
public:
    YoloONNX(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_(nullptr) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++)
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator).release());
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++)
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).release());

        std::cout << "[INFO] Model loaded: " << model_path << std::endl;
    }

    std::vector<float> infer(const cv::Mat& img, std::vector<int64_t>& output_shape) {
        // --- 预处理 ---
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(640, 640));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        std::vector<float> input_tensor_values(1 * 3 * 640 * 640);
        int idx = 0;
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 640; ++y)
                for (int x = 0; x < 640; ++x)
                    input_tensor_values[idx++] = rgb.at<cv::Vec3f>(y, x)[c];

        std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // --- 推理 ---
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        output_shape = tensor_info.GetShape();

        size_t total_len = 1;
        for (auto s : output_shape) total_len *= s;

        std::vector<float> output(output_data, output_data + total_len);
        std::cout << "[INFO] Inference done. Output shape = [";
        for (auto s : output_shape) std::cout << s << " ";
        std::cout << "]" << std::endl;
        return output;
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

// ========== 主程序 ==========
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./main <model.onnx> <data.yaml> <image>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string yaml_path = argv[2];
    std::string image_path = argv[3];

    auto class_names = loadClassNames(yaml_path);
    if (class_names.empty()) class_names.push_back("unknown");
    std::cout << "[INFO] Loaded " << class_names.size() << " class names." << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << image_path << std::endl;
        return 1;
    }

    int orig_w = img.cols;
    int orig_h = img.rows;

    YoloONNX yolo(model_path);
    std::vector<int64_t> output_shape;
    auto output = yolo.infer(img, output_shape);

    // ---------- 维度自动识别 ----------
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
            float scale_x = static_cast<float>(orig_w) / 640.0f;
            float scale_y = static_cast<float>(orig_h) / 640.0f;
            x1 *= scale_x; x2 *= scale_x;
            y1 *= scale_y; y2 *= scale_y;

            x1 = std::clamp(x1, 0.0f, (float)orig_w - 1);
            y1 = std::clamp(y1, 0.0f, (float)orig_h - 1);
            x2 = std::clamp(x2, 0.0f, (float)orig_w - 1);
            y2 = std::clamp(y2, 0.0f, (float)orig_h - 1);

            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(conf);
            class_ids.push_back(cls_id);
        }
    }

    auto keep = NMS(boxes, scores, 0.45f);

    for (int idx : keep) {
        cv::Rect2f box = boxes[idx];
        float conf = scores[idx];
        int cls_id = class_ids[idx];
        std::string label = (cls_id >= 0 && cls_id < (int)class_names.size())
                                ? class_names[cls_id]
                                : "unknown";

        std::cout << "Detected: " << label
                  << " conf=" << conf
                  << " box=(" << box.x << "," << box.y << ","
                  << box.x + box.width << "," << box.y + box.height << ")\n";

        cv::rectangle(img, box, {0,255,0}, 2);
        cv::putText(img, label + " " + std::to_string(conf),
                    cv::Point((int)box.x, (int)box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
    }

    cv::imwrite("result.jpg", img);
    std::cout << "[INFO] Saved visualization to result.jpg" << std::endl;
    return 0;
}

