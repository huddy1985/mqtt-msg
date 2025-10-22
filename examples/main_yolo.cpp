#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <yaml-cpp/yaml.h>

// -------------------- 预处理 --------------------
struct PreprocessInfo {
    std::vector<float> input_tensor;
    float scale;
    int pad_x, pad_y;
    cv::Mat letterbox;
};

PreprocessInfo preprocess_letterbox(const cv::Mat& img, int input_w, int input_h) {
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min((float)input_w / img_w, (float)input_h / img_h);
    int new_w = static_cast<int>(std::round(img_w * scale));
    int new_h = static_cast<int>(std::round(img_h * scale));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_x = (input_w - new_w) / 2;
    int pad_y = (input_h - new_h) / 2;

    cv::Mat letterbox(input_h, input_w, img.type(), cv::Scalar(114,114,114));
    resized.copyTo(letterbox(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat float_img;
    letterbox.convertTo(float_img, CV_32F, 1.0/255.0);
    // ✅ YOLO/Ultralytics 使用 RGB
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> chw(3);
    cv::split(float_img, chw);

    std::vector<float> input_tensor;
    input_tensor.reserve(input_w * input_h * 3);
    for (int c = 0; c < 3; ++c) {
        const float* begin = reinterpret_cast<const float*>(chw[c].datastart);
        const float* end   = reinterpret_cast<const float*>(chw[c].dataend);
        input_tensor.insert(input_tensor.end(), begin, end);
    }
    return {input_tensor, scale, pad_x, pad_y, letterbox};
}

// -------------------- 读取 YAML names --------------------
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

// -------------------- 主程序 --------------------
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <data.yaml> <image>\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string yaml_path  = argv[2];
    std::string image_path = argv[3];

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) { std::cerr << "Failed to load image\n"; return 1; }
    int orig_w = img.cols, orig_h = img.rows;

    auto class_names = loadClassNames(yaml_path);
    if (class_names.empty()) std::cerr << "[WARN] Cannot parse class names from yaml. Use empty names.\n";

    // ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yo");
    Ort::SessionOptions so; so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    Ort::Session session(env, model_path.c_str(), so);

    // 输入尺寸（假定 NCHW）
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name  = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);


    int C=3, H=1280, W=1280;  // 默认兜底
    
    try {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        auto in_shape = tensor_info.GetShape(); // 可能抛异常或返回奇怪的 count
        if (in_shape.size() == 4 ) {
            auto fix = [](int64_t d, int def) {
                    return d>0 ? (int)d : def; 
                };
            if (in_shape[1]==3 || in_shape[1]==1) { C=fix(in_shape[1],3); H=fix(in_shape[2],1280); W=fix(in_shape[3],1280); }
            else { C=fix(in_shape[3],3); H=fix(in_shape[1],1280); W=fix(in_shape[2],1280); }
        } else {
            std::cerr << "[WARN] input shape rank != 4, fallback to 1x3x1280x1280\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARN] GetInput shape failed: " << e.what()
                << "\n       fallback to 1x3x1280x1280\n";
    }

    // 预处理（letterbox 到 W×H）
    PreprocessInfo lb = preprocess_letterbox(img, W, H);

    // 构造输入 tensor
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, C, H, W};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, lb.input_tensor.data(), lb.input_tensor.size(), input_shape.data(), input_shape.size()
    );

    const char* input_names[]  = { input_name.get() };
    const char* output_names[] = { output_name.get() };

    // 推理
    auto outs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    if (outs.empty()) { std::cerr << "No output\n"; return 1; }

    // 输出 shape（支持 [1,7,N] 或 [1,N,7]）
    auto& out = outs.front();
    auto out_info  = out.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    if (out_shape.size()!=3) {
        std::cerr << "[ERROR] Unexpected output dims (expect 3). Got: ";
        for (auto d: out_shape) std::cerr << d << " ";
        std::cerr << "\n";
        return 1;
    }

    const float* out_ptr = out.GetTensorData<float>();
    int dim1 = (int)out_shape[1];
    int dim2 = (int)out_shape[2];

    bool attr_first = (dim1 == 7); // [1,7,N]
    int N = attr_first ? dim2 : dim1;
    if (N <= 0) { std::cerr << "[ERROR] N<=0\n"; return 1; }

    // 展平到 [N,7]
    std::vector<float> preds((size_t)N * 7);
    if (attr_first) {
        // [1,7,N] -> [N,7]
        for (int a=0; a<7; ++a) {
            const float* src = out_ptr + (size_t)a * N;
            float* dst_col = preds.data() + a;

            for (int i=0; i<N; ++i) 
                dst_col[i*7] = src[i];
        }
    } else {
        // [1,N,7] 直接拷贝
        std::memcpy(preds.data(), out_ptr, sizeof(float) * (size_t)N * 7);
    }

    // =============== 评分 + TopK 预筛（避免全量 136000 进入 NMS） ===============
    constexpr int TOPK = 2000; // 可按需调整
    std::vector<float> scores_all; scores_all.reserve(N);
    for (int i=0; i<N; ++i) {
        float f4 = preds[i*7 + 4];     // ✅ YOLO11 导出已融合：f4 就是最终 conf
        // float f6 = preds[i*7 + 6];  // 常为占位，不参与评分
        scores_all.push_back(f4);
    }
    std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
    int K = std::min(TOPK, N);
    std::nth_element(order.begin(), order.begin()+K, order.end(),
                     [&](int a, int b){ return scores_all[a] > scores_all[b]; });
    order.resize(K);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores_all[a] > scores_all[b]; });

    // =============== 过滤 + 逆 letterbox 映射回原图坐标 ===============
    const float conf_thres = 0.10f;   // 你的指定阈值
    const float iou_thres  = 0.45f;

    std::vector<cv::Rect> boxes;  boxes.reserve(K);
    std::vector<float>    scores; scores.reserve(K);
    std::vector<int>      class_ids; class_ids.reserve(K);

    int passed = 0;
    for (int id : order) {
        float conf = scores_all[id];
        if (conf < conf_thres) continue;

        float cx = preds[id*7 + 0];
        float cy = preds[id*7 + 1];
        float w  = preds[id*7 + 2];
        float h  = preds[id*7 + 3];

        float x1m = cx - w * 0.5f;
        float y1m = cy - h * 0.5f;
        float x2m = cx + w * 0.5f;
        float y2m = cy + h * 0.5f;

        float f5  = preds[id*7 + 5];  // ✅ class_id

        // 逆 letterbox
        float x1 = (x1m - lb.pad_x) / lb.scale;
        float y1 = (y1m - lb.pad_y) / lb.scale;
        float x2 = (x2m - lb.pad_x) / lb.scale;
        float y2 = (y2m - lb.pad_y) / lb.scale;

        x1 = std::max(0.f, std::min(x1, (float)orig_w - 1.f));
        y1 = std::max(0.f, std::min(y1, (float)orig_h - 1.f));
        x2 = std::max(0.f, std::min(x2, (float)orig_w - 1.f));
        y2 = std::max(0.f, std::min(y2, (float)orig_h - 1.f));

        if (x2 <= x1 + 1.f || y2 <= y1 + 1.f) continue;

        std::cout << "conf: " << conf << std::endl;
        boxes.emplace_back(
            (int)std::round(x1),
            (int)std::round(y1),
            (int)std::round(x2 - x1),
            (int)std::round(y2 - y1)
        );
        scores.emplace_back(conf);

        int cid = (int)std::round(f5);
        if (!class_names.empty()) {
            cid = std::max(0, std::min(cid, (int)class_names.size()-1));
        }
        class_ids.emplace_back(cid);
        ++passed;
    }
    std::cout << "[INFO] candidates after conf: " << passed << " / " << N
              << " (topK=" << K << ")\n";

    // =============== NMS ===============
    std::vector<int> keep;
    if (!boxes.empty())
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, keep);

    // =============== 绘制 ===============
    cv::Mat vis = img.clone();
    for (int idx : keep) {
        const auto& r = boxes[idx];
        float sc = scores[idx];
        int cid  = class_ids[idx];
        cv::rectangle(vis, r, cv::Scalar(0,255,0), 2);

        std::string name = (cid>=0 && cid<(int)class_names.size())
            ? class_names[cid] : ("cls_" + std::to_string(cid));
        
        char buf[128]; std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), sc);
        int base=0; auto t = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        
        cv::rectangle(vis, cv::Rect(cv::Point(r.x, std::max(0, r.y - t.height - 6)),
                                    cv::Size(t.width+6, t.height+6)), cv::Scalar(0,255,0), cv::FILLED);
        cv::putText(vis, buf, cv::Point(r.x+3, r.y-3), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0,0,0), 1, cv::LINE_AA);
    }

    cv::imwrite("result.jpg", vis);
    std::cout << "[INFO] Kept boxes: " << keep.size() << "\n";
    std::cout << "[INFO] Saved to result.jpg\n";
    return 0;
}

