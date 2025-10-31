#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <string>

using namespace std;

// 将图像转换为浮点张量并归一化
std::vector<float> preprocessImage(const cv::Mat& img, int height, int width) {
    cv::Mat resized, float_img;
    cv::resize(img, resized, cv::Size(width, height));
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    const std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    const std::vector<float> std  = {0.5f, 0.5f, 0.5f};

    std::vector<float> input_tensor_values(3 * height * width);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float val = float_img.at<cv::Vec3f>(h, w)[c];
                // 与 Python 中的 Normalize 一致: (x - 0.5) / 0.5 = 2x - 1
                val = (val - mean[c]) / std[c];
                input_tensor_values[c * height * width + h * width + w] = val;
            }
        }
    }
    return input_tensor_values;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    const std::string image_path = argv[1];
    const std::string model_path = "../models/cnn_haze.onnx";

    try {
        // 初始化环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CNNInference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        Ort::Session session(env, model_path.c_str(), session_options);

        std::cout << "模型加载成功: " << model_path << std::endl;

        // 读取并预处理图像
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "无法读取图片: " << image_path << std::endl;
            return -1;
        }

        int input_h = 128;
        int input_w = 128;
        std::vector<float> input_tensor_values = preprocessImage(img, input_h, input_w);
        std::vector<int64_t> input_tensor_shape = {1, 3, input_h, input_w};

        // 创建输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_tensor_shape.data(),
            input_tensor_shape.size()
        );

        // 输入输出名
        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"probabilities"};

        // 推理
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_dim = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::cout << "推理结果:" << std::endl;
        for (size_t i = 0; i < output_dim; ++i) {
            std::cout << "Class[" << i << "] = " << output_data[i] << std::endl;
        }

        // 输出预测类别
        auto max_iter = std::max_element(output_data, output_data + output_dim);
        int predicted_class = std::distance(output_data, max_iter);
        std::cout << "预测类别: " << predicted_class << " (置信度: " << *max_iter << ")" << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "未知错误发生！" << std::endl;
        return -1;
    }

    return 0;
}

