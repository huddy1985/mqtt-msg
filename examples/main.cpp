#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

int main() 
{
    try {
        // 初始化 ONNX Runtime 环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CNNInference");
        std::cout << "ONNX Runtime 环境初始化成功！" << std::endl;

        // 加载 ONNX 模型
        const std::string model_path = "cnn_model.onnx";  // 替换为你的 ONNX 模型文件路径
        std::unique_ptr<Ort::Session> session;

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        std::cout << "模型加载成功！" << std::endl;

        // 创建输入张量 (假设输入是 1x3x128x128 的图像)
        std::vector<float> input_tensor_values(1 * 3 * 128 * 128, 1.0f);  // 示例数据，请用实际数据
        std::vector<int64_t> input_tensor_shape = {1, 3, 128, 128};  // [batch_size, channels, height, width]

        // 创建张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_tensor_shape.data(),
                input_tensor_shape.size());

        std::cout << "输入张量创建成功！" << std::endl;

        // 准备输入和输出张量名称
        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"probabilities"};

        // 执行推理
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));

        std::vector<Ort::Value> ort_outputs;  // 存储输出结果

        std::cout << "开始执行推理..." << std::endl;

        // 执行推理并检查错误
        ort_outputs = session->Run(Ort::RunOptions{}, 
                     input_names.data(), 
                     ort_inputs.data(), 
                     ort_inputs.size(), 
                     output_names.data(), 
                     output_names.size());

        std::cout << "推理完成！" << std::endl;

        // 处理输出 (例如：打印第一个输出)
        float* output_data = ort_outputs[0].GetTensorMutableData<float>();
        std::cout << "模型输出: " << output_data[0] << std::endl;  // 根据模型的输出调整
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "未知错误发生！" << std::endl;
        return -1;
    }

    return 0;
}

