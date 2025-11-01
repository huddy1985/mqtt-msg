#!/usr/bin/env python3
import torch
from classifier.model import CNNModel

# 1. 加载模型
model = CNNModel()
model.load_state_dict(torch.load("../models/model.pth", map_location="cpu"))
model.eval()

# 2. 构造输入
dummy_input = torch.randn(1, 3, 128, 128)

# 3. 导出
torch.onnx.export(
    model,
    dummy_input,
    "cnn_haze_classifier.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["probabilities"],
    dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}}
)

print("✅ 导出成功: cnn_haze_classifier.onnx")

