import onnx
import numpy as np
import onnxruntime as ort

def inspect_model(path: str):
    model = onnx.load(path)
    graph = model.graph

    print("\n=== 🔍 模型 I/O 信息 ===")
    print("Inputs:")
    for inp in graph.input:
        t = inp.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in t.shape.dim]
        print(f" - name={inp.name}, shape={shape}, dtype={t.elem_type}")

    print("\nOutputs:")
    for out in graph.output:
        t = out.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in t.shape.dim]
        print(f" - name={out.name}, shape={shape}, dtype={t.elem_type}")

    print("\n=== 🔎 检查是否包含标准化运算 ===")
    op_types = set([n.op_type for n in graph.node])
    print("Operators:", op_types)

    if {"Sub", "Div"} & op_types:
        print("✅ 检测到 Sub/Div 运算，说明 Normalize 很可能已经被写入 ONNX 图中")
    else:
        print("❌ 未发现 Sub/Div，Normalize 很可能在外部（C++必须手动实现）")

    # 用随机输入跑一遍推理尝试推测输出范围
    print("\n=== ▶️ 测试一次推理输出范围 ===")
    sess = ort.InferenceSession(path)
    input_node = sess.get_inputs()[0]
    shape = input_node.shape

    # 构造随机张量（全部填充 0.5，等价图像中心像素）
    fake = np.ones([d if isinstance(d, int) and d > 0 else 1 for d in shape], dtype=np.float32) * 0.5
    out = sess.run(None, {input_node.name: fake})[0]
    print("Output sample:", out)
    print("Output shape:", out.shape)
    print("Output range:", float(np.min(out)), "to", float(np.max(out)))

if __name__ == "__main__":
    inspect_model("../models/cnn_haze.onnx")

