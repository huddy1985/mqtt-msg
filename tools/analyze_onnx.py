import onnx

def analyze_onnx_model(model_path):
    print(f"=== Analyzing ONNX model: {model_path} ===")
    model = onnx.load(model_path)
    graph = model.graph

    # 1. 输出节点形状
    print("\n[1] Model Output Info:")
    for output in graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
        print(f" - {output.name}: shape={shape}")

    # 2. NMS 检测
    nms_ops = [n for n in graph.node if n.op_type in ("NonMaxSuppression", "BatchedNMS", "NMS", "NonMaxSuppressionV4")]
    print("\n[2] NMS inside model?:", "YES ✅" if len(nms_ops) else "NO ❌")

    # 3. 检查激活函数 (Sigmoid, Softmax)
    activations = set([n.op_type for n in graph.node if n.op_type in ("Sigmoid", "Softmax")])
    print("\n[3] Activation functions present:", activations if activations else "None")

    # 4. 检查是否有 decode (xywh -> xyxy)
    # decode 通常通过 Add, Mul 和 anchor/grid 节点体现
    decode_ops = [n for n in graph.node if n.op_type in ("Mul", "Add", "Sub", "Div")]
    print("\n[4] Decode-related ops (Mul/Add/Sub/Div) count:", len(decode_ops))

    # 5. 检查 anchor/grid 信息 (yolo head 是否依然存在)
    anchors = [i for i in graph.initializer if "anchor" in i.name.lower()]
    print("\n[5] Anchor tensors present:", "YES (probably raw head)" if anchors else "NO (likely already decoded)")

    # 6. 统计 Op 类型列表 (方便进一步分析)
    ops = set([node.op_type for node in graph.node])
    print("\n[6] Graph Operations:", ops)

    print("\n=== Analysis Finished ===")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_onnx.py model.onnx")
    else:
        analyze_onnx_model(sys.argv[1])

