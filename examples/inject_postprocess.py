import onnx
from onnx import helper, TensorProto, numpy_helper
import argparse

def make_slice_col(g, src, start, end, out_name):
    """Slice last-dim [start:end] while preserving [1, N, 1]"""
    starts = helper.make_tensor(out_name+"_starts", TensorProto.INT64, [3], [0, start, 0])
    ends   = helper.make_tensor(out_name+"_ends",   TensorProto.INT64, [3], [1, end, 9223372036854775807])
    axes   = helper.make_tensor(out_name+"_axes",   TensorProto.INT64, [3], [0, 1, 2])
    steps  = helper.make_tensor(out_name+"_steps",  TensorProto.INT64, [3], [1, 1, 1])
    g.initializer.extend([starts, ends, axes, steps])
    return helper.make_node("Slice", [src, starts.name, ends.name, axes.name, steps.name], [out_name])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.inp)
    g = model.graph

    # 原输出名
    raw_out = g.output[0].name

    # 删除原输出（因为 shape 不合法）
    old_out_vi = g.output.pop()

    # 原输出改成中间层
    raw_new = raw_out + "_raw"
    for node in g.node:
        for i, o in enumerate(node.output):
            if o == raw_out:
                node.output[i] = raw_new

    # 补一个合法的中间 ValueInfo： [1,7,-1]
    g.value_info.append(
        helper.make_tensor_value_info(
            raw_new,
            TensorProto.FLOAT,
            [1, 7, "unk"]     # 这里填 [1,7,-1] / 动态 N
        )
    )

    # transpose -> [1, N, 7]
    t1 = helper.make_node("Transpose", [raw_new], ["n7"], perm=[0, 2, 1])

    # slice 7 列
    s_x1 = make_slice_col(g, "n7", 0, 1, "c_x1")
    s_y1 = make_slice_col(g, "n7", 1, 2, "c_y1")
    s_x2 = make_slice_col(g, "n7", 2, 3, "c_x2")
    s_y2 = make_slice_col(g, "n7", 3, 4, "c_y2")
    s_f4 = make_slice_col(g, "n7", 4, 5, "c_f4")
    s_f5 = make_slice_col(g, "n7", 5, 6, "c_f5")
    s_f6 = make_slice_col(g, "n7", 6, 7, "c_f6")

    # scoreA = f4 * f6
    mul = helper.make_node("Mul", ["c_f4", "c_f6"], ["scoreA"])
    # score = Max(scoreA, f4)
    mx = helper.make_node("Max", ["scoreA", "c_f4"], ["score"])

    # cls_id = Round(f5)
    rnd = helper.make_node("Round", ["c_f5"], ["cls_id"])

    # concat -> [1,N,6]
    cat = helper.make_node("Concat",
                           ["c_x1","c_y1","c_x2","c_y2","score","cls_id"],
                           ["post_dets"],
                           axis=2)

    # 新 ValueInfo: [1, -1, 6]
    new_out = helper.make_tensor_value_info(
        "post_dets",
        TensorProto.FLOAT,
        [1, "unk", 6]
    )
    g.output.append(new_out)

    g.node.extend([t1, s_x1, s_y1, s_x2, s_y2, s_f4, s_f5, s_f6, mul, mx, rnd, cat])

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print("✅ patched model saved:", args.out)

if __name__ == "__main__":
    main()

