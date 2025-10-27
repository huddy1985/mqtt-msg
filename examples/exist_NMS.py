import onnx
m=onnx.load('../models/smoke.onnx')
print([n.op_type for n in m.graph.node if 'NonMaxSuppression' in n.op_type])
