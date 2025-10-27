import onnxruntime as ort
import numpy as np
import cv2

model = "../models/droplet.onnx"
img_path = "../captures/edge-video-analyzer/liquid_leak_detection/frame_000000.jpg"
imgsz = 1280  # 你的导出尺寸

# 读取图像 + letterbox到(imgsz,imgsz)
im = cv2.imread(img_path)
h0, w0 = im.shape[:2]
r = min(imgsz / w0, imgsz / h0)
nw, nh = int(round(w0 * r)), int(round(h0 * r))
padx, pady = (imgsz - nw) // 2, (imgsz - nh) // 2
im_resz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
canvas[pady:pady+nh, padx:padx+nw] = im_resz
inp = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0-1
inp = np.transpose(inp, (2,0,1))[None, ...]  # NCHW

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
y = sess.run([out_name], {inp_name: inp})[0]  # [1,7,N] or [1,N,7]

# 统一为 [N,7]
if y.shape[1] == 7:
    y = np.transpose(y, (0,2,1))
y = y[0]  # [N,7]

# 拆列
x1, y1, x2, y2, f4, f5, f6 = [y[:,i] for i in range(7)]
cls_id = np.rint(f5).astype(np.int32)

# 兼容两种导出：conf=max(f4, f4*f6)
conf_a = f4 * f6
conf_b = f4
conf = np.maximum(conf_a, conf_b)

# 先取 top-K，避免全量NMS过慢
K = 200
idx_top = np.argpartition(-conf, K-1)[:K]
idx_top = idx_top[np.argsort(-conf[idx_top])]

boxes = np.stack([x1, y1, x2, y2], axis=1)[idx_top].astype(np.float32)
scores = conf[idx_top].astype(np.float32)
cids   = cls_id[idx_top]

# 去letterbox映射回原图
boxes[:, [0,2]] = (boxes[:, [0,2]] - padx) / r
boxes[:, [1,3]] = (boxes[:, [1,3]] - pady) / r
boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0-1)
boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0-1)

# 过滤 + NMS（OpenCV）
conf_thres = 0.25  # 若还看不到框，可先降到0.10试试
iou_thres  = 0.45
keep_mask = scores >= conf_thres
boxes = boxes[keep_mask]; scores = scores[keep_mask]; cids = cids[keep_mask]

if len(boxes):
    rects = [tuple(map(int, [b[0], b[1], b[2]-b[0], b[3]-b[1]])) for b in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf_thres, iou_thres)
    vis = im.copy()
    if len(idxs) > 0:
        for j in idxs.flatten():
            x,y,w,h = rects[j]
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"id{int(cids[j])} {scores[j]:.2f}", (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        print("Kept:", len(idxs))
    else:
        print("After NMS kept: 0")
    cv2.imwrite("result_verify.jpg", vis)
    print("Saved: result_verify.jpg")
else:
    print("No boxes over conf_thres before NMS. Try lowering conf_thres to 0.10 and re-run.")

