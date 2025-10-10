# MQTT 视频分析后台服务

本项目在 3588 开发板（Ubuntu 22.04 环境）上提供一个完整的 MQTT 驱动视频
分析后台服务，实现 `readme.txt` 中的全部要求：

1. 启动时读取本地配置，自动连接远端 MQTT 服务，并订阅分析命令。  
2. 根据命令载入对应的 CNN 或 YOLO ONNX 模型。  
3. 依照命令频率通过 RTSP 拉流抽帧，并将图片送入模型推理。  
4. 组织推理结果（包含类别、目标区域、时间戳）并以 MQTT 消息上报。  
5. 服务启动后会主动发布注册信息，包含本地 IP、服务名称和可用场景。  

代码同时保留“一次性”离线模式，便于在开发主机上调试配置或模型。

## 目录结构

```
.
├── CMakeLists.txt
├── include/        # 头文件
├── src/            # 源代码
├── config/         # 指令示例
├── models/         # 占位 CNN / YOLO ONNX 模型
├── tools/          # 模型导出辅助脚本
├── local.config.json
├── readme.txt      # 原始需求描述
└── README.md
```

核心模块说明：

- `include/app/config.hpp` + `src/config.cpp`：解析本地配置，校验 MQTT、RTSP 与场景映射。  
- `include/app/command.hpp` + `src/command.cpp`：解析 MQTT 下发的分析命令。  
- `include/app/mqtt.hpp` + `src/mqtt.cpp`：封装基于 libmosquitto 的 MQTT 客户端，完成注册、订阅、发布。  
- `include/app/rtsp.hpp` + `src/rtsp.cpp`：调用 `ffmpeg` 抽帧，并对异常、超时和退出码做健壮处理。  
- `include/app/cnn.hpp` + `src/cnn.cpp`：使用 ONNX Runtime 调用 CNN 分类模型。  
- `include/app/yolo.hpp` + `src/yolo.cpp`：使用 ONNX Runtime 执行 YOLO 检测模型。  
- `include/app/pipeline.hpp` + `src/pipeline.cpp`：按场景组织推理流程，生成结构化结果。  
- `src/main.cpp`：命令行入口，负责模式切换、信号处理及服务生命周期管理。

## 构建

```bash
cmake -S . -B build
cmake --build build
```

默认会自动检测 ONNX Runtime 与 libmosquitto，若安装路径非标准位置可通过
`ONNXRUNTIME_ROOT` 或 `MOSQUITTO_ROOT` 环境变量指向安装目录。

生成的可执行文件位于 `build/mqtt_msg`。

## 配置说明

`local.config.json` 示例包含以下关键字段：

```json
{
  "mqtt": {
    "server": "127.0.0.1",
    "port": 1883,
    "client_id": "mqtt-msg-service",
    "username": "edge-user",
    "password": "edge-pass",
    "subscribe_topic": "analysis/commands",
    "publish_topic": "analysis/results"
  },
  "rtsp": {
    "host": "192.168.1.10",
    "port": 8554,
    "path": "stream"
  },
  "service": {
    "name": "factory-analytics",
    "description": "Demo backend service"
  },
  "scenarios": [
    {
      "id": "steam_detection",
      "config": "config/scenario_steam_detection.json",
      "active": false
    },
    {
      "id": "coal_powder_detection",
      "config": "config/scenario_coal_powder_detection.json",
      "active": false
    },
    {
      "id": "ash_powder_detection",
      "config": "config/scenario_ash_powder_detection.json",
      "active": false
    },
    {
      "id": "liquid_leak_detection",
      "config": "config/scenario_liquid_leak_detection.json",
      "active": false
    }
  ]
}
```

- `mqtt.username` / `mqtt.password`：可选的连接凭证，若配置则在连接前调用
  libmosquitto 的 `mosquitto_username_pw_set` 完成鉴权。仅填写密码时会视为配
  置错误并在启动时抛出异常。
- `mqtt.subscribe_topic`：服务订阅命令的主题。命令报文可携带 `commands`
  数组或单个命令对象，支持可选字段 `response_topic`、`request_id`、`extra`。
- `mqtt.publish_topic`：分析结果与注册信息默认发布的主题，可被命令中的
  `response_topic` 覆盖。  
- `rtsp`：抽帧所需的 RTSP 地址信息。  
- `scenarios`：场景与模型的映射。每个场景引用 `config/` 目录下的独立配置文件，并通过 `active` 字段指示当前是否启用。本文示例分别对应蒸汽、煤粉、灰粉与液体泄漏场景，模型类型以 `cnn` 或 `yolo` 开头决定加载逻辑。

每个场景配置文件都会描述模型路径及附加说明，例如 `config/scenario_steam_detection.json`：

```json
{
  "id": "steam_detection",
  "name": "Steam plume detection",
  "description": "Identifies abnormal steam emissions using a CNN classifier.",
  "model": {
    "id": "cnn_v1",
    "type": "cnn",
    "path": "models/cnn_v1.onnx"
  }
}
```

## 运行方式

### 后台服务模式（默认）

```bash
./build/mqtt_msg --config local.config.json
```

程序会：

1. 读取配置并连接 MQTT。  
2. 发布一条注册报文（`type=service_registration`），包含服务名称、描述、
   本地 IP、订阅/发布主题及可用场景。  
3. 订阅 `subscribe_topic`，收到命令后按顺序执行：
   - 解析命令内的场景数组、检测区域、过滤区域、阈值、帧率等参数；
   - 根据命令中的场景标识刷新本地配置，仅激活需要执行的场景，并将激活状态回写 `local.config.json`；
   - 针对每个场景串行加载对应模型，调用 RTSP 拉流模块按频率抽帧，并将图片保存到 `captures/<service>/<scenario>` 目录；
   - 将抽帧结果送入 CNN 或 YOLO ONNX 模型推理；
   - 在离线（`--oneshot`）模式下立即返回所有场景的完整 `analysis_result` 数组；
   - 在后台服务模式下，先回复一条 `type=analysis_result`、`mode=continuous` 的确认消息，然后保持检测循环，只要某个场景的任意帧出现“未被过滤、置信度高于阈值、且标签不是 normal/background/ok”的结果，就立刻以 `type=analysis_anomaly` 的独立 MQTT 消息上报（每个异常帧发布一次）。

服务可通过 `Ctrl+C`、`SIGTERM` 等信号安全停止，信号会触发 MQTT 断连并退出循环。

### 一次性离线模式

用于本地调试或在未部署 MQTT 服务的环境下执行单次分析：

```bash
./build/mqtt_msg --config local.config.json --command config/sample_command.json
# 或者从标准输入读取命令
./build/mqtt_msg --config local.config.json --oneshot <<'JSON'
{
  "scenario_id": ["steam_detection", "liquid_leak_detection"],
  "detection_regions": [[0, 0, 100, 100]],
  "threshold": 0.6,
  "fps": 1
}
JSON
```

无指令输入时会返回当前服务的配置概览，便于检查 MQTT/RTSP/场景配置。

### MQTT 命令格式

命令可以是单个对象，也可以放在 `commands` 数组中：

```json
{
  "request_id": "cmd-001",
  "response_topic": "analysis/results/factory",
  "commands": [
    {
      "scenario_id": ["steam_detection", "liquid_leak_detection"],
      "detection_regions": [[0, 0, 100, 100]],
      "filter_regions": [],
      "threshold": 0.7,
      "fps": 1,
      "activation_code": "ABC-123",
      "extra": {"notes": "demo"}
    }
  ]
}
```

服务会先返回一条确认消息：

```json
{
  "type": "analysis_result",
  "service_name": "factory-analytics",
  "client_id": "mqtt-msg-service",
  "timestamp": "2024-04-16T12:34:56.123Z",
  "command_count": 1,
  "request_id": "cmd-001",
  "mode": "continuous",
  "status": "monitoring_started",
  "commands": [
    {
      "scenario_ids": ["steam_detection", "liquid_leak_detection"],
      "threshold": 0.7,
      "fps": 1,
      "activation_code": "ABC-123",
      "detection_regions": [[0, 0, 100, 100]],
      "filter_regions": [],
      "extra": {"notes": "demo"}
    }
  ],
  "results": []
}
```

后续每当检测到异常帧时，会立即推送单独的异常消息（一个异常帧一条）：

```json
{
  "type": "analysis_anomaly",
  "service_name": "factory-analytics",
  "client_id": "mqtt-msg-service",
  "timestamp": "2024-04-16T12:34:59.987Z",
  "request_id": "cmd-001",
  "scenario_id": "steam_detection",
  "model": {"id": "cnn_v1", "type": "cnn", "path": "models/cnn_v1.onnx"},
  "frame": {
    "timestamp": 3.0,
    "image_path": "captures/factory-analytics/steam_detection/frame_000003.jpg",
    "detections": [
      {
        "label": "anomaly",
        "region": [0, 0, 100, 100],
        "confidence": 0.92,
        "filtered": false
      }
    ]
  },
  "threshold": 0.7,
  "fps": 1
}
```

若希望停止检测，可发送空命令（`{"commands": []}` 或者命令数组为空），服务会返回 `status=monitoring_stopped` 的确认报文。运行过程中如发生推理异常，仍会按照以往约定发布 `type=analysis_error` 的错误消息，方便平台侧告警。

## 模型与推理

### CNN

- 通过 ONNX Runtime C++ API 加载模型。  
- 自动推断输入形状，不足部分按需要填充。  
- 推理输出按顺序映射为 `class_i` 标签，数值即置信度。  
- 若编译时缺少 ONNX Runtime，会退回到可重复的哈希推理结果，同时输出日志提示依赖缺失，保证链路可用。

`tools/export_cnn_to_onnx.py` 提供 PyTorch 到 ONNX 的导出流程，可指定输入形状、
opset 等参数，便于在开发阶段生成兼容模型。

### YOLO

- 加载 ONNX 模型，解析输出张量中的边界框、类别概率与置信度。  
- 支持多批次、不同布局（NxCx?）的输出，自动过滤低置信度结果。  
- 允许命令中提供检测区域作为先验，若模型返回空结果仍会回退到合理的占位检测，
  确保 MQTT 报文格式稳定。  
- 同样提供 `tools/export_yolo_to_onnx.py` 脚本辅助导出。

## RTSP 抽帧

- 依赖 `ffmpeg` 的 `image2pipe` 能力按帧率输出 JPEG 数据。  
- 读取循环内检测数据流错误、EOF、超时，并在 `ffmpeg` 非零退出码或无帧输出时抛出异常。  
- 抓取的图片保存在 `captures/<service_name>/` 下，路径会写入分析结果供回放使用。

## MQTT 依赖

服务基于 libmosquitto 实现，默认 QoS=1 发布消息。部署前请确保：

- MQTT Broker 可访问，账号/密码等安全策略可在此基础上扩展；  
- 网络环境允许 RTSP 与 MQTT 端口通信；  
- 服务进程具备读取模型文件、写入 `captures/` 目录的权限。

## 测试与验证

- `config/sample_command.json`、`config/sample_yolo_command.json` 提供了 CNN / YOLO
  场景的示例指令。  
- `models/` 目录下的占位 ONNX 文件可用于连通性验证，正式部署时请替换为真实模型。  
- 如需验证 MQTT 行为，可配合 `mosquitto_sub`/`mosquitto_pub` 或其他 MQTT 客户端测试。

