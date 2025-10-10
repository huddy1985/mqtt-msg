# MQTT Message Driven Video Analytics Simulator

本项目是一个在 3588 开发板（Ubuntu 22.04 环境）上运行的 C++ 示例程序，
用于模拟 README.TXT 中描述的 MQTT 消息驱动视频分析流程。程序读取本地
配置文件 `local.config.json`，解析 MQTT 下发的分析指令（以 JSON 格式提
供），根据不同场景选择预置模型，并输出模拟的分析结果。

## 目录结构

```
.
├── CMakeLists.txt
├── include/        # 头文件
├── src/            # 源代码
├── config/         # 示例指令
├── models/         # 占位 CNN 模型文件
├── local.config.json
└── README.md
```

核心功能模块说明：

- `include/app/json.hpp`：轻量级 JSON 解析与序列化实现，无第三方依赖。
- `include/app/config.hpp` 与 `src/config.cpp`：加载本地配置文件。
- `include/app/command.hpp` 与 `src/command.cpp`：解析 MQTT 指令。
- `include/app/pipeline.hpp` 与 `src/pipeline.cpp`：根据配置模拟分析流程，并在每次分析时将抓取到的帧保存到本地，同时支持在 CNN 场景下调用推理接口。
- `include/app/cnn.hpp` 与 `src/cnn.cpp`：封装基于 PyTorch C++ API（TorchScript）加载 CNN 模型的逻辑，并在无法提供实际模型时使用占位推理结果。
- `include/app/rtsp.hpp` 与 `src/rtsp.cpp`：使用 `ffmpeg` 命令行工具通过 RTSP 协议按指定帧率拉取视频流并生成图片。
- `src/main.cpp`：命令行入口，整合配置读取、指令解析与结果输出。

## 构建

```bash
cmake -S . -B build
cmake --build build
```

生成的可执行文件位于 `build/mqtt_msg`。

## 使用示例

1. 直接查看当前设备的配置汇总（无指令输入时会返回服务信息）：

   ```bash
   ./build/mqtt_msg --config local.config.json
   ```

2. 指定指令文件执行模拟分析（项目提供 `config/sample_command.json` 示例）：

   ```bash
   ./build/mqtt_msg --config local.config.json --command config/sample_command.json
   ```

   也可以通过标准输入传入 JSON 指令：

   ```bash
   ./build/mqtt_msg <<'JSON'
   {
     "scenario_id": "factory_floor",
     "detection_regions": [[0, 0, 0, 0]],
     "threshold": 0.6,
     "fps": 1
   }
   JSON
   ```

3. 如果需要紧凑格式输出，可加入 `--compact` 参数。

程序输出的 JSON 包含所选模型信息、模拟的时间戳、抓取图片的存储路径
以及每个检测区域的处理结果，可作为在目标硬件环境上开发真实 MQTT 分
析服务的参考框架。当场景类型为 `cnn` 时，程序会加载配置中指定的
TorchScript 模型，针对抓取到的帧生成分类标签和置信度，并将推理结
果写入最终的 JSON。若未提供真实模型，工程内置的占位文件会确保流程
仍能执行并生成结构化数据。

## CNN 模型加载与推理

`local.config.json` 中的场景列表包含一个 `factory_floor` 示例，其模型
类型为 `cnn` 并指向 `models/cnn_v1.pt` 占位文件。程序会尝试通过
PyTorch C++ API (`torch::jit::load`) 加载该文件，以模拟 TorchScript
模型的部署方式。如果系统未安装 libtorch，编译过程仍然会成功，但推理
逻辑将退化为对帧数据的哈希指纹计算，从而生成可重复的标签与置信度，
便于 MQTT 报文流程的端到端验证。

要替换为真实模型，可将导出的 TorchScript `.pt` 文件放置在 `models/`
目录下（或更新配置中的路径），并确保编译环境已经安装与配置好的
libtorch。重新编译后即可在 CNN 场景下获得真实的推理输出。

## RTSP 帧采集

RTSP 相关配置位于 `local.config.json`（或实际部署使用的配置文件）
中的 `rtsp` 字段，包含服务端地址、端口与路径。程序运行时会根据
指令中的 `fps` 参数和检测区域数量，使用 `ffmpeg` 的 `image2pipe`
能力从 RTSP 流中提取相应数量的 JPEG 图片。抓取的图片默认保存在
`captures/<service_name>/` 目录下，并在最终的 JSON 输出中通过
`image_path` 字段返回，便于后续识别模型加载或结果回放。

> **注意**：需要在设备上预先安装 `ffmpeg`，并保证可执行文件可
> 在系统 `PATH` 中找到。如果 RTSP 抓取失败，程序会记录错误并继续
> 生成模拟数据，确保整体流程不被中断。

