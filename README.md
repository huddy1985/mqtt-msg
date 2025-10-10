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
├── local.config.json
└── README.md
```

核心功能模块说明：

- `include/app/json.hpp`：轻量级 JSON 解析与序列化实现，无第三方依赖。
- `include/app/config.hpp` 与 `src/config.cpp`：加载本地配置文件。
- `include/app/command.hpp` 与 `src/command.cpp`：解析 MQTT 指令。
- `include/app/pipeline.hpp` 与 `src/pipeline.cpp`：根据配置模拟分析流程。
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
     "scenario_id": "traffic_junction",
     "detection_regions": [[10, 10, 120, 160]],
     "threshold": 0.7,
     "fps": 3
   }
   JSON
   ```

3. 如果需要紧凑格式输出，可加入 `--compact` 参数。

程序输出的 JSON 包含所选模型信息、模拟的时间戳以及每个检测区域的
处理结果，可作为在目标硬件环境上开发真实 MQTT 分析服务的参考框架。

