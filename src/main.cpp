#include "app/command.hpp"
#include "app/config.hpp"
#include "app/mqtt.hpp"
#include "app/pipeline.hpp"
#include "app/rtsp.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace app {

namespace {

std::string read_file(const std::string &path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

std::vector<std::string> unique_scenarios(const std::vector<std::string> &ids) {
    std::set<std::string> unique(ids.begin(), ids.end());
    return std::vector<std::string>(unique.begin(), unique.end());
}

} // namespace

} // namespace app

int main(int argc, char **argv) {
    std::string config_path = "local.config.json";
    std::string command_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--command" && i + 1 < argc) {
            command_path = argv[++i];
        }
    }

    std::filesystem::path root = std::filesystem::current_path();
    app::ConfigStore store(root.string());
    app::LocalConfig local_config = store.load_local(config_path);
    app::Pipeline pipeline(local_config, &store);

    app::MqttClient mqtt;
    mqtt.connect(local_config.mqtt);

    if (!command_path.empty()) {
        std::string command_content = app::read_file(command_path);
        app::Json command_json = app::Json::parse(command_content);
        app::Command command = app::parse_command(command_json);

        auto requested_scenarios = app::unique_scenarios(command.scenario_ids);
        pipeline.sync_active_scenarios(requested_scenarios);
        store.save_local(config_path, pipeline.config());

        int frame_count = command.frame_rate > 0 ? command.frame_rate : local_config.rtsp.frame_rate;
        if (frame_count <= 0) {
            frame_count = 1;
        }

        app::RtspFrameGrabber grabber(local_config.rtsp.url, local_config.rtsp.output_dir, frame_count);
        auto frames = grabber.capture_frames(frame_count);

        for (const auto &frame : frames) {
            auto scenario_results = pipeline.analyze_frame(frame, command.threshold);
            app::Json::array_t scenario_results_json;
            bool has_detection = false;
            for (const auto &scenario_result : scenario_results) {
                if (scenario_result.detections.empty()) {
                    continue;
                }
                has_detection = true;
                app::Json::array_t detections_json;
                for (const auto &det : scenario_result.detections) {
                    app::Json::object_t det_json;
                    det_json.emplace("model_id", det.model_id);
                    det_json.emplace("label", det.label);
                    det_json.emplace("confidence", det.confidence);
                    app::Json::array_t bbox_json;
                    for (int value : det.bbox) {
                        bbox_json.emplace_back(static_cast<double>(value));
                    }
                    det_json.emplace("bbox", bbox_json);
                    det_json.emplace("image_path", det.image_path);
                    det_json.emplace("timestamp", det.timestamp);
                    detections_json.emplace_back(det_json);
                }
                app::Json::object_t scenario_json;
                scenario_json.emplace("scenario_id", scenario_result.scenario_id);
                scenario_json.emplace("detections", detections_json);
                scenario_results_json.emplace_back(scenario_json);
            }

            if (!has_detection) {
                continue;
            }

            app::Json::object_t frame_json;
            frame_json.emplace("path", frame.image_path);
            frame_json.emplace("timestamp", frame.timestamp);

            app::Json::object_t payload;
            payload.emplace("service", pipeline.config().service_name);
            payload.emplace("frame", frame_json);
            payload.emplace("results", scenario_results_json);

            mqtt.publish(pipeline.config().mqtt.result_topic, payload);
        }
    }

    return 0;
}

