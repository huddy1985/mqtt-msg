#include "app/config.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>

namespace app {

namespace {

Json load_json_file(const std::string &path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return Json::parse(content);
}

void write_json_file(const std::string &path, const Json &json) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to write file: " + path);
    }
    output << json.dump(2) << "\n";
}

std::string resolve_path(const std::string &root, const std::string &path) {
    if (path.empty()) {
        return path;
    }
    std::filesystem::path p(path);
    if (p.is_absolute()) {
        return p.string();
    }
    return (std::filesystem::path(root) / p).lexically_normal().string();
}

} // namespace

ConfigStore::ConfigStore(std::string root_dir) : root_dir_(std::move(root_dir)) {}

LocalConfig ConfigStore::load_local(const std::string &path) {
    Json root = load_json_file(path);
    LocalConfig config;

    if (root.contains("service")) {
        const auto &service = root["service"].as_object();
        auto it_name = service.find("name");
        if (it_name != service.end() && it_name->second.is_string()) {
            config.service_name = it_name->second.as_string();
        }
    }

    if (root.contains("rtsp")) {
        config.rtsp = parse_rtsp_config(root["rtsp"]);
        config.rtsp.output_dir = resolve_path(root_dir_, config.rtsp.output_dir);
    }

    if (root.contains("mqtt")) {
        config.mqtt = parse_mqtt_config(root["mqtt"]);
    }

    if (root.contains("scenarios")) {
        const auto &scenarios = root["scenarios"].as_object();
        for (const auto &kv : scenarios) {
            std::string resolved = resolve_path(root_dir_, kv.second.as_string());
            config.scenario_files.emplace(kv.first, resolved);
        }
    }

    if (root.contains("active_scenarios")) {
        for (const auto &entry : root["active_scenarios"].as_array()) {
            config.active_scenarios.push_back(entry.as_string());
        }
    }

    return config;
}

void ConfigStore::save_local(const std::string &path, const LocalConfig &config) const {
    Json root = local_config_to_json(config, root_dir_);
    write_json_file(path, root);
}

ScenarioDefinition ConfigStore::load_scenario_file(const std::string &path) const {
    Json root = load_json_file(path);
    return parse_scenario_definition(root);
}

RtspConfig parse_rtsp_config(const Json &node) {
    RtspConfig config;
    if (node.contains("url")) {
        config.url = node["url"].as_string();
    }
    if (node.contains("frame_rate")) {
        config.frame_rate = static_cast<int>(node["frame_rate"].as_number());
    }
    if (node.contains("output_dir")) {
        config.output_dir = node["output_dir"].as_string();
    }
    return config;
}

MqttConfig parse_mqtt_config(const Json &node) {
    MqttConfig config;
    if (node.contains("host")) {
        config.host = node["host"].as_string();
    }
    if (node.contains("port")) {
        config.port = static_cast<int>(node["port"].as_number());
    }
    if (node.contains("client_id")) {
        config.client_id = node["client_id"].as_string();
    }
    if (node.contains("username")) {
        config.username = node["username"].as_string();
    }
    if (node.contains("password")) {
        config.password = node["password"].as_string();
    }
    if (node.contains("command_topic")) {
        config.command_topic = node["command_topic"].as_string();
    }
    if (node.contains("result_topic")) {
        config.result_topic = node["result_topic"].as_string();
    }
    return config;
}

ScenarioDefinition parse_scenario_definition(const Json &root) {
    ScenarioDefinition def;
    if (root.contains("id")) {
        def.id = root["id"].as_string();
    }
    if (root.contains("name")) {
        def.name = root["name"].as_string();
    }
    if (root.contains("threshold")) {
        def.threshold = root["threshold"].as_number();
    }
    if (root.contains("models")) {
        for (const auto &model : root["models"].as_array()) {
            def.models.push_back(parse_model_config(model));
        }
    }
    return def;
}

ModelConfig parse_model_config(const Json &node) {
    ModelConfig model;
    if (node.contains("id")) {
        model.id = node["id"].as_string();
    }
    if (node.contains("type")) {
        model.type = node["type"].as_string();
    }
    if (node.contains("path")) {
        model.path = node["path"].as_string();
    }
    if (node.contains("threshold")) {
        model.threshold = node["threshold"].as_number();
    }
    if (node.contains("labels")) {
        for (const auto &label : node["labels"].as_array()) {
            model.labels.push_back(label.as_string());
        }
    }
    return model;
}

Json scenario_definition_to_json(const ScenarioDefinition &definition) {
    Json::object_t root;
    root.emplace("id", definition.id);
    root.emplace("name", definition.name);
    root.emplace("threshold", definition.threshold);

    Json::array_t models_json;
    for (const auto &model : definition.models) {
        Json::object_t node;
        node.emplace("id", model.id);
        node.emplace("type", model.type);
        node.emplace("path", model.path);
        node.emplace("threshold", model.threshold);
        Json::array_t labels_json;
        for (const auto &label : model.labels) {
            labels_json.emplace_back(label);
        }
        node.emplace("labels", labels_json);
        models_json.emplace_back(node);
    }
    root.emplace("models", models_json);

    return root;
}

Json local_config_to_json(const LocalConfig &config, const std::string &root_dir) {
    Json::object_t root;

    Json::object_t service;
    service.emplace("name", config.service_name);
    root.emplace("service", service);

    Json::object_t rtsp;
    rtsp.emplace("url", config.rtsp.url);
    rtsp.emplace("frame_rate", static_cast<double>(config.rtsp.frame_rate));
    std::filesystem::path rtsp_path(config.rtsp.output_dir);
    if (!root_dir.empty()) {
        std::error_code ec;
        auto relative = std::filesystem::relative(rtsp_path, root_dir, ec);
        rtsp.emplace("output_dir", ec ? rtsp_path.string() : relative.string());
    } else {
        rtsp.emplace("output_dir", rtsp_path.string());
    }
    root.emplace("rtsp", rtsp);

    Json::object_t mqtt;
    mqtt.emplace("host", config.mqtt.host);
    mqtt.emplace("port", static_cast<double>(config.mqtt.port));
    mqtt.emplace("client_id", config.mqtt.client_id);
    mqtt.emplace("username", config.mqtt.username);
    mqtt.emplace("password", config.mqtt.password);
    mqtt.emplace("command_topic", config.mqtt.command_topic);
    mqtt.emplace("result_topic", config.mqtt.result_topic);
    root.emplace("mqtt", mqtt);

    Json::object_t scenarios;
    for (const auto &kv : config.scenario_files) {
        std::filesystem::path scenario_path(kv.second);
        std::string stored = scenario_path.string();
        if (!root_dir.empty()) {
            std::error_code ec;
            auto relative = std::filesystem::relative(scenario_path, root_dir, ec);
            if (!ec) {
                stored = relative.string();
            }
        }
        scenarios.emplace(kv.first, stored);
    }
    root.emplace("scenarios", scenarios);

    Json::array_t active;
    for (const auto &id : config.active_scenarios) {
        active.emplace_back(id);
    }
    root.emplace("active_scenarios", active);

    return root;
}

} // namespace app

