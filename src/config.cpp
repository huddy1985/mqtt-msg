#include "app/config.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace app {
namespace {

ModelInfo parseModel(const simplejson::JsonValue& value) {
    ModelInfo info;
    info.id = value.getString("id");
    info.type = value.getString("type");
    info.path = value.getString("path");
    return info;
}

ScenarioConfig parseScenario(const simplejson::JsonValue& value, const std::filesystem::path& baseDir) {
    ScenarioConfig scenario;
    scenario.id = value.getString("id");
    scenario.active = value.getBool("active", false);

    const simplejson::JsonValue* modelSource = nullptr;
    simplejson::JsonValue externalConfig;

    if (value.contains("config")) {
        std::filesystem::path scenarioPath = value.at("config").asString();
        if (!scenarioPath.is_absolute()) {
            scenarioPath = baseDir / scenarioPath;
        }
        scenarioPath = scenarioPath.lexically_normal();
        scenario.config_path = scenarioPath.generic_string();

        externalConfig = simplejson::parseFile(scenario.config_path);
        if (!externalConfig.isObject()) {
            throw std::runtime_error("Scenario config must be a JSON object: " + scenario.config_path);
        }

        if (externalConfig.contains("id")) {
            std::string fileId = externalConfig.getString("id");
            if (!fileId.empty() && !scenario.id.empty() && fileId != scenario.id) {
                throw std::runtime_error("Scenario id mismatch between local config and " + scenario.config_path);
            }
            scenario.id = fileId;
        }
        if (externalConfig.contains("active")) {
            scenario.active = externalConfig.at("active").asBool(scenario.active);
        }
        if (!externalConfig.contains("model")) {
            throw std::runtime_error("Scenario config missing 'model': " + scenario.config_path);
        }
        modelSource = &externalConfig.at("model");
    } else if (value.contains("model")) {
        modelSource = &value.at("model");
    }

    if (!modelSource) {
        throw std::runtime_error("Scenario entry missing model information for id: " + scenario.id);
    }

    scenario.model = parseModel(*modelSource);
    return scenario;
}

}  // namespace

AppConfig loadConfig(const std::string& path) {
    simplejson::JsonValue root = simplejson::parseFile(path);

    std::filesystem::path configPath(path);
    std::filesystem::path absoluteConfigPath = std::filesystem::absolute(configPath).lexically_normal();
    std::filesystem::path baseDir = absoluteConfigPath.has_parent_path() ? absoluteConfigPath.parent_path()
                                                                       : std::filesystem::path(".");

    AppConfig config;
    config.source_path = absoluteConfigPath.generic_string();
    config.version = root.getString("version");

    if (!root.contains("mqtt")) {
        throw std::runtime_error("Configuration missing 'mqtt' section");
    }
    const auto& mqtt = root.at("mqtt");
    config.mqtt.server = mqtt.getString("server");
    config.mqtt.port = static_cast<int>(mqtt.getNumber("port"));

    /** here we add MAC address tricklly **/
    std::string mac_address = detectLocalMac();
    std::string with_mac_cliend_id = mqtt.getString("client_id") + "_" + mac_address;
    config.mqtt.client_id = with_mac_cliend_id;

    std::string with_mac_commands = mqtt.getString("subscribe_topic") + mac_address;
    config.mqtt.subscribe_topic = with_mac_commands;

    config.mqtt.publish_topic = mqtt.getString("publish_topic");
    config.mqtt.username = mqtt.getString("username");
    config.mqtt.password = mqtt.getString("password");
    config.mqtt.heartbeat_time = mqtt.getNumber("heartbeat_time");
    config.mqtt.heartbeat_topic = mqtt.getString("heartbeat_topic");
  
    if (!root.contains("rtsp")) {
        throw std::runtime_error("Configuration missing 'rtsp' section");
    }
    const auto& rtsp = root.at("rtsp");
    config.rtsp.host = rtsp.getString("host");
    config.rtsp.port = static_cast<int>(rtsp.getNumber("port"));
    config.rtsp.path = rtsp.getString("path");

    if (root.contains("service")) {
        const auto& service = root.at("service");
        config.service.name = service.getString("name");
        config.service.description = service.getString("description", "");
    }

    if (!root.contains("scenarios")) {
        throw std::runtime_error("Configuration missing 'scenarios' section");
    }

    if (root.contains("thread_pool_size")) {
        config.thread_pool_size = root.getNumber("thread_pool_size");
        std::cout << "multi-thread: " << config.thread_pool_size;
    }

    const auto& scenarios = root.at("scenarios").asArray();
    for (const auto& scenario : scenarios) {
        ScenarioConfig entry = parseScenario(scenario, baseDir);
        config.scenarios.push_back(entry);
        config.scenario_lookup[config.scenarios.back().id] = config.scenarios.size() - 1;
    }

    return config;
}

RtspConfig parse_rtsp_config(const simplejson::JsonValue &node) {
    RtspConfig config;
    if (node.contains("host")) {
        config.host = node["host"].asString();
    }
    if (node.contains("frame_rate")) {
        config.frame_rate = static_cast<int>(node["frame_rate"].asNumber());
    }
    if (node.contains("output_dir")) {
        config.output_dir = node["output_dir"].asString();
    }
    return config;
}

MqttConfig parse_mqtt_config(const simplejson::JsonValue &node) {
    MqttConfig config;
    if (node.contains("server")) {
        config.server = node["server"].asString();
    }

    if (node.contains("port")) {
        config.port = static_cast<int>(node["port"].asNumber());
    }

    if (node.contains("client_id")) {
        config.client_id = node["client_id"].asString();
    }

    if (node.contains("username")) {
        config.username = node["username"].asString();
    }

    if (node.contains("password")) {
        config.password = node["password"].asString();
    }

    if (node.contains("subscribe_topic")) {
        std::string mac_addr = detectLocalMac();
        std::string subscribe_tpoic = node["subscribe_topic"].asString();
        std::cout << "subs topic: " << subscribe_tpoic + mac_addr << std::endl;

        config.subscribe_topic = subscribe_tpoic + mac_addr;
    }

    if (node.contains("heartbeat_topic")) {
        config.heartbeat_topic = node["heartbeat_topic"].asString();
    }

    if (node.contains("heartbeat_time")) {
        config.heartbeat_time = node["heartbeat_time"].asNumber();
    }

    if (node.contains("publish_topic")) {
        config.publish_topic = node["publish_topic"].asString();
    }

    return config;
}

simplejson::JsonValue load_json_file(const std::string &path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return simplejson::parse(content);
}

simplejson::JsonValue local_config_to_json(const LocalConfig &config, const std::string &root_dir) {
    simplejson::JsonValue::Object root;

    simplejson::JsonValue::Object service;
    service.emplace("name", config.service_name);
    root.emplace("service", service);

    simplejson::JsonValue::Object rtsp;
    rtsp.emplace("host", config.rtsp.host);
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

    simplejson::JsonValue::Object mqtt;
    mqtt.emplace("server", config.mqtt.server);
    mqtt.emplace("port", static_cast<double>(config.mqtt.port));
    mqtt.emplace("client_id", config.mqtt.client_id);
    mqtt.emplace("username", config.mqtt.username);
    mqtt.emplace("password", config.mqtt.password);
    mqtt.emplace("subscribe_topic", config.mqtt.subscribe_topic);
    mqtt.emplace("publish_topic", config.mqtt.publish_topic);
    root.emplace("mqtt", mqtt);

    simplejson::JsonValue::Object scenarios;
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

    simplejson::JsonValue::Array active;
    for (const auto &id : config.active_scenarios) {
        active.emplace_back(id);
    }
    root.emplace("active_scenarios", active);

    return root;
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

void write_json_file(const std::string &path, const simplejson::JsonValue &json) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to write file: " + path);
    }
    output << json.dump(2) << "\n";
}

ConfigStore::ConfigStore(std::string root_dir) : root_dir_(std::move(root_dir)) {}

LocalConfig ConfigStore::load_local(const std::string &path) {
    simplejson::JsonValue root = simplejson::parseFile(path);
    LocalConfig config;

    if (root.contains("service")) {
        const auto &service = root["service"].asObject();
        auto it_name = service.find("name");
        if (it_name != service.end() && it_name->second.isString()) {
            config.service_name = it_name->second.asString();
        }
    }

    if (root.contains("rtsp")) {
        config.rtsp = parse_rtsp_config(root["rtsp"]);
        config.rtsp.output_dir = resolve_path(root_dir_, config.rtsp.output_dir);
    }

    if (root.contains("mqtt")) {
        config.mqtt = parse_mqtt_config(root["mqtt"]);
        std::cout << "mqtt" << root["mqtt"].dump(4) << std::endl;
    }

    if (root.contains("scenarios")) {
        const auto &scenarios = root["scenarios"].asObject();
        for (const auto &kv : scenarios) {
            std::string resolved = resolve_path(root_dir_, kv.second.asString());
            config.scenario_files.emplace(kv.first, resolved);
        }
    }

    if (root.contains("active_scenarios")) {
        for (const auto &entry : root["active_scenarios"].asArray()) {
            config.active_scenarios.push_back(entry.asString());
        }
    }

    return config;
}

void ConfigStore::save_local(const std::string &path, const LocalConfig &config) const {
    simplejson::JsonValue root = local_config_to_json(config, root_dir_);
    write_json_file(path, root);
}

ModelConfig parse_model_config(const simplejson::JsonValue &node) {
    ModelConfig model;
    if (node.contains("id")) {
        model.id = node["id"].asString();
    }
    if (node.contains("type")) {
        model.type = node["type"].asString();
    }
    if (node.contains("path")) {
        model.path = node["path"].asString();
    }

    return model;
}

ScenarioDefinition parse_scenario_definition(const simplejson::JsonValue &root) {
    ScenarioDefinition def;
    if (root.contains("scenario_id")) {
        def.id = root["scenario_id"].asString();
    }

    if (root.contains("name")) {
        def.name = root["name"].asString();
    }

    if (root.contains("description")) {
        def.description = root["description"].asString();
    }

    if (root.contains("model")) {
        def.model = parse_model_config(root["model"]);
    }

    if (root.contains("mode")) {
        def.mode = root["mode"].asString();
    }

    if (root.contains("detection_regions")) {
        def.detection_regions = parseRegions(root["detection_regions"].asArray());
    }

    if (root.contains("filter_regions")) {
        def.filter_regions = parseRegions(root["filter_regions"].asArray());
    }

    if (root.contains("confidence_threshold")) {
        def.threshold = root["confidence_threshold"].asNumber();
    }

    if (root.contains("labels")) {
        def.labels = parseLabels(root["labels"]);
    }

    return def;
}

ScenarioDefinition ConfigStore::load_scenario_file(const std::string &path) const {
    simplejson::JsonValue root = load_json_file(path);
    return parse_scenario_definition(root);
}

}  // namespace app



