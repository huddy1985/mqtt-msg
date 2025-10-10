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

    if (!root.contains("mqtt")) {
        throw std::runtime_error("Configuration missing 'mqtt' section");
    }
    const auto& mqtt = root.at("mqtt");
    config.mqtt.server = mqtt.getString("server");
    config.mqtt.port = static_cast<int>(mqtt.getNumber("port"));
    config.mqtt.client_id = mqtt.getString("client_id");
    config.mqtt.subscribe_topic = mqtt.getString("subscribe_topic");
    config.mqtt.publish_topic = mqtt.getString("publish_topic");

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
    const auto& scenarios = root.at("scenarios").asArray();
    for (const auto& scenario : scenarios) {
        ScenarioConfig entry = parseScenario(scenario, baseDir);
        config.scenarios.push_back(entry);
        config.scenario_lookup[config.scenarios.back().id] = config.scenarios.size() - 1;
    }

    return config;
}

}  // namespace app

