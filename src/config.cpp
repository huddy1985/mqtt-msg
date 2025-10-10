#include "app/config.hpp"

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

}  // namespace

AppConfig loadConfig(const std::string& path) {
    simplejson::JsonValue root = simplejson::parseFile(path);

    AppConfig config;

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
        ScenarioConfig entry;
        entry.id = scenario.getString("id");
        entry.model = parseModel(scenario.at("model"));
        config.scenario_lookup.emplace(entry.id, entry.model);
        config.scenario_list.push_back(entry);
    }

    return config;
}

}  // namespace app

