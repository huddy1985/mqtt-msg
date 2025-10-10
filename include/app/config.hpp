#pragma once

#include <map>
#include <string>
#include <vector>

#include "app/json.hpp"

namespace app {

struct ModelInfo {
    std::string id;
    std::string type;
    std::string path;
};

struct ScenarioConfig {
    std::string id;
    std::string config_path;
    bool active = false;
    ModelInfo model;
};

struct RtspConfig {
    std::string host;
    int port = 0;
    std::string path;
};

struct MqttConfig {
    std::string server;
    int port = 0;
    std::string client_id;
    std::string subscribe_topic;
    std::string publish_topic;
};

struct ServiceInfo {
    std::string name;
    std::string description;
};

struct AppConfig {
    std::string source_path;
    MqttConfig mqtt;
    RtspConfig rtsp;
    ServiceInfo service;
    std::vector<ScenarioConfig> scenarios;
    std::map<std::string, std::size_t> scenario_lookup;
};

AppConfig loadConfig(const std::string& path);

}  // namespace app

