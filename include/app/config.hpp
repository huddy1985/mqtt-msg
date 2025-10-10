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
    MqttConfig mqtt;
    RtspConfig rtsp;
    ServiceInfo service;
    std::map<std::string, ModelInfo> scenario_lookup;
    std::vector<ScenarioConfig> scenario_list;
};

AppConfig loadConfig(const std::string& path);

}  // namespace app

