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
    
    int frame_rate{1};
    std::string output_dir;
};

struct MqttConfig {
    std::string server;
    int port = 0;
    std::string client_id;
    std::string subscribe_topic;
    std::string publish_topic;
    std::string username;
    std::string password;
};

struct ServiceInfo {
    std::string name;
    std::string description;
};

struct ModelConfig {
    std::string id;
    std::string type; // "cnn" or "yolo"
    std::string path;
    double threshold{0.5};
    std::vector<std::string> labels;
};

struct ScenarioDefinition {
    std::string id;
    std::string name;
    double threshold{0.5};
    std::vector<ModelConfig> models;
};

struct AppConfig {
    std::string source_path;
    MqttConfig mqtt;
    RtspConfig rtsp;
    ServiceInfo service;
    std::vector<ScenarioConfig> scenarios;
    std::map<std::string, std::size_t> scenario_lookup;
    
    std::map<std::string, std::string> scenario_files;
    std::vector<std::string> active_scenarios;
};

AppConfig loadConfig(const std::string& path);

struct LocalConfig {
    std::string service_name;
    RtspConfig rtsp;
    MqttConfig mqtt;
    std::map<std::string, std::string> scenario_files;
    std::vector<std::string> active_scenarios;
};

class ConfigStore {
public:
    explicit ConfigStore(std::string root_dir);

    LocalConfig load_local(const std::string &path);
    void save_local(const std::string &path, const LocalConfig &config) const;

    ScenarioDefinition load_scenario_file(const std::string &path) const;

    const std::string &root() const { return root_dir_; }

private:
    std::string root_dir_;
};

}  // namespace app

