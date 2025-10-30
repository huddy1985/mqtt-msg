#pragma once

#include <map>
#include <string>
#include <vector>

#include "app/json.hpp"
#include "app/command.hpp"
#include "app/common.hpp"

namespace app {

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
    std::string mac_addr;
    std::string server;
    int port = 0;

    std::string client_id;
    std::string subscribe_topic;
    std::string heartbeat_topic;
    std::string publish_topic;
    std::string username;
    std::string password;

    int heartbeat_time;
};

struct ServiceInfo {
    std::string name;
    std::string description;
};

struct ModelConfig {
    std::string id;
    std::string type; // "cnn" or "yolo"
    std::string path;
};

struct ScenarioDefinition {
    std::string id;
    std::string name;
    std::string description;
    std::string mode;

    std::vector<app::Region> detection_regions;
    std::vector<app::Region> filter_regions;

    double threshold{0.5};
    ModelConfig model;
    std::vector<std::string> labels;
};

struct AppConfig {
    std::string version;
    std::string source_path;

    MqttConfig mqtt;
    RtspConfig rtsp;
    ServiceInfo service;
    std::vector<ScenarioConfig> scenarios;
    std::map<std::string, std::size_t> scenario_lookup;
    
    std::map<std::string, std::string> scenario_files;
    std::vector<std::string> active_scenarios;

    int thread_pool_size;
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

