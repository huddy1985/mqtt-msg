#pragma once

#include "app/json.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace app {

struct RtspConfig {
    std::string url;
    int frame_rate{1};
    std::string output_dir;
};

struct MqttConfig {
    std::string host;
    int port{1883};
    std::string client_id;
    std::string username;
    std::string password;
    std::string command_topic;
    std::string result_topic;
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

RtspConfig parse_rtsp_config(const Json &node);
MqttConfig parse_mqtt_config(const Json &node);
ScenarioDefinition parse_scenario_definition(const Json &root);
ModelConfig parse_model_config(const Json &node);

Json scenario_definition_to_json(const ScenarioDefinition &definition);
Json local_config_to_json(const LocalConfig &config, const std::string &root_dir);

} // namespace app

