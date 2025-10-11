#pragma once

#include "app/command.hpp"
#include "app/config.hpp"
#include "app/model.hpp"
#include "app/scenario.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace app {

struct ScenarioResult {
    std::string scenario_id;
    std::vector<Detection> detections;
};

class Pipeline {
public:
    Pipeline(LocalConfig config, ConfigStore *store);

    const LocalConfig &config() const { return config_; }
    LocalConfig &config() { return config_; }

    void sync_active_scenarios(const std::vector<std::string> &scenario_ids);
    std::vector<ScenarioResult> analyze_frame(const Frame &frame, double command_threshold);

private:
    LocalConfig config_;
    ConfigStore *store_{nullptr};
    std::map<std::string, std::unique_ptr<Scenario>> active_scenarios_;

    void remove_inactive(const std::vector<std::string> &scenario_ids);
    void add_missing(const std::vector<std::string> &scenario_ids);
};

} // namespace app

