#pragma once

#include "app/config.hpp"
#include "app/cnn.hpp"
#include "app/yolo.hpp"

#include <memory>
#include <string>
#include <vector>

namespace app {

class Scenario {
public:
    Scenario(ScenarioDefinition definition, std::string config_path);

    const std::string &id() const { return definition_.id; }
    const std::string &name() const { return definition_.name; }
    double threshold() const { return definition_.threshold; }

    bool load_models();
    std::vector<Detection> analyze(const CapturedFrame& frame);

private:
    ScenarioDefinition definition_;
    std::string config_path_;
    std::vector<std::unique_ptr<Model>> models_;
};

} // namespace app
