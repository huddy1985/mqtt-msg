#pragma once

#include <string>
#include <vector>

#include "app/json.hpp"
#include "app/config.hpp"
#include "app/common.hpp"

namespace app {

struct Command {
    std::vector<std::string> scenario_ids;
    std::vector<Region> detection_regions;
    std::vector<Region> filter_regions;
    double threshold = 0.5;
    double fps = 1.0;
    ModelInfo model_info;
    std::string activation_code;
    simplejson::JsonValue extra;
};

Command parseCommand(const simplejson::JsonValue& json);
std::vector<Command> parseCommandList(const simplejson::JsonValue& json);

}  // namespace app

