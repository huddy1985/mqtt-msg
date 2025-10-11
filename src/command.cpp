#include "app/command.hpp"

#include <stdexcept>

namespace app {

namespace {

Region parse_region(const Json &node) {
    Region region;
    if (node.contains("x")) {
        region.x = static_cast<int>(node["x"].as_number());
    }
    if (node.contains("y")) {
        region.y = static_cast<int>(node["y"].as_number());
    }
    if (node.contains("width")) {
        region.width = static_cast<int>(node["width"].as_number());
    }
    if (node.contains("height")) {
        region.height = static_cast<int>(node["height"].as_number());
    }
    return region;
}

} // namespace

Command parse_command(const Json &json) {
    Command command;
    if (!json.contains("scenario_id")) {
        throw std::runtime_error("command missing scenario_id");
    }
    const Json &scenario = json["scenario_id"];
    if (scenario.is_array()) {
        for (const auto &item : scenario.as_array()) {
            command.scenario_ids.push_back(item.as_string());
        }
    } else {
        command.scenario_ids.push_back(scenario.as_string());
    }

    if (json.contains("regions")) {
        for (const auto &region : json["regions"].as_array()) {
            command.regions.push_back(parse_region(region));
        }
    }
    if (json.contains("filters")) {
        for (const auto &filter : json["filters"].as_array()) {
            command.filters.push_back(parse_region(filter));
        }
    }

    if (json.contains("threshold")) {
        command.threshold = json["threshold"].as_number();
    }
    if (json.contains("frame_rate")) {
        command.frame_rate = static_cast<int>(json["frame_rate"].as_number());
    }
    if (json.contains("activation_code")) {
        command.activation_code = json["activation_code"].as_string();
    }
    if (json.contains("extra")) {
        command.extra = json["extra"];
    }

    return command;
}

} // namespace app

