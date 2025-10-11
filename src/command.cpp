#include "app/command.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace app {
namespace {

Region parseRegion(const simplejson::JsonValue& value) {
    if (!value.isArray()) {
        throw std::runtime_error("Region must be an array of four integers");
    }
    const auto& arr = value.asArray();
    if (arr.size() != 4) {
        throw std::runtime_error("Region must contain four numbers");
    }
    Region region;
    region.x1 = static_cast<int>(arr[0].asNumber());
    region.y1 = static_cast<int>(arr[1].asNumber());
    region.x2 = static_cast<int>(arr[2].asNumber());
    region.y2 = static_cast<int>(arr[3].asNumber());
    return region;
}

std::vector<Region> parseRegions(const simplejson::JsonValue& value) {
    std::vector<Region> regions;
    if (!value.isArray()) {
        return regions;
    }
    for (const auto& entry : value.asArray()) {
        regions.push_back(parseRegion(entry));
    }
    return regions;
}

}  // namespace

Command parseCommand(const simplejson::JsonValue& json) {
    Command command;
    if (!json.contains("scenario_id")) {
        throw std::runtime_error("Command must contain scenario_id");
    }

    const auto& scenarioValue = json.at("scenario_id");
    if (scenarioValue.isArray()) {
        for (const auto& entry : scenarioValue.asArray()) {
            command.scenario_ids.push_back(entry.asString());
        }
    } else {
        command.scenario_ids.push_back(scenarioValue.asString());
    }
    if (command.scenario_ids.empty()) {
        throw std::runtime_error("scenario_id must not be empty");
    }
    if (json.contains("detection_regions")) {
        command.detection_regions = parseRegions(json.at("detection_regions"));
    }
    if (json.contains("filter_regions")) {
        command.filter_regions = parseRegions(json.at("filter_regions"));
    }
    if (json.contains("threshold")) {
        command.threshold = json.at("threshold").asNumber();
    }
    if (json.contains("fps")) {
        command.fps = json.at("fps").asNumber();
    }
    if (json.contains("activation_code")) {
        command.activation_code = json.at("activation_code").asString();
    }
    if (json.contains("extra")) {
        command.extra = json.at("extra");
    } else {
        command.extra = simplejson::makeObject();
    }
    return command;
}

std::vector<Command> parseCommandList(const simplejson::JsonValue& json) {
    std::cout << json.dump(4) << std::endl;
    
    std::vector<Command> commands;
    if (json.isArray()) {
        for (const auto& entry : json.asArray()) {
            commands.push_back(parseCommand(entry));
        }
    } else if (json.isObject()) {
        commands.push_back(parseCommand(json));
    } else {
        throw std::runtime_error("Commands must be a JSON object or array");
    }
    return commands;
}

}  // namespace app

