#include "app/command.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace app {
namespace {

ModelInfo parseModel(const simplejson::JsonValue& value)
{
    ModelInfo modelinfo;

    if (value.contains("id")) {
        modelinfo.id = value["id"].asString();
    }

    if (value.contains("type")) {
        modelinfo.type = value["type"].asString();
    }

    if (value.contains("path")) {
        modelinfo.path = value["path"].asString();
    }

    return modelinfo;
}

}  // namespace

Command parseCommand(const simplejson::JsonValue& json) {
    Command command;
    if (!json.contains("scenario_id")) {
        throw std::runtime_error("Command must contain scenario_id");
    }

    const auto& scenarioValue = json.at("scenario_id");
    if (scenarioValue.isString()) {
        command.scenario_id = scenarioValue.asString();
    }

    if (command.scenario_id.empty()) {
        throw std::runtime_error("scenario_id must not be empty");
    }

    if (json.contains("detection_regions")) {
        command.detection_regions = parseRegions(json.at("detection_regions"));
    }

    if (json.contains("filter_regions")) {
        command.filter_regions = parseRegions(json.at("filter_regions"));
    }

    if (json.contains("confidence_threshold")) {
        command.threshold = json.at("confidence_threshold").asNumber();
    }

    if (json.contains("fps")) {
        command.fps = json.at("fps").asNumber();
    }

    if (json.contains("activation_code")) {
        command.activation_code = json.at("activation_code").asString();
    }

    if (json.contains("action")) {
        command.action = json.at("action").asString();
    }

    if (json.contains("model")) {
        command.model_info = parseModel(json.at("model"));
    }

    if (json.contains("extra")) {
        command.extra = json.at("extra");
    } else {
        command.extra = simplejson::makeObject();
    }
    return command;
}

std::vector<Command> parseCommandList(const simplejson::JsonValue& json) {
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

