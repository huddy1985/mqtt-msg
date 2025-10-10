#pragma once

#include <string>
#include <vector>

#include "app/json.hpp"

namespace app {

struct Region {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    bool operator==(const Region& other) const {
        return x1 == other.x1 && y1 == other.y1 && x2 == other.x2 && y2 == other.y2;
    }
};

struct Command {
    std::string scenario_id;
    std::vector<Region> detection_regions;
    std::vector<Region> filter_regions;
    double threshold = 0.5;
    double fps = 1.0;
    std::string activation_code;
    simplejson::JsonValue extra;
};

Command parseCommand(const simplejson::JsonValue& json);
std::vector<Command> parseCommandList(const simplejson::JsonValue& json);

}  // namespace app

