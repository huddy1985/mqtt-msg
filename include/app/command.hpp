#pragma once

#include "app/json.hpp"

#include <string>
#include <vector>

namespace app {

struct Region {
    int x{0};
    int y{0};
    int width{0};
    int height{0};
};

struct Command {
    std::vector<std::string> scenario_ids;
    std::vector<Region> regions;
    std::vector<Region> filters;
    double threshold{0.5};
    int frame_rate{1};
    std::string activation_code;
    Json extra;
};

Command parse_command(const Json &json);

} // namespace app

