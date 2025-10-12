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

struct ModelInfo {
    std::string id;
    std::string type;
    std::string path;
};

Region parseRegion(const simplejson::JsonValue& value);
std::vector<Region> parseRegions(const simplejson::JsonValue& value);

} // namespace app