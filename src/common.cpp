#include "app/common.hpp"

namespace app {

Region parseRegion(const simplejson::JsonValue& value)
{
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

std::vector<Region> parseRegions(const simplejson::JsonValue& value)
{
    std::vector<Region> regions;
    if (!value.isArray()) {
        return regions;
    }
    for (const auto& entry : value.asArray()) {
        regions.push_back(parseRegion(entry));
    }
    return regions;
}

};