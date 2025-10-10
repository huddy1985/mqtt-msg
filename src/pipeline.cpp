#include "app/pipeline.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace app {
namespace {

bool isFiltered(const Region& region, const std::vector<Region>& filters) {
    for (const auto& filter : filters) {
        if (region == filter) {
            return true;
        }
    }
    return false;
}

DetectionResult makeDetection(const Region& region, const ModelInfo& model, double threshold, bool filtered) {
    DetectionResult detection;
    detection.region = region;
    detection.filtered = filtered;
    detection.confidence = std::min(1.0, std::max(threshold, 0.1));
    detection.label = model.id + "_" + model.type;
    return detection;
}

}  // namespace

simplejson::JsonValue toJson(const AnalysisResult& result) {
    simplejson::JsonValue root = simplejson::makeObject();
    auto& obj = root.asObject();
    obj["scenario_id"] = result.scenario_id;

    simplejson::JsonValue modelObj = simplejson::makeObject();
    auto& modelMap = modelObj.asObject();
    modelMap["id"] = result.model.id;
    modelMap["type"] = result.model.type;
    modelMap["path"] = result.model.path;
    obj["model"] = modelObj;

    simplejson::JsonValue framesValue = simplejson::makeArray();
    auto& framesArray = framesValue.asArray();
    for (const auto& frame : result.frames) {
        simplejson::JsonValue frameValue = simplejson::makeObject();
        auto& frameMap = frameValue.asObject();
        frameMap["timestamp"] = frame.timestamp;

        simplejson::JsonValue detectionsValue = simplejson::makeArray();
        auto& detectionsArray = detectionsValue.asArray();
        for (const auto& detection : frame.detections) {
            simplejson::JsonValue detectionValue = simplejson::makeObject();
            auto& detectionMap = detectionValue.asObject();
            detectionMap["label"] = detection.label;
            simplejson::JsonValue regionValue = simplejson::makeArray();
            auto& regionArray = regionValue.asArray();
            regionArray.push_back(detection.region.x1);
            regionArray.push_back(detection.region.y1);
            regionArray.push_back(detection.region.x2);
            regionArray.push_back(detection.region.y2);
            detectionMap["region"] = regionValue;
            detectionMap["confidence"] = detection.confidence;
            detectionMap["filtered"] = detection.filtered;
            detectionsArray.push_back(detectionValue);
        }

        frameMap["detections"] = detectionsValue;
        framesArray.push_back(frameValue);
    }

    obj["frames"] = framesValue;
    return root;
}

ProcessingPipeline::ProcessingPipeline(AppConfig config) : config_(std::move(config)) {}

AnalysisResult ProcessingPipeline::process(const Command& command) const {
    AnalysisResult result;
    result.scenario_id = command.scenario_id;

    auto modelIt = config_.scenario_lookup.find(command.scenario_id);
    if (modelIt == config_.scenario_lookup.end()) {
        throw std::runtime_error("Unknown scenario: " + command.scenario_id);
    }
    result.model = modelIt->second;

    double fps = command.fps > 0.0 ? command.fps : 1.0;
    double interval = 1.0 / fps;

    std::vector<Region> regions = command.detection_regions;
    if (regions.empty()) {
        regions.push_back(Region{0, 0, 0, 0});
    }

    double timestamp = 0.0;
    for (const auto& region : regions) {
        bool filtered = isFiltered(region, command.filter_regions);
        DetectionResult detection = makeDetection(region, result.model, command.threshold, filtered);

        FrameResult frame;
        frame.timestamp = timestamp;
        frame.detections.push_back(detection);
        result.frames.push_back(frame);

        timestamp += interval;
    }

    return result;
}

}  // namespace app

