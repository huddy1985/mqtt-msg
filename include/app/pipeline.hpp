#pragma once

#include <vector>

#include "app/command.hpp"
#include "app/config.hpp"

namespace app {

struct DetectionResult {
    std::string label;
    Region region;
    double confidence = 0.0;
    bool filtered = false;
};

struct FrameResult {
    double timestamp = 0.0;
    std::vector<DetectionResult> detections;
};

struct AnalysisResult {
    std::string scenario_id;
    ModelInfo model;
    std::vector<FrameResult> frames;
};

simplejson::JsonValue toJson(const AnalysisResult& result);

class ProcessingPipeline {
public:
    explicit ProcessingPipeline(AppConfig config);

    AnalysisResult process(const Command& command) const;

    const AppConfig& config() const { return config_; }

private:
    AppConfig config_;
};

}  // namespace app

