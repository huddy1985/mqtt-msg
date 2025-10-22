#pragma once

#include <string>
#include <vector>
#include <shared_mutex>

#include "app/command.hpp"
#include "app/config.hpp"
#include "app/rtsp.hpp"
#include "app/scenario.hpp"

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
    std::string image_path;
};

struct AnalysisResult {
    std::string scenario_id;
    ModelInfo model;
    std::vector<FrameResult> frames;
};

simplejson::JsonValue toJson(const AnalysisResult& result);

class ProcessingPipeline {
public:
    explicit ProcessingPipeline(AppConfig config, ConfigStore *store);

    std::vector<AnalysisResult> process(const Command& command);

    const AppConfig& config() const { return config_; }

    void remove_inactive(const std::string &scenario_id);
    void add_missing(const std::string &scenario_id);

private:
    const ScenarioConfig* findScenario(const std::string& scenario_id) const;

private:
    AppConfig config_;
    RtspFrameGrabber frame_grabber_;
    ConfigStore *store_{nullptr};
    mutable std::shared_mutex scenarios_mutex_;
    std::map<std::string, std::shared_ptr<Scenario>> active_scenarios_;
};

}  // namespace app

