#include "app/scenario.hpp"

#include <iostream>

namespace app {

Scenario::Scenario(ScenarioDefinition definition, std::string config_path)
    : definition_(std::move(definition)), config_path_(std::move(config_path)) {}

bool Scenario::load_models() {
    models_.clear();
    for (const auto &model_config : definition_.models) {
        auto model = create_model(model_config);
        if (!model) {
            std::cerr << "Unsupported model type for scenario " << definition_.id << "\n";
            return false;
        }
        if (!model->load()) {
            std::cerr << "Failed to load model " << model_config.id << "\n";
            return false;
        }
        models_.push_back(std::move(model));
    }
    return true;
}

std::vector<Detection> Scenario::analyze(const Frame &frame, double override_threshold) {
    std::vector<Detection> results;
    double scenario_threshold = override_threshold > 0.0 ? override_threshold : definition_.threshold;
    for (const auto &model : models_) {
        auto detections = model->infer(frame, scenario_threshold);
        for (auto &detection : detections) {
            detection.scenario_id = definition_.id;
            results.push_back(detection);
        }
    }
    return results;
}

} // namespace app

