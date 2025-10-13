#include "app/scenario.hpp"

#include <iostream>

namespace app {

Scenario::Scenario(ScenarioDefinition definition, std::string config_path)
    : definition_(std::move(definition)), config_path_(std::move(config_path)) {}

bool Scenario::load_models() {
    model_ = create_model(definition_);
    if (!model_) {
        std::cerr << "Unsupported model type for scenario " << definition_.id << "\n";
        return false;
    }
    if (!model_->load()) {
        std::cerr << "Failed to load model " << definition_.id << "\n";
        return false;
    }
    return true;
}

bool Scenario::release_models() 
{
    if (!model_) {
        std::cerr << "Model is not loaded " << definition_.id << "\n";
        return false;
    }
    if (model_->release()) {
        std::cerr << "Failed to load model " << definition_.id << "\n";
        return false;
    }
    return true;
}

std::string Scenario::model_type()
{
    return model_->model_type();
}

std::vector<Detection> Scenario::analyze(const CapturedFrame &frame) {
    std::vector<Detection> results;
    double scenario_threshold = definition_.threshold;
    
    auto detections = model_->infer(frame);
    for (auto &detection : detections) {
        detection.scenario_id = definition_.id;
        results.push_back(detection);
    }
    return results;
}

} // namespace app