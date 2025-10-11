#include "app/pipeline.hpp"

#include <algorithm>
#include <iostream>

namespace app {

Pipeline::Pipeline(LocalConfig config, ConfigStore *store)
    : config_(std::move(config)), store_(store) {
    if (!config_.active_scenarios.empty()) {
        sync_active_scenarios(config_.active_scenarios);
    }
}

void Pipeline::sync_active_scenarios(const std::vector<std::string> &scenario_ids) {
    remove_inactive(scenario_ids);
    add_missing(scenario_ids);
    config_.active_scenarios = scenario_ids;
}

std::vector<ScenarioResult> Pipeline::analyze_frame(const Frame &frame, double command_threshold) {
    std::vector<ScenarioResult> results;
    for (auto &kv : active_scenarios_) {
        ScenarioResult result;
        result.scenario_id = kv.first;
        result.detections = kv.second->analyze(frame, command_threshold);
        results.push_back(result);
    }
    return results;
}

void Pipeline::remove_inactive(const std::vector<std::string> &scenario_ids) {
    std::vector<std::string> to_remove;
    for (const auto &kv : active_scenarios_) {
        if (std::find(scenario_ids.begin(), scenario_ids.end(), kv.first) == scenario_ids.end()) {
            to_remove.push_back(kv.first);
        }
    }
    for (const auto &id : to_remove) {
        std::cout << "Deactivating scenario " << id << "\n";
        active_scenarios_.erase(id);
    }
}

void Pipeline::add_missing(const std::vector<std::string> &scenario_ids) {
    for (const auto &id : scenario_ids) {
        if (active_scenarios_.find(id) != active_scenarios_.end()) {
            continue;
        }
        if (!store_) {
            std::cerr << "No configuration store available to load scenario " << id << "\n";
            continue;
        }
        auto path_it = config_.scenario_files.find(id);
        if (path_it == config_.scenario_files.end()) {
            std::cerr << "Scenario " << id << " not found in configuration map\n";
            continue;
        }
        try {
            ScenarioDefinition def = store_->load_scenario_file(path_it->second);
            if (def.id.empty()) {
                def.id = id;
            }
            auto scenario = std::make_unique<Scenario>(def, path_it->second);
            if (!scenario->load_models()) {
                std::cerr << "Failed to load models for scenario " << id << "\n";
                continue;
            }
            std::cout << "Activating scenario " << id << "\n";
            active_scenarios_.emplace(id, std::move(scenario));
        } catch (const std::exception &ex) {
            std::cerr << "Error loading scenario " << id << ": " << ex.what() << "\n";
        }
    }
}

} // namespace app

