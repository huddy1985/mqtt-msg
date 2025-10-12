#include "app/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "app/cnn.hpp"
#include "app/yolo.hpp"

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
        if (!frame.image_path.empty()) {
            frameMap["image_path"] = frame.image_path;
        }

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

namespace {

std::string sanitizeName(const std::string& name) {
    std::string sanitized;
    sanitized.reserve(name.size());
    for (char ch : name) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.') {
            sanitized.push_back(ch);
        }
    }
    if (sanitized.empty()) {
        sanitized = "captures";
    }
    return sanitized;
}

std::filesystem::path ensureCaptureDirectory(const std::string& serviceName, const std::string& scenarioId) {
    std::filesystem::path directory = std::filesystem::path("captures") / sanitizeName(serviceName);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        // fall back to base captures directory if nested creation fails
        directory = std::filesystem::path("captures");
        std::filesystem::create_directories(directory, ec);
    }
    if (!scenarioId.empty()) {
        std::filesystem::path scenarioDir = directory / sanitizeName(scenarioId);
        std::error_code scenarioEc;
        std::filesystem::create_directories(scenarioDir, scenarioEc);
        if (!scenarioEc) {
            directory = scenarioDir;
        }
    }
    return directory;
}

std::string saveFrameToDisk(const std::filesystem::path& directory,
                            std::size_t index,
                            const CapturedFrame& frame) {
    if (frame.data.empty()) {
        return {};
    }

    std::ostringstream name;
    name << "frame_" << std::setw(6) << std::setfill('0') << index;
    std::string extension = ".jpg";
    if (!frame.format.empty()) {
        if (frame.format == "png") {
            extension = ".png";
        } else if (frame.format == "jpeg" || frame.format == "jpg") {
            extension = ".jpg";
        }
    }

    std::filesystem::path filePath = directory / (name.str() + extension);
    std::ofstream output(filePath, std::ios::binary);
    if (!output) {
        return {};
    }
    output.write(reinterpret_cast<const char*>(frame.data.data()), static_cast<std::streamsize>(frame.data.size()));
    output.close();
    return filePath.generic_string();
}

}  // namespace

ProcessingPipeline::ProcessingPipeline(AppConfig config, ConfigStore *store)
    : config_(std::move(config)), frame_grabber_(config_.rtsp), store_(store) {
        if (!config_.active_scenarios.empty()) {
            sync_active_scenarios(config_.active_scenarios);
        }
    }

const ScenarioConfig* ProcessingPipeline::findScenario(const std::string& scenario_id) const {
    auto it = config_.scenario_lookup.find(scenario_id);
    if (it == config_.scenario_lookup.end()) {
        return nullptr;
    }
    std::size_t index = it->second;
    if (index >= config_.scenarios.size()) {
        return nullptr;
    }
    return &config_.scenarios[index];
}

/* void ProcessingPipeline::persistActiveScenarios() const {
    if (config_.source_path.empty()) {
        return;
    }

    try {
        simplejson::JsonValue root = simplejson::parseFile(config_.source_path);
        if (!root.isObject()) {
            return;
        }

        auto& obj = root.asObject();
        auto it = obj.find("scenarios");
        if (it == obj.end()) {
            return;
        }

        auto& scenarios = it->second.asArray();
        for (auto& scenarioValue : scenarios) {
            if (!scenarioValue.isObject()) {
                continue;
            }
            auto& scenarioObj = scenarioValue.asObject();
            auto idIt = scenarioObj.find("id");
            if (idIt == scenarioObj.end()) {
                continue;
            }
            std::string id = idIt->second.asString();
            auto lookupIt = config_.scenario_lookup.find(id);
            if (lookupIt == config_.scenario_lookup.end()) {
                continue;
            }
            std::size_t index = lookupIt->second;
            if (index >= config_.scenarios.size()) {
                continue;
            }
            bool active = config_.scenarios[index].active;
            scenarioObj["active"] = active;
        }

        std::ofstream output(config_.source_path);
        if (output) {
            output << root.dump(2);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Failed to persist scenario activation state: " << ex.what() << "\n";
    }
} */

/* void ProcessingPipeline::setActiveScenarios(const std::vector<std::string>& scenario_ids) {
    std::vector<bool> desired(config_.scenarios.size(), false);
    for (const auto& id : scenario_ids) {
        auto it = config_.scenario_lookup.find(id);
        if (it != config_.scenario_lookup.end() && it->second < desired.size()) {
            desired[it->second] = true;
        }
    }

    bool changed = false;
    for (std::size_t index = 0; index < config_.scenarios.size(); ++index) {
        bool newState = index < desired.size() ? desired[index] : false;
        if (config_.scenarios[index].active != newState) {
            config_.scenarios[index].active = newState;
            changed = true;
        }
    }

    if (changed) {
        persistActiveScenarios();
    }
} */

void ProcessingPipeline::sync_active_scenarios(const std::vector<std::string> &scenario_ids) {
    remove_inactive(scenario_ids);
    add_missing(scenario_ids);
    config_.active_scenarios = scenario_ids;
}

void ProcessingPipeline::remove_inactive(const std::vector<std::string> &scenario_ids) {
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

void ProcessingPipeline::add_missing(const std::vector<std::string> &scenario_ids) {
    for (const auto &id : scenario_ids) {
        if (active_scenarios_.find(id) != active_scenarios_.end()) {
            continue;
        }
        if (!store_) {
            std::cerr << "No configuration store available to load scenario " << id << "\n";
            continue;
        }

        auto path_it = config_.scenarios.begin();

        for (; path_it != config_.scenarios.end(); path_it++) {
            if (path_it->id == id) {
                break;
            }
        }
        
        if (path_it == config_.scenarios.end()) {
            std::cerr << "Scenario " << id << " not found in configuration map\n";
            continue;
        }
        try {
            ScenarioDefinition def = store_->load_scenario_file(path_it->config_path);
            if (def.id.empty()) {
                def.id = id;
            }
            auto scenario = std::make_unique<Scenario>(def, path_it->config_path);
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

std::vector<AnalysisResult> ProcessingPipeline::process(const Command& command) {
    if (command.scenario_ids.empty()) {
        throw std::runtime_error("Command must define at least one scenario");
    }

    std::vector<AnalysisResult> results;
    results.reserve(command.scenario_ids.size());

    for (const auto& scenarioId : command.scenario_ids) {
        const ScenarioConfig* scenarioConfig = findScenario(scenarioId);
        if (!scenarioConfig) {
            throw std::runtime_error("Unknown scenario: " + scenarioId);
        }

        if (!scenarioConfig->active) {
            std::cerr << "Skipping inactive scenario: " << scenarioConfig->id << "\n";
            continue;
        }

        auto active_scenario = active_scenarios_.find(scenarioId);
        if (active_scenario == active_scenarios_.end()) {
            continue;
        }

        AnalysisResult result;
        result.scenario_id = scenarioConfig->id;
        result.model = scenarioConfig->model;

        double fps = command.fps > 0.0 ? command.fps : 1.0;
        double interval = 1.0 / fps;

        std::vector<Region> regions = command.detection_regions;
        if (regions.empty()) {
            regions.push_back(Region{0, 0, 0, 0});
        }

        bool useCnn = (result.model.type == "cnn");
        bool useYolo = (!useCnn && result.model.type.rfind("yolo", 0) == 0);
        std::size_t frameCount = regions.size();

        std::unique_ptr<CnnModel> cnnModel;
        std::unique_ptr<YoloModel> yoloModel;

        ScenarioDefinition _config;
        if (useCnn) {
            try {
                cnnModel = std::make_unique<CnnModel>(_config);
            } catch (const std::exception& ex) {
                std::cerr << "CNN model load failed: " << ex.what() << "\n";
            }
            if (!frameCount) {
                frameCount = 1;
            }
        } else if (useYolo) {
            try {
                yoloModel = std::make_unique<YoloModel>(_config);
            } catch (const std::exception& ex) {
                std::cerr << "YOLO model load failed: " << ex.what() << "\n";
            }
            if (!frameCount) {
                frameCount = 1;
            }
        }

        std::vector<CapturedFrame> capturedFrames;
        try {
            capturedFrames = frame_grabber_.capture(fps, frameCount, std::chrono::milliseconds(5000));
        } catch (const std::exception& ex) {
            std::cerr << "RTSP capture failed: " << ex.what() << "\n";
        }

        std::filesystem::path captureDir = ensureCaptureDirectory(config_.service.name, scenarioConfig->id);

        for (std::size_t index = 0; index < frameCount; ++index) {
            FrameResult frame;
            if (index < capturedFrames.size()) {
                frame.timestamp = capturedFrames[index].timestamp;
                std::string path = saveFrameToDisk(captureDir, index, capturedFrames[index]);
                frame.image_path = path;
            } else {
                frame.timestamp = index * interval;
            }

            if (useCnn && cnnModel && cnnModel->isLoaded()) {
                const CapturedFrame* frameData = (index < capturedFrames.size()) ? &capturedFrames[index] : nullptr;
                CapturedFrame synthetic;
                if (!frameData) {
                    synthetic.timestamp = frame.timestamp;
                    synthetic.format = "synthetic";
                    synthetic.data.reserve(regions.size() * 4 + scenarioConfig->id.size());
                    for (const auto& region : regions) {
                        synthetic.data.push_back(static_cast<std::uint8_t>((region.x1 + region.y1) & 0xFF));
                        synthetic.data.push_back(static_cast<std::uint8_t>((region.x2 + region.y2) & 0xFF));
                    }
                    for (char ch : scenarioConfig->id) {
                        synthetic.data.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(ch)));
                    }
                    frameData = &synthetic;
                }

                auto predictions = cnnModel->infer(*frameData);
                if (predictions.empty()) {
                    predictions.push_back(Detection{"unknown"});
                }

                for (std::size_t detIndex = 0; detIndex < predictions.size(); ++detIndex) {
                    Region region = detIndex < regions.size() ? regions[detIndex] : Region{0, 0, 0, 0};
                    bool filtered = isFiltered(region, command.filter_regions);
                    DetectionResult detection;
                    detection.region = region;
                    detection.filtered = filtered;
                    detection.label = predictions[detIndex].label;
                    detection.confidence = predictions[detIndex].confidence;
                    frame.detections.push_back(std::move(detection));
                }
            } else if (useYolo && yoloModel && yoloModel->isLoaded()) {
                const CapturedFrame* frameData = (index < capturedFrames.size()) ? &capturedFrames[index] : nullptr;
                CapturedFrame synthetic;
                if (!frameData) {
                    synthetic.timestamp = frame.timestamp;
                    synthetic.format = "synthetic";
                    synthetic.data.reserve(regions.size() * 4 + scenarioConfig->id.size());
                    for (const auto& region : regions) {
                        synthetic.data.push_back(static_cast<std::uint8_t>((region.x1 ^ region.y2) & 0xFF));
                        synthetic.data.push_back(static_cast<std::uint8_t>((region.x2 ^ region.y1) & 0xFF));
                    }
                    for (char ch : scenarioConfig->id) {
                        synthetic.data.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(ch)));
                    }
                    frameData = &synthetic;
                }

                auto detections = yoloModel->infer(*frameData);
                for (const auto& yoloDet : detections) {
                    Region region = yoloDet.region;
                    bool filtered = isFiltered(region, command.filter_regions);
                    DetectionResult detection;
                    detection.region = region;
                    detection.filtered = filtered;
                    detection.label = yoloDet.label;
                    detection.confidence = yoloDet.confidence;
                    frame.detections.push_back(std::move(detection));
                }
            } else {
                const auto& region = regions[index % regions.size()];
                bool filtered = isFiltered(region, command.filter_regions);
                DetectionResult detection = makeDetection(region, result.model, command.threshold, filtered);
                frame.detections.push_back(detection);
            }

            result.frames.push_back(std::move(frame));
        }

        results.push_back(std::move(result));
    }

    return results;
}

}  // namespace app

