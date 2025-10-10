#include "app/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
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

std::filesystem::path ensureCaptureDirectory(const std::string& serviceName) {
    std::string sanitized;
    sanitized.reserve(serviceName.size());
    for (char ch : serviceName) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.') {
            sanitized.push_back(ch);
        }
    }
    if (sanitized.empty()) {
        sanitized = "captures";
    }

    std::filesystem::path directory = std::filesystem::path("captures") / sanitized;
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        // fall back to base captures directory if nested creation fails
        directory = std::filesystem::path("captures");
        std::filesystem::create_directories(directory, ec);
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

ProcessingPipeline::ProcessingPipeline(AppConfig config)
    : config_(std::move(config)), frame_grabber_(config_.rtsp) {}

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

    std::size_t frameCount = regions.size();
    std::vector<CapturedFrame> capturedFrames;
    try {
        capturedFrames = frame_grabber_.capture(fps, frameCount, std::chrono::milliseconds(5000));
    } catch (const std::exception& ex) {
        std::cerr << "RTSP capture failed: " << ex.what() << "\n";
    }

    std::filesystem::path captureDir = ensureCaptureDirectory(config_.service.name);

    for (std::size_t index = 0; index < frameCount; ++index) {
        const auto& region = regions[index];
        bool filtered = isFiltered(region, command.filter_regions);
        DetectionResult detection = makeDetection(region, result.model, command.threshold, filtered);

        FrameResult frame;
        if (index < capturedFrames.size()) {
            frame.timestamp = capturedFrames[index].timestamp;
            std::string path = saveFrameToDisk(captureDir, index, capturedFrames[index]);
            frame.image_path = path;
        } else {
            frame.timestamp = index * interval;
        }
        frame.detections.push_back(detection);
        result.frames.push_back(frame);
    }

    return result;
}

}  // namespace app

