#include "app/model.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <random>

namespace app {

Model::Model(ModelConfig config) : config_(std::move(config)) {}

bool Model::load() {
    return true;
}

CnnModel::CnnModel(ModelConfig config) : Model(std::move(config)) {}

std::vector<Detection> CnnModel::infer(const Frame &frame, double scenario_threshold) {
    std::vector<Detection> detections;
    double conf = random_confidence(frame.image_path + config_.id + "cnn");
    double min_threshold = std::max(config_.threshold, scenario_threshold);
    if (conf >= min_threshold) {
        Detection detection;
        detection.scenario_id = "";
        detection.model_id = config_.id;
        detection.label = config_.labels.empty() ? "cnn_label" : config_.labels.front();
        detection.confidence = conf;
        detection.bbox = pseudo_box(frame.image_path + config_.id);
        detection.image_path = frame.image_path;
        detection.timestamp = frame.timestamp;
        detections.push_back(detection);
    }
    return detections;
}

YoloModel::YoloModel(ModelConfig config) : Model(std::move(config)) {}

std::vector<Detection> YoloModel::infer(const Frame &frame, double scenario_threshold) {
    std::vector<Detection> detections;
    double conf = random_confidence(frame.image_path + config_.id + "yolo");
    double min_threshold = std::max(config_.threshold, scenario_threshold);
    if (conf >= min_threshold) {
        Detection detection;
        detection.scenario_id = "";
        detection.model_id = config_.id;
        detection.label = config_.labels.empty() ? "yolo_label" : config_.labels.front();
        detection.confidence = conf;
        detection.bbox = pseudo_box(frame.image_path + config_.id + "y");
        detection.image_path = frame.image_path;
        detection.timestamp = frame.timestamp;
        detections.push_back(detection);
    }
    return detections;
}

std::unique_ptr<Model> create_model(const ModelConfig &config) {
    if (config.type == "cnn") {
        return std::make_unique<CnnModel>(config);
    }
    if (config.type == "yolo") {
        return std::make_unique<YoloModel>(config);
    }
    return nullptr;
}

double random_confidence(const std::string &seed) {
    std::size_t hash = std::hash<std::string>{}(seed);
    std::mt19937 rng(static_cast<unsigned int>(hash));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

std::array<int, 4> pseudo_box(const std::string &seed) {
    std::size_t hash = std::hash<std::string>{}(seed);
    int x = static_cast<int>((hash >> 8) % 400);
    int y = static_cast<int>((hash >> 16) % 300);
    int w = 50 + static_cast<int>((hash >> 24) % 150);
    int h = 50 + static_cast<int>((hash >> 32) % 150);
    return {x, y, w, h};
}

} // namespace app

