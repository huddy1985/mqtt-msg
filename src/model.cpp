#include "app/model.hpp"
#include "app/cnn.hpp"
#include "app/yolo.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <random>

namespace app {

Model::Model(ModelConfig config) : config_(std::move(config)) {}

bool Model::load() {
    return true;
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
