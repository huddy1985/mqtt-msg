#pragma once

#include "app/config.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace app {

struct Detection {
    std::string scenario_id;
    std::string model_id;
    std::string label;
    double confidence{0.0};
    std::array<int, 4> bbox{0, 0, 0, 0};
    std::string image_path;
    std::string timestamp;
};

struct Frame {
    std::string image_path;
    std::string timestamp;
};

class Model {
public:
    explicit Model(ModelConfig config);
    virtual ~Model() = default;

    virtual bool load();
    virtual std::vector<Detection> infer(const Frame &frame, double scenario_threshold) = 0;

    const ModelConfig &config() const { return config_; }

protected:
    ModelConfig config_;
};

class CnnModel : public Model {
public:
    explicit CnnModel(ModelConfig config);

    std::vector<Detection> infer(const Frame &frame, double scenario_threshold) override;
};

class YoloModel : public Model {
public:
    explicit YoloModel(ModelConfig config);

    std::vector<Detection> infer(const Frame &frame, double scenario_threshold) override;
};

std::unique_ptr<Model> create_model(const ModelConfig &config);

double random_confidence(const std::string &seed);
std::array<int, 4> pseudo_box(const std::string &seed);

} // namespace app

