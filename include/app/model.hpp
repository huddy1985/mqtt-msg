#pragma once

#include "app/config.hpp"
#include "app/command.hpp"
#include "app/rtsp.hpp"

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

    Region region;
};

class Model {
public:
    explicit Model(ScenarioDefinition config);
    virtual ~Model() = default;

    virtual bool load() = 0;
    virtual bool release() = 0;
    virtual std::vector<Detection> infer(const CapturedFrame& frame) const = 0;

    const ScenarioDefinition &config() const { return config_; }

protected:
    ScenarioDefinition config_;
};

std::unique_ptr<Model> create_model(const ScenarioDefinition &config);

double random_confidence(const std::string &seed);
std::array<int, 4> pseudo_box(const std::string &seed);

} // namespace app