#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app/command.hpp"
#include "app/rtsp.hpp"
#include "app/model.hpp"

namespace app {

class YoloModel : public Model {
public:
    YoloModel(const ScenarioDefinition& config);
    explicit YoloModel(const std::string& model_path);
    ~YoloModel();

    bool load();
    bool release();
    std::string model_type();

    bool isLoaded() const noexcept { return loaded_; }
    const std::string& path() const noexcept { return config_.model.path; }

    std::vector<Detection> infer(const CapturedFrame& frame) const;

private:
    struct Impl;

    std::string type;
    bool loaded_ = false;
    ScenarioDefinition config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app

