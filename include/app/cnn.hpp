#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app/rtsp.hpp"
#include "app/model.hpp"

namespace app {

class CnnModel: public Model {
public:
    CnnModel(const ModelConfig& config);
    explicit CnnModel(const std::string& model_path);
    ~CnnModel();

    bool load();

    bool isLoaded() const noexcept { return loaded_; }
    const std::string& path() const noexcept { return config_.path; }

    std::vector<Detection> infer(const CapturedFrame& frame) const;

private:
    struct Impl;

    bool loaded_ = false;
    ModelConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app

