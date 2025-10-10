#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app/rtsp.hpp"

namespace app {

struct CnnPrediction {
    std::string label;
    double confidence = 0.0;
};

class CnnModel {
public:
    CnnModel();
    explicit CnnModel(const std::string& model_path);
    ~CnnModel();

    void load(const std::string& model_path);

    bool isLoaded() const noexcept { return loaded_; }
    const std::string& path() const noexcept { return model_path_; }

    std::vector<CnnPrediction> infer(const CapturedFrame& frame) const;

private:
    struct Impl;

    bool loaded_ = false;
    std::string model_path_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app

