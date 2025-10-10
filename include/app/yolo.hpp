#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app/command.hpp"
#include "app/rtsp.hpp"

namespace app {

struct YoloDetection {
    Region region;
    std::string label;
    double confidence = 0.0;
};

class YoloModel {
public:
    YoloModel();
    explicit YoloModel(const std::string& model_path);
    ~YoloModel();

    void load(const std::string& model_path);

    bool isLoaded() const noexcept { return loaded_; }
    const std::string& path() const noexcept { return model_path_; }

    std::vector<YoloDetection> infer(const CapturedFrame& frame, const std::vector<Region>& hints) const;

private:
    struct Impl;

    bool loaded_ = false;
    std::string model_path_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app

