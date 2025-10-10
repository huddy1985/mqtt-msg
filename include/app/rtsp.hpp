#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "app/config.hpp"

namespace app {

struct CapturedFrame {
    double timestamp = 0.0;                 // seconds since capture start
    std::vector<std::uint8_t> data;         // encoded image bytes (JPEG)
    std::string format = "jpeg";           // image format identifier
};

class RtspFrameGrabber {
public:
    explicit RtspFrameGrabber(RtspConfig config);

    std::vector<CapturedFrame> capture(double fps,
                                       std::size_t max_frames,
                                       std::chrono::milliseconds timeout) const;

    const RtspConfig& config() const { return config_; }

private:
    RtspConfig config_;

    std::string buildRtspUrl() const;
};

}  // namespace app

