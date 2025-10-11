#pragma once

#include "app/model.hpp"

#include <string>
#include <vector>

namespace app {

class RtspFrameGrabber {
public:
    RtspFrameGrabber(std::string url, std::string output_dir, int frame_rate);

    std::vector<Frame> capture_frames(int count) const;

private:
    std::string url_;
    std::string output_dir_;
    int frame_rate_{1};
};

} // namespace app

