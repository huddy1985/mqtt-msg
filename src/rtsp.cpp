#include "app/rtsp.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace app {

RtspFrameGrabber::RtspFrameGrabber(std::string url, std::string output_dir, int frame_rate)
    : url_(std::move(url)), output_dir_(std::move(output_dir)), frame_rate_(frame_rate) {}

std::vector<Frame> RtspFrameGrabber::capture_frames(int count) const {
    std::vector<Frame> frames;
    if (count <= 0) {
        return frames;
    }

    std::filesystem::create_directories(output_dir_);

    for (int i = 0; i < count; ++i) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &time);
#else
        localtime_r(&time, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << i << ".jpg";
        std::string filename = oss.str();
        std::filesystem::path path = std::filesystem::path(output_dir_) / filename;

        std::ofstream out(path, std::ios::binary);
        out << "Simulated frame for URL: " << url_ << "\n";
        out.close();

        Frame frame;
        frame.image_path = path.string();
        frame.timestamp = oss.str();
        frames.push_back(frame);
    }

    return frames;
}

} // namespace app

