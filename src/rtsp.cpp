#include "app/rtsp.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

namespace app {
RtspFrameGrabber::RtspFrameGrabber(RtspConfig config) : config_(std::move(config)) {}

std::string RtspFrameGrabber::buildRtspUrl() const {
    std::ostringstream url;
    url << "rtsp://" << config_.host;
    if (config_.port > 0) {
        url << ':' << config_.port;
    }
    if (!config_.path.empty()) {
        if (config_.path.front() != '/') {
            url << '/';
        }
        url << config_.path;
    }
    return url.str();
}

std::vector<CapturedFrame> RtspFrameGrabber::capture(double fps,
                                                     std::size_t max_frames,
                                                     std::chrono::milliseconds timeout) const
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    if (fps <= 0.0) {
        throw std::invalid_argument("FPS must be positive for RTSP capture");
    }
    if (max_frames == 0) {
        return {};
    }

    // ---------- COMMAND 拼接修正 ----------
    std::ostringstream command;
    command
        << "ffmpeg -nostdin -hide_banner -loglevel error "
        << "-rtsp_transport udp "
        << "-i '" << buildRtspUrl() << "' "
        << "-an "
        << "-vf \"select='eq(pict_type\\\\,I)',fps=" << fps << "\" "
        << "-vcodec mjpeg -q:v 2 -f image2pipe - 2>/dev/null";

    int exitStatus = 0;
    auto closer = [&exitStatus](FILE* f) {
        if (f) {
            exitStatus = pclose(f);
        }
    };

    std::unique_ptr<FILE, decltype(closer)> pipe(popen(command.str().c_str(), "r"), closer);
    if (!pipe) {
        throw std::runtime_error("Failed to execute ffmpeg for RTSP capture");
    }

    std::vector<std::uint8_t> frameBuffer;
    frameBuffer.reserve(1024 * 1024 * 3);

    std::array<std::uint8_t, 4096> buffer{};
    std::uint8_t previous = 0;
    bool havePrevious = false;
    bool capturing = false;
    std::size_t frameIndex = 0;
    auto startTime = std::chrono::steady_clock::now();
    bool enforceTimeout = timeout.count() > 0;

    auto finalizeFrame = [&](bool force) {
        if (!capturing && frameBuffer.empty()) {
            return;
        }
        if (!frameBuffer.empty() &&
            (force ||
                (frameBuffer.size() >= 2 &&
                frameBuffer[frameBuffer.size() - 2] == 0xFF &&
                frameBuffer.back() == 0xD9))) {

            CapturedFrame frame;
            frame.timestamp = static_cast<double>(frameIndex) / fps;
            frame.data = frameBuffer;
            frame.format = "jpeg";
            frames.push_back(std::move(frame));
            frameBuffer.clear();
            capturing = false;
            ++frameIndex;
        }
    };

    while (frameIndex < max_frames) {
        std::size_t bytesRead = std::fread(buffer.data(), 1, buffer.size(), pipe.get());
        if (bytesRead == 0) break;

        for (std::size_t i = 0; i < bytesRead && frameIndex < max_frames; ++i) {
            std::uint8_t byte = buffer[i];
            if (!capturing) {
                if (havePrevious && previous == 0xFF && byte == 0xD8) {
                    frameBuffer.clear();
                    frameBuffer.push_back(previous);
                    frameBuffer.push_back(byte);
                    capturing = true;
                    havePrevious = false;
                    continue;
                }
            } else {
                frameBuffer.push_back(byte);
                if (havePrevious && previous == 0xFF && byte == 0xD9) {
                    finalizeFrame(false);
                }
            }
            havePrevious = true;
            previous = byte;
        }

        if (enforceTimeout) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
            if (elapsed > timeout + std::chrono::milliseconds(200)) {
                break;
            }
        }
    }

    finalizeFrame(true);

    pipe.reset();

    if (frames.empty()) {
        throw std::runtime_error("RTSP capture produced no frames");
    }

    return frames;
}


}  // namespace app

