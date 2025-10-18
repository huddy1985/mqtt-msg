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

    std::vector<CapturedFrame> frames;
    frames.reserve(max_frames);

    const auto frame_interval = std::chrono::duration<double>(1.0 / fps);
    auto last_ts = t0;

    const std::string path = config_.path;
    const bool use_camera = (path.rfind("/dev/video", 0) == 0);

    if (use_camera) {
        // ===== 摄像头分支（仅当 path 形如 /dev/videoX）=====
        cv::VideoCapture cap;
        // 从 /dev/videoN 中解析设备索引
        int index = 0;
        try {
            index = std::stoi(path.substr(std::string("/dev/video").size()));
        } catch (...) {
            index = 0;
        }
        if (!cap.open(index)) {
            throw std::runtime_error("Failed to open local camera: " + path);
        }

        int w = (config_.width  > 0 ? config_.width  : 1920);
        int h = (config_.height > 0 ? config_.height : 1080);

        cap.set(cv::CAP_PROP_FRAME_WIDTH,  w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        cap.set(cv::CAP_PROP_FPS,          fps);
        
        // 若摄像头支持 MJPG，可降低 CPU（不支持时忽略）
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

        while (frames.size() < max_frames) {
            const auto now = clock::now();
            if (now - t0 > timeout) break;

            // 控制采样节奏
            const auto since_last = now - last_ts;
            if (since_last < frame_interval) {
                std::this_thread::sleep_for(frame_interval - since_last);
            }
            last_ts = clock::now();

            cv::Mat mat;
            if (!cap.read(mat) || mat.empty()) {
                // 允许短暂获取失败，直到超时
                if (clock::now() - t0 > timeout) break;
                continue;
            }

            // 编码为 JPEG
            std::vector<uchar> buf;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
            if (!cv::imencode(".jpg", mat, buf, params)) {
                if (clock::now() - t0 > timeout) break;
                continue;
            }

            CapturedFrame f;
            f.timestamp = std::chrono::duration<double>(clock::now() - t0).count();
            f.data.assign(buf.begin(), buf.end());
            f.format = "jpeg";
            frames.emplace_back(std::move(f));
        }

        cap.release();

        if (frames.empty()) {
            throw std::runtime_error("RTSP capture produced no frames (source=camera:" + path + ")");
        }
        return frames;
    }

    std::ostringstream command;
    command << "ffmpeg -nostdin -rtsp_transport tcp -loglevel debug ";
    if (timeout.count() > 0) {
        // stimeout expects microseconds
        command << "-stimeout " << (timeout.count() * 1000) << ' ';
    }
    command << "-i '" << buildRtspUrl() << "' ";
    command << "-vf fps=" << fps << ' ';
    command << "-vframes " << max_frames << ' ';
    command << "-vcodec mjpeg -q:v 2 -f image2pipe - 2>/dev/null";

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
        if (!frameBuffer.empty() && (force || (!frameBuffer.empty() && frameBuffer.size() >= 2 &&
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
        if (bytesRead == 0) {
            if (std::feof(pipe.get())) {
                break;
            }
            if (std::ferror(pipe.get())) {
                throw std::runtime_error("Error while reading RTSP frame data");
            }
            break;
        }

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

            if (capturing && frameBuffer.size() == 1) {
                // ensure we keep the first byte even when we skip continue above
                frameBuffer[0] = previous;
            }
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

    if (exitStatus != 0) {
        std::ostringstream error;
        error << "ffmpeg exited with status " << exitStatus;
        throw std::runtime_error(error.str());
    }

    return frames;
}

}  // namespace app

