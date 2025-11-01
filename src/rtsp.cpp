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
#include <iostream>

#ifdef APP_HAS_RKMPP
extern "C" {
#include <rk_mpi.h>
#include <mpp_buffer.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
}
#endif

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

#ifdef APP_HAS_RKMPP
namespace {

std::size_t findStartCode(const std::vector<std::uint8_t>& buffer, std::size_t offset)
{
    const std::size_t size = buffer.size();
    for (std::size_t i = offset; i + 3 < size; ++i) {
        if (buffer[i] == 0x00 && buffer[i + 1] == 0x00 && buffer[i + 2] == 0x01) {
            return i;
        }
        if (i + 4 < size && buffer[i] == 0x00 && buffer[i + 1] == 0x00 &&
            buffer[i + 2] == 0x00 && buffer[i + 3] == 0x01) {
            return i;
        }
    }
    return std::string::npos;
}

void copyNv12Plane(const std::uint8_t* src, std::uint8_t* dst,
                   int src_stride, int dst_stride, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + y * dst_stride, src + y * src_stride, width);
    }
}

}  // namespace
#endif

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

#ifdef APP_HAS_RKMPP
    try {
        std::ostringstream command;
        command << "ffmpeg -nostdin -hide_banner -loglevel error "
                << "-rtsp_transport udp "
                << "-i '" << buildRtspUrl() << "' "
                << "-an -c copy -bsf:v h264_mp4toannexb -f h264 - 2>/dev/null";

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

        MppCtx ctx = nullptr;
        MppApi* mpi = nullptr;
        if (MPP_RET ret = mpp_create(&ctx, &mpi); ret != MPP_OK || !mpi) {
            throw std::runtime_error("Failed to create MPP decoder context");
        }

        auto ctx_guard = std::unique_ptr<MppCtx, decltype(&mpp_destroy)>(ctx, mpp_destroy);

        if (MPP_RET ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC); ret != MPP_OK) {
            throw std::runtime_error("Failed to initialize MPP decoder");
        }

        MppBufferGroup frame_group = nullptr;
        auto group_guard = std::unique_ptr<MppBufferGroup, decltype(&mpp_buffer_group_put)>(
            nullptr, mpp_buffer_group_put);
        std::size_t frameIndex = 0;
        std::vector<CapturedFrame> frames;
        frames.reserve(max_frames);

        std::vector<std::uint8_t> streamBuffer;
        streamBuffer.reserve(1 << 20);

        std::array<std::uint8_t, 4096> ioBuffer{};
        auto readStart = std::chrono::steady_clock::now();
        bool enforceTimeout = timeout.count() > 0;

        auto pullFrames = [&](bool drain) {
            while (frameIndex < max_frames) {
                MppFrame frame = nullptr;
                MPP_RET ret = mpi->decode_get_frame(ctx, &frame);
                if (ret != MPP_OK) {
                    break;
                }
                if (!frame) {
                    break;
                }

                if (mpp_frame_get_info_change(frame)) {
                    int width = mpp_frame_get_width(frame);
                    int height = mpp_frame_get_height(frame);
                    if (!frame_group) {
                        if (mpp_buffer_group_get_internal(&frame_group, MPP_BUFFER_TYPE_DRM) != MPP_OK) {
                            mpp_frame_deinit(&frame);
                            throw std::runtime_error("Failed to allocate MPP buffer group");
                        }
                        group_guard.reset(frame_group);
                        mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, frame_group);
                    }
                    mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
                    mpp_frame_deinit(&frame);
                    continue;
                }

                if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
                    mpp_frame_deinit(&frame);
                    if (!drain) {
                        continue;
                    }
                    break;
                }

                MppBuffer buffer = mpp_frame_get_buffer(frame);
                if (!buffer) {
                    mpp_frame_deinit(&frame);
                    continue;
                }

                const int width = mpp_frame_get_width(frame);
                const int height = mpp_frame_get_height(frame);
                const int hor_stride = mpp_frame_get_hor_stride(frame);
                const int ver_stride = mpp_frame_get_ver_stride(frame);

                const std::uint8_t* src = static_cast<std::uint8_t*>(mpp_buffer_get_ptr(buffer));
                if (!src) {
                    mpp_frame_deinit(&frame);
                    continue;
                }

                std::vector<std::uint8_t> nv12;
                nv12.resize(static_cast<std::size_t>(width) * height * 3 / 2);

                const std::uint8_t* src_y = src;
                const std::uint8_t* src_uv = src + static_cast<std::size_t>(hor_stride) * ver_stride;

                std::uint8_t* dst_y = nv12.data();
                std::uint8_t* dst_uv = dst_y + static_cast<std::size_t>(width) * height;

                copyNv12Plane(src_y, dst_y, hor_stride, width, width, height);
                copyNv12Plane(src_uv, dst_uv, hor_stride, width, width, height / 2);

                CapturedFrame out;
                out.timestamp = static_cast<double>(frameIndex) / fps;
                out.data = std::move(nv12);
                out.format = "nv12";
                out.width = width;
                out.height = height;
                out.stride = width;
                out.uv_stride = width;

                frames.push_back(std::move(out));
                ++frameIndex;

                mpp_frame_deinit(&frame);

                if (frameIndex >= max_frames) {
                    break;
                }
            }
        };

        auto feedPacket = [&](const std::uint8_t* data, std::size_t size, bool eos) {
            if (size == 0 && !eos) {
                return;
            }
            MppPacket packet = nullptr;
            MPP_RET ret = mpp_packet_init(&packet, const_cast<std::uint8_t*>(data), size);
            if (ret != MPP_OK) {
                throw std::runtime_error("Failed to create MPP packet");
            }
            if (eos) {
                mpp_packet_set_eos(packet);
            }
            ret = mpi->decode_put_packet(ctx, packet);
            mpp_packet_deinit(&packet);
            if (ret != MPP_OK) {
                throw std::runtime_error("MPP decode_put_packet failed");
            }
            pullFrames(eos);
        };

        while (frameIndex < max_frames) {
            std::size_t bytesRead = std::fread(ioBuffer.data(), 1, ioBuffer.size(), pipe.get());
            if (bytesRead == 0) {
                break;
            }

            streamBuffer.insert(streamBuffer.end(), ioBuffer.begin(), ioBuffer.begin() + bytesRead);

            std::size_t searchPos = 0;
            while (true) {
                std::size_t start = findStartCode(streamBuffer, searchPos);
                if (start == std::string::npos) {
                    break;
                }
                std::size_t next = findStartCode(streamBuffer, start + 4);
                if (next == std::string::npos) {
                    searchPos = start;
                    break;
                }

                const std::uint8_t* nal = streamBuffer.data() + start;
                std::size_t nalSize = next - start;
                feedPacket(nal, nalSize, false);

                streamBuffer.erase(streamBuffer.begin(), streamBuffer.begin() + next);
                searchPos = 0;

                if (frameIndex >= max_frames) {
                    break;
                }
            }

            if (enforceTimeout) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - readStart);
                if (elapsed > timeout + std::chrono::milliseconds(200)) {
                    break;
                }
            }

            if (frameIndex >= max_frames) {
                break;
            }
        }

        if (!streamBuffer.empty() && frameIndex < max_frames) {
            feedPacket(streamBuffer.data(), streamBuffer.size(), false);
            streamBuffer.clear();
        }

        feedPacket(nullptr, 0, true);

        if (frames.empty()) {
            throw std::runtime_error("RTSP capture produced no frames");
        }
        return frames;
    } catch (const std::exception& ex) {
        std::cerr << "MPP capture failed, falling back to FFmpeg: " << ex.what() << std::endl;
    }
#else
    (void)t0;
#endif

    // fallback to FFmpeg software decoding if MPP is unavailable
    std::vector<CapturedFrame> frames;
    frames.reserve(max_frames);

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
    frameBuffer.reserve(1024 * 1024 * 4);

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
            frame.width = 0;
            frame.height = 0;
            frame.stride = 0;
            frame.uv_stride = 0;
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

