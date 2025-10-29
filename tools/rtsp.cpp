#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace {

void ensure_directory(const std::filesystem::path &directory)
{
    if (std::filesystem::exists(directory)) {
        if (!std::filesystem::is_directory(directory)) {
            throw std::runtime_error("Output path exists and is not a directory: " + directory.string());
        }
        return;
    }

    std::filesystem::create_directories(directory);
}

std::string make_frame_name(const std::string &prefix, std::size_t index)
{
    std::ostringstream oss;
    oss << prefix << std::setw(6) << std::setfill('0') << index << ".ppm";
    return oss.str();
}

void save_frame_as_ppm(const AVFrame *frame, const std::filesystem::path &path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << "P6\n" << frame->width << " " << frame->height << "\n255\n";

    const int stride = frame->linesize[0];
    const int width_in_bytes = frame->width * 3;

    for (int y = 0; y < frame->height; ++y) {
        out.write(reinterpret_cast<const char *>(frame->data[0] + y * stride), width_in_bytes);
    }
}

struct FormatContextDeleter {
    void operator()(AVFormatContext *ctx) const noexcept
    {
        if (ctx != nullptr) {
            avformat_close_input(&ctx);
        }
    }
};

struct CodecContextDeleter {
    void operator()(AVCodecContext *ctx) const noexcept
    {
        if (ctx != nullptr) {
            avcodec_free_context(&ctx);
        }
    }
};

struct FrameDeleter {
    void operator()(AVFrame *frame) const noexcept
    {
        av_frame_free(&frame);
    }
};

struct PacketDeleter {
    void operator()(AVPacket *packet) const noexcept
    {
        av_packet_free(&packet);
    }
};

using FormatContextPtr = std::unique_ptr<AVFormatContext, FormatContextDeleter>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;

std::string av_error_to_string(int errnum)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, buffer, sizeof(buffer));
    return buffer;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rtsp-url> <output-directory> [prefix]" << std::endl;
        return 1;
    }

    const std::string url = argv[1];
    const std::filesystem::path output_dir = argv[2];
    const std::string prefix = (argc >= 4) ? argv[3] : "frame_";

    try {
        ensure_directory(output_dir);
    } catch (const std::exception &ex) {
        std::cerr << "Error creating output directory: " << ex.what() << std::endl;
        return 1;
    }

    av_log_set_level(AV_LOG_INFO);

    if (int err = avformat_network_init(); err < 0) {
        std::cerr << "Failed to initialise network components: " << av_error_to_string(err) << std::endl;
        return 1;
    }

    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "max_delay", "0", 0);
    av_dict_set(&options, "buffer_size", "102400", 0);

    AVFormatContext *raw_format_ctx = nullptr;
    if (int err = avformat_open_input(&raw_format_ctx, url.c_str(), nullptr, &options); err < 0) {
        std::cerr << "Failed to open RTSP stream: " << av_error_to_string(err) << std::endl;
        av_dict_free(&options);
        return 1;
    }
    FormatContextPtr format_ctx(raw_format_ctx);
    av_dict_free(&options);

    if (int err = avformat_find_stream_info(format_ctx.get(), nullptr); err < 0) {
        std::cerr << "Failed to retrieve stream info: " << av_error_to_string(err) << std::endl;
        return 1;
    }

    av_dump_format(format_ctx.get(), 0, url.c_str(), 0);

    AVCodec *codec = nullptr;
    int stream_index = av_find_best_stream(format_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_index < 0) {
        std::cerr << "Could not find a video stream in the input" << std::endl;
        return 1;
    }

    CodecContextPtr codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        return 1;
    }

    if (int err = avcodec_parameters_to_context(codec_ctx.get(), format_ctx->streams[stream_index]->codecpar); err < 0) {
        std::cerr << "Failed to copy codec parameters: " << av_error_to_string(err) << std::endl;
        return 1;
    }

    codec_ctx->thread_count = 0; // auto
    codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

    if (codec->capabilities & AV_CODEC_CAP_TRUNCATED) {
        codec_ctx->flags |= AV_CODEC_FLAG_TRUNCATED;
    }

    if (int err = avcodec_open2(codec_ctx.get(), codec, nullptr); err < 0) {
        std::cerr << "Failed to open codec: " << av_error_to_string(err) << std::endl;
        return 1;
    }

    FramePtr frame(av_frame_alloc());
    FramePtr rgb_frame(av_frame_alloc());
    PacketPtr packet(av_packet_alloc());

    if (!frame || !rgb_frame || !packet) {
        std::cerr << "Failed to allocate frame or packet" << std::endl;
        return 1;
    }

    const AVPixelFormat target_pixel_format = AV_PIX_FMT_RGB24;

    SwsContext *sws_ctx = sws_getContext(
        codec_ctx->width,
        codec_ctx->height,
        codec_ctx->pix_fmt,
        codec_ctx->width,
        codec_ctx->height,
        target_pixel_format,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr);

    if (!sws_ctx) {
        std::cerr << "Failed to create sws context" << std::endl;
        return 1;
    }

    int buffer_size = av_image_get_buffer_size(target_pixel_format, codec_ctx->width, codec_ctx->height, 1);
    std::vector<uint8_t> buffer(buffer_size);

    if (int err = av_image_fill_arrays(
            rgb_frame->data,
            rgb_frame->linesize,
            buffer.data(),
            target_pixel_format,
            codec_ctx->width,
            codec_ctx->height,
            1);
        err < 0) {
        std::cerr << "Failed to allocate RGB frame data: " << av_error_to_string(err) << std::endl;
        sws_freeContext(sws_ctx);
        return 1;
    }

    rgb_frame->width = codec_ctx->width;
    rgb_frame->height = codec_ctx->height;
    rgb_frame->format = target_pixel_format;

    std::size_t frame_count = 0;

    while (av_read_frame(format_ctx.get(), packet.get()) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet.get());
            continue;
        }

        if (int err = avcodec_send_packet(codec_ctx.get(), packet.get()); err < 0) {
            std::cerr << "Error sending packet to decoder: " << av_error_to_string(err) << std::endl;
            av_packet_unref(packet.get());
            break;
        }

        av_packet_unref(packet.get());

        while (true) {
            int ret = avcodec_receive_frame(codec_ctx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                std::cerr << "Error decoding frame: " << av_error_to_string(ret) << std::endl;
                sws_freeContext(sws_ctx);
                return 1;
            }

            sws_scale(
                sws_ctx,
                frame->data,
                frame->linesize,
                0,
                codec_ctx->height,
                rgb_frame->data,
                rgb_frame->linesize);

            try {
                auto filename = make_frame_name(prefix, frame_count++);
                auto filepath = output_dir / filename;
                save_frame_as_ppm(rgb_frame.get(), filepath);
                std::cout << "Saved frame " << frame_count << " to " << filepath << std::endl;
            } catch (const std::exception &ex) {
                std::cerr << "Failed to save frame: " << ex.what() << std::endl;
                sws_freeContext(sws_ctx);
                return 1;
            }
        }
    }

    sws_freeContext(sws_ctx);
    std::cout << "Finished saving " << frame_count << " frames." << std::endl;

    return 0;
}
