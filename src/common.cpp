#include "app/common.hpp"
#include "app/rtsp.hpp"

#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <unistd.h> 

namespace app {

Region parseRegion(const simplejson::JsonValue& value)
{
    if (!value.isArray()) {
        throw std::runtime_error("Region must be an array of four integers");
    }
    const auto& arr = value.asArray();
    if (arr.size() != 4) {
        throw std::runtime_error("Region must contain four numbers");
    }
    Region region;
    region.x = static_cast<int>(arr[0].asNumber());
    region.y = static_cast<int>(arr[1].asNumber());
    region.width = static_cast<int>(arr[2].asNumber());
    region.height = static_cast<int>(arr[3].asNumber());
    return region;
}

std::vector<Region> parseRegions(const simplejson::JsonValue& value)
{
    std::vector<Region> regions;
    if (!value.isArray()) {
        return regions;
    }
    for (const auto& entry : value.asArray()) {
        regions.push_back(parseRegion(entry));
    }
    return regions;
}

std::vector<std::string> parseLabels(const simplejson::JsonValue& value)
{
    std::vector<std::string> labels;
    if (!value.isArray()) {
        return labels;
    }
    for (const auto& entry : value.asArray()) {
        labels.push_back(entry.asString());
    }
    return labels;
}

std::string detectLocalIp() 
{
    std::string fallback = "0.0.0.0";
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0 || !ifaddr) {
        return fallback;
    }
    std::string result = fallback;
    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        if ((ifa->ifa_flags & IFF_UP) == 0 || (ifa->ifa_flags & IFF_LOOPBACK) != 0) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) {
            auto* addr = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
            char buffer[INET_ADDRSTRLEN] = {0};
            if (inet_ntop(AF_INET, &addr->sin_addr, buffer, sizeof(buffer))) {
                result = buffer;
                break;
            }
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

std::string detectLocalMac() 
{
    std::string fallback = "00:00:00:00:00:00";
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0 || !ifaddr) {
        return fallback;
    }

    std::string mac = fallback;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        freeifaddrs(ifaddr);
        return fallback;
    }

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr)
            continue;

        if ((ifa->ifa_flags & IFF_UP) == 0 || (ifa->ifa_flags & IFF_LOOPBACK))
            continue;

        if (ifa->ifa_addr->sa_family == AF_INET) {
            struct ifreq ifr {};
            std::strncpy(ifr.ifr_name, ifa->ifa_name, IFNAMSIZ - 1);
            if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                unsigned char* hw = reinterpret_cast<unsigned char*>(ifr.ifr_hwaddr.sa_data);
                std::ostringstream oss;
                oss << std::hex << std::setfill('0');
                for (int i = 0; i < 6; ++i) {
                    oss << std::setw(2) << static_cast<int>(hw[i]);
                    if (i < 5) oss << ":";
                }
                mac = oss.str();
                break;
            }
        }
    }

    close(sock);
    freeifaddrs(ifaddr);
    return mac;
}

PreprocessInfo preprocess_letterbox(const cv::Mat& img, int input_w, int input_h) 
{
    int img_w = img.cols;
    int img_h = img.rows;

    float scale = std::min((float)input_w / img_w, (float)input_h / img_h);

    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_x = (input_w - new_w) / 2;
    int pad_y = (input_h - new_h) / 2;                                                                                                                  

    cv::Mat letterbox(input_h, input_w, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat float_img;
    letterbox.convertTo(float_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> chw(3);
    cv::split(float_img, chw);

    std::vector<float> input_tensor;
    input_tensor.reserve(input_w * input_h * 3);
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(input_tensor.end(),
                            (float*)chw[c].datastart,
                            (float*)chw[c].dataend);
    }

    return {input_tensor, scale, pad_x, pad_y, letterbox};
}

cv::Mat decodeFrameToMat(const CapturedFrame& frame)
{
    cv::Mat encoded(1, frame.data.size(), CV_8UC1, const_cast<uint8_t*>(frame.data.data()));

    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to decode JPEG frame");
    }
    return image;
}

cv::Mat extractROI(const cv::Mat& image, int x, int y, int width, int height)
{
    if (x < 0 || y < 0 || x + width > image.cols || y + height > image.rows) {
        throw std::runtime_error("ROI out of bounds");
    }

    cv::Rect roi(x, y, width, height);

    return image(roi).clone();
}

};