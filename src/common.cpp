#include "app/common.hpp"

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
    region.x1 = static_cast<int>(arr[0].asNumber());
    region.y1 = static_cast<int>(arr[1].asNumber());
    region.x2 = static_cast<int>(arr[2].asNumber());
    region.y2 = static_cast<int>(arr[3].asNumber());
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

};