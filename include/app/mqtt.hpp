#pragma once

#include "app/config.hpp"
#include "app/json.hpp"

#include <string>

namespace app {

class MqttClient {
public:
    bool connect(const MqttConfig &config);
    void publish(const std::string &topic, const Json &payload);
};

} // namespace app

