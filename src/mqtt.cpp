#include "app/mqtt.hpp"

#include <iostream>

namespace app {

bool MqttClient::connect(const MqttConfig &config) {
    std::cout << "Connecting to MQTT broker at " << config.host << ":" << config.port;
    if (!config.username.empty()) {
        std::cout << " using username '" << config.username << "'";
    }
    std::cout << "\n";
    return true;
}

void MqttClient::publish(const std::string &topic, const Json &payload) {
    std::cout << "Publishing to topic '" << topic << "':\n";
    std::cout << payload.dump(2) << "\n";
}

} // namespace app

