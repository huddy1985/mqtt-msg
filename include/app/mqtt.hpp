#pragma once

#include <functional>
#include <memory>
#include <string>

#include "app/config.hpp"
#include "app/json.hpp"

namespace app {

class MqttService {
public:
    using Processor = std::function<simplejson::JsonValue(const simplejson::JsonValue&, std::string&)>;
    using StatusBuilder = std::function<simplejson::JsonValue()>;

    MqttService(AppConfig config, Processor processor, StatusBuilder status_builder);
    ~MqttService();

    // Starts the MQTT event loop. The call blocks until stop() is invoked or a fatal error occurs.
    void run();

    // Requests the service to stop. Safe to call from any thread.
    void stop();

    // Publishes a JSON payload to the broker using either the default publish
    // topic or the supplied override. Safe to call from any thread while the
    // service is running.
    void publish(simplejson::JsonValue value, const std::string& topic = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app
