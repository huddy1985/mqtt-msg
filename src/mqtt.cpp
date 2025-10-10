#include "app/mqtt.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef APP_HAS_MOSQUITTO
#include <mosquitto.h>
#endif

namespace app {

namespace {
#ifdef APP_HAS_MOSQUITTO
std::string toCompactJson(const simplejson::JsonValue& value) {
    return value.dump(-1);
}
#endif
}  // namespace

struct MqttService::Impl {
#ifdef APP_HAS_MOSQUITTO
    Impl(AppConfig cfg, Processor proc, StatusBuilder status)
        : config(std::move(cfg)), processor(std::move(proc)), status_builder(std::move(status)) {
        if (!processor) {
            throw std::invalid_argument("MQTT processor callback must not be empty");
        }
        mosquitto_lib_init();
        const char* client_id = nullptr;
        if (!config.mqtt.client_id.empty()) {
            client_id = config.mqtt.client_id.c_str();
        }
        client.reset(mosquitto_new(client_id, true, this));
        if (!client) {
            mosquitto_lib_cleanup();
            throw std::runtime_error("Failed to create MQTT client");
        }

        mosquitto_connect_callback_set(client.get(), &Impl::onConnect);
        mosquitto_disconnect_callback_set(client.get(), &Impl::onDisconnect);
        mosquitto_message_callback_set(client.get(), &Impl::onMessage);
        mosquitto_reconnect_delay_set(client.get(), 1, 8, true);

        if (!config.mqtt.username.empty()) {
            const char* username = config.mqtt.username.c_str();
            const char* password = config.mqtt.password.empty() ? nullptr : config.mqtt.password.c_str();
            int rc = mosquitto_username_pw_set(client.get(), username, password);
            if (rc != MOSQ_ERR_SUCCESS) {
                mosquitto_destroy(client.release());
                mosquitto_lib_cleanup();
                throw std::runtime_error(std::string("Failed to set MQTT credentials: ") + mosquitto_strerror(rc));
            }
        } else if (!config.mqtt.password.empty()) {
            mosquitto_destroy(client.release());
            mosquitto_lib_cleanup();
            throw std::runtime_error("MQTT password provided without username");
        }

        publish_topic = config.mqtt.publish_topic;
        if (publish_topic.empty()) {
            publish_topic = config.mqtt.subscribe_topic + std::string("/response");
        }
    }

    ~Impl() {
        if (client) {
            mosquitto_destroy(client.release());
        }
        mosquitto_lib_cleanup();
    }

    void run() {
        stop_requested.store(false);
        const std::string& server = config.mqtt.server;
        if (server.empty()) {
            throw std::runtime_error("MQTT server address is empty");
        }
        int port = config.mqtt.port > 0 ? config.mqtt.port : 1883;
        int keep_alive = 60;
        int rc = mosquitto_connect(client.get(), server.c_str(), port, keep_alive);
        if (rc != MOSQ_ERR_SUCCESS) {
            throw std::runtime_error(std::string("Failed to connect to MQTT broker: ") + mosquitto_strerror(rc));
        }

        while (!stop_requested.load()) {
            rc = mosquitto_loop(client.get(), 1000, 1);
            if (stop_requested.load()) {
                break;
            }
            if (rc != MOSQ_ERR_SUCCESS) {
                std::cerr << "MQTT loop warning: " << mosquitto_strerror(rc) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                mosquitto_reconnect(client.get());
            }
        }
    }

    void stop() {
        bool expected = false;
        if (stop_requested.compare_exchange_strong(expected, true)) {
            mosquitto_disconnect(client.get());
        } else {
            stop_requested.store(true);
            mosquitto_disconnect(client.get());
        }
    }

    void publishJson(simplejson::JsonValue value, const std::string& topic_override = {}) {
        std::lock_guard<std::mutex> lock(publish_mutex);
        std::string topic = topic_override.empty() ? publish_topic : topic_override;
        if (topic.empty()) {
            topic = config.mqtt.publish_topic;
        }
        if (topic.empty()) {
            topic = "mqtt_msg/response";
        }
        std::string payload = toCompactJson(value);
        int rc = mosquitto_publish(client.get(), nullptr, topic.c_str(), static_cast<int>(payload.size()), payload.data(), 1, false);
        if (rc != MOSQ_ERR_SUCCESS) {
            std::cerr << "Failed to publish MQTT message: " << mosquitto_strerror(rc) << std::endl;
        }
    }

    void publishStatus(const std::string& state) {
        simplejson::JsonValue payload;
        if (status_builder) {
            payload = status_builder();
            if (!payload.isObject()) {
                payload = simplejson::makeObject();
            }
        } else {
            payload = simplejson::makeObject();
        }
        auto& obj = payload.asObject();
        obj["type"] = "service_registration";
        obj["state"] = state;
        obj["service_name"] = config.service.name;
        obj["client_id"] = config.mqtt.client_id;
        publishJson(payload);
    }

    void publishError(const std::string& error, const std::string& request_id) {
        simplejson::JsonValue payload = simplejson::makeObject();
        auto& obj = payload.asObject();
        obj["type"] = "analysis_error";
        obj["service_name"] = config.service.name;
        obj["client_id"] = config.mqtt.client_id;
        obj["error"] = error;
        if (!request_id.empty()) {
            obj["request_id"] = request_id;
        }
        publishJson(payload);
    }

    void handleMessage(const mosquitto_message* message) {
        if (!message || !message->payload || message->payloadlen <= 0) {
            return;
        }
        std::string payload(static_cast<const char*>(message->payload), static_cast<std::size_t>(message->payloadlen));
        try {
            simplejson::JsonValue json = simplejson::parse(payload);
            std::string response_topic;
            std::string request_id;
            if (json.isObject() && json.asObject().count("request_id")) {
                try {
                    request_id = json.at("request_id").asString();
                } catch (const std::exception&) {
                    request_id.clear();
                }
            }
            simplejson::JsonValue response = processor(json, response_topic);
            if (!response.isObject()) {
                // Wrap non-object responses so metadata stays consistent.
                simplejson::JsonValue wrapper = simplejson::makeObject();
                auto& obj = wrapper.asObject();
                obj["type"] = "analysis_result";
                obj["service_name"] = config.service.name;
                obj["client_id"] = config.mqtt.client_id;
                obj["payload"] = response;
                response = wrapper;
            } else {
                auto& obj = response.asObject();
                if (!obj.count("type")) {
                    obj["type"] = "analysis_result";
                }
                obj["service_name"] = config.service.name;
                obj["client_id"] = config.mqtt.client_id;
                if (!request_id.empty()) {
                    obj["request_id"] = request_id;
                }
            }
            publishJson(response, response_topic);
        } catch (const std::exception& ex) {
            publishError(ex.what());
        }
    }

    static void onConnect(struct mosquitto* mosq, void* userdata, int rc) {
        auto* self = static_cast<Impl*>(userdata);
        if (!self) {
            return;
        }
        if (rc == 0) {
            self->publishStatus("online");
            if (!self->config.mqtt.subscribe_topic.empty()) {
                mosquitto_subscribe(mosq, nullptr, self->config.mqtt.subscribe_topic.c_str(), 1);
            }
        } else {
            std::cerr << "MQTT connect failed: " << mosquitto_strerror(rc) << std::endl;
        }
    }

    static void onDisconnect(struct mosquitto* mosq, void* userdata, int rc) {
        (void)mosq;
        auto* self = static_cast<Impl*>(userdata);
        if (!self) {
            return;
        }
        if (rc == 0) {
            self->publishStatus("offline");
        } else {
            std::cerr << "MQTT unexpected disconnect: " << mosquitto_strerror(rc) << std::endl;
        }
    }

    static void onMessage(struct mosquitto* mosq, void* userdata, const mosquitto_message* message) {
        (void)mosq;
        auto* self = static_cast<Impl*>(userdata);
        if (!self) {
            return;
        }
        self->handleMessage(message);
    }

    AppConfig config;
    Processor processor;
    StatusBuilder status_builder;
    std::unique_ptr<mosquitto, decltype(&mosquitto_destroy)> client{nullptr, mosquitto_destroy};
    std::atomic<bool> stop_requested{false};
    std::mutex publish_mutex;
    std::string publish_topic;
#else
    Impl(AppConfig, Processor, StatusBuilder) {
        throw std::runtime_error("MQTT support not available: libmosquitto not linked");
    }
    void run() {}
    void stop() {}
#endif
};

MqttService::MqttService(AppConfig config, Processor processor, StatusBuilder status_builder)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(processor), std::move(status_builder))) {}

MqttService::~MqttService() = default;

void MqttService::run() {
#ifdef APP_HAS_MOSQUITTO
    impl_->run();
#else
    throw std::runtime_error("MQTT support not available: libmosquitto not linked");
#endif
}

void MqttService::stop() {
#ifdef APP_HAS_MOSQUITTO
    impl_->stop();
#else
    (void)impl_;
#endif
}

void MqttService::publish(simplejson::JsonValue value, const std::string& topic) {
#ifdef APP_HAS_MOSQUITTO
    impl_->publishJson(std::move(value), topic);
#else
    (void)value;
    (void)topic;
    throw std::runtime_error("MQTT support not available: libmosquitto not linked");
#endif
}

}  // namespace app
