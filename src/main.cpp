#include <chrono>
#include <csignal>
#include <fstream>
#include <future>
#include <iomanip>
#include <ifaddrs.h>
#include <iostream>
#include <iterator>
#include <net/if.h>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>

#include "app/command.hpp"
#include "app/config.hpp"
#include "app/mqtt.hpp"
#include "app/pipeline.hpp"

namespace {

volatile std::sig_atomic_t gSignalStatus = 0;

std::string readStream(std::istream& stream) {
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [--config <path>] [--command <path>] [--compact]\n"
              << "Reads the local configuration, ingests MQTT-style analysis commands from"
              << " STDIN or a file, and emits analysis results as JSON." << std::endl;
    std::cout << "       " << executable << " [--config <path>] [--service | --oneshot]"
              << "\n"
              << "Runs either as a long-lived MQTT backend service or a one-shot command"
              << " processor." << std::endl;
}

void signalHandler(int signal) {
    gSignalStatus = signal;
}

std::string detectLocalIp() {
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

simplejson::JsonValue buildServiceSnapshot(const app::AppConfig& config, const std::string& localIp) {
    simplejson::JsonValue root = simplejson::makeObject();
    auto& obj = root.asObject();
    obj["service_name"] = config.service.name;
    if (!config.service.description.empty()) {
        obj["description"] = config.service.description;
    }
    obj["client_id"] = config.mqtt.client_id;
    obj["mqtt_server"] = config.mqtt.server;
    obj["mqtt_port"] = config.mqtt.port;
    obj["subscribe_topic"] = config.mqtt.subscribe_topic;
    obj["publish_topic"] = config.mqtt.publish_topic;
    obj["local_ip"] = localIp;
    simplejson::JsonValue rtsp = simplejson::makeObject();
    auto& rtspObj = rtsp.asObject();
    rtspObj["host"] = config.rtsp.host;
    rtspObj["port"] = config.rtsp.port;
    rtspObj["path"] = config.rtsp.path;
    obj["rtsp"] = rtsp;

    simplejson::JsonValue scenarios = simplejson::makeArray();
    auto& array = scenarios.asArray();
    for (const auto& scenario : config.scenario_list) {
        simplejson::JsonValue entry = simplejson::makeObject();
        auto& map = entry.asObject();
        map["id"] = scenario.id;
        simplejson::JsonValue model = simplejson::makeObject();
        auto& modelObj = model.asObject();
        modelObj["id"] = scenario.model.id;
        modelObj["type"] = scenario.model.type;
        modelObj["path"] = scenario.model.path;
        map["model"] = model;
        array.push_back(entry);
    }
    obj["scenarios"] = scenarios;
    return root;
}

std::string currentIsoTimestamp() {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    auto time = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    auto fractional = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setw(3) << std::setfill('0') << fractional.count() << "Z";
    return oss.str();
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string configPath = "local.config.json";
    std::string commandPath;
    bool prettyPrint = true;
    bool forceService = false;
    bool forceOneshot = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--command" && i + 1 < argc) {
            commandPath = argv[++i];
        } else if (arg == "--compact") {
            prettyPrint = false;
        } else if (arg == "--service") {
            forceService = true;
        } else if (arg == "--oneshot") {
            forceOneshot = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!commandPath.empty()) {
        forceOneshot = true;
    }

    try {
        app::AppConfig config = app::loadConfig(configPath);
        app::ProcessingPipeline pipeline(config);
        const app::AppConfig& effectiveConfig = pipeline.config();

        bool runService = forceService || (!forceOneshot && commandPath.empty());

        if (!runService) {
            std::string commandData;
            if (!commandPath.empty()) {
                std::ifstream commandFile(commandPath);
                if (!commandFile) {
                    std::cerr << "Failed to open command file: " << commandPath << std::endl;
                    return 1;
                }
                commandData = readStream(commandFile);
            } else {
                commandData = readStream(std::cin);
            }

            if (trim(commandData).empty()) {
                simplejson::JsonValue info = buildServiceSnapshot(effectiveConfig, detectLocalIp());
                std::cout << info.dump(prettyPrint ? 2 : -1) << std::endl;
                return 0;
            }

            simplejson::JsonValue commandsJson = simplejson::parse(commandData);
            std::vector<app::Command> commands = app::parseCommandList(commandsJson);

            simplejson::JsonValue output = simplejson::makeObject();
            auto& obj = output.asObject();
            obj["service_name"] = effectiveConfig.service.name;
            obj["timestamp"] = currentIsoTimestamp();
            simplejson::JsonValue frames = simplejson::makeArray();
            auto& array = frames.asArray();
            for (const auto& command : commands) {
                app::AnalysisResult result = pipeline.process(command);
                array.push_back(app::toJson(result));
            }
            obj["results"] = frames;
            std::cout << output.dump(prettyPrint ? 2 : -1) << std::endl;
            return 0;
        }

        std::string localIp = detectLocalIp();
        auto statusBuilder = [effectiveConfig, localIp]() mutable {
            return buildServiceSnapshot(effectiveConfig, localIp);
        };

        auto processor = [&pipeline, &effectiveConfig](const simplejson::JsonValue& payload, std::string& responseTopic) {
            const simplejson::JsonValue* commandSource = &payload;
            if (payload.isObject()) {
                const auto& obj = payload.asObject();
                auto it = obj.find("response_topic");
                if (it != obj.end()) {
                    try {
                        responseTopic = it->second.asString();
                    } catch (const std::exception&) {
                        responseTopic.clear();
                    }
                }
                auto cmdIt = obj.find("commands");
                if (cmdIt != obj.end()) {
                    commandSource = &cmdIt->second;
                }
            }

            std::vector<app::Command> commands = app::parseCommandList(*commandSource);
            simplejson::JsonValue response = simplejson::makeObject();
            auto& root = response.asObject();
            root["type"] = "analysis_result";
            root["service_name"] = effectiveConfig.service.name;
            root["timestamp"] = currentIsoTimestamp();
            root["command_count"] = static_cast<int>(commands.size());

            simplejson::JsonValue resultArray = simplejson::makeArray();
            auto& array = resultArray.asArray();
            for (const auto& command : commands) {
                app::AnalysisResult analysis = pipeline.process(command);
                array.push_back(app::toJson(analysis));
            }
            root["results"] = resultArray;

            if (payload.isObject()) {
                const auto& obj = payload.asObject();
                auto requestIt = obj.find("request_id");
                if (requestIt != obj.end()) {
                    try {
                        root["request_id"] = requestIt->second.asString();
                    } catch (const std::exception&) {
                        // ignore conversion errors
                    }
                }
                auto extraIt = obj.find("extra");
                if (extraIt != obj.end()) {
                    root["command_metadata"] = extraIt->second;
                }
            }

            return response;
        };

        app::MqttService service(effectiveConfig, processor, statusBuilder);

        std::promise<void> runPromise;
        std::future<void> runFuture = runPromise.get_future();
        std::thread worker([&service, &runPromise]() {
            try {
                service.run();
                runPromise.set_value();
            } catch (...) {
                try {
                    runPromise.set_exception(std::current_exception());
                } catch (...) {
                    // set_exception may throw if promise already satisfied
                }
            }
        });

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);

        while (runFuture.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout) {
            if (gSignalStatus != 0) {
                service.stop();
            }
        }

        if (gSignalStatus != 0) {
            service.stop();
        }

        worker.join();
        runFuture.get();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
