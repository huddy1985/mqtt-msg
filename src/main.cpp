#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <ifaddrs.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <net/if.h>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <filesystem>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <unistd.h> 

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

std::string detectLocalMac() {
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
    for (const auto& scenario : config.scenarios) {
        simplejson::JsonValue entry = simplejson::makeObject();
        auto& map = entry.asObject();
        map["id"] = scenario.id;
        map["active"] = scenario.active;
        if (!scenario.config_path.empty()) {
            map["config"] = scenario.config_path;
        }
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

struct MonitoringSession {
    std::vector<app::Command> commands;
    std::string request_id;
    std::string response_topic;
};

bool isAnomalousDetection(const app::DetectionResult& detection, double threshold) {
    if (detection.filtered) {
        return false;
    }
    if (detection.confidence < threshold) {
        return false;
    }
    if (detection.label.empty()) {
        return false;
    }

    std::string lowered;
    lowered.reserve(detection.label.size());
    for (char ch : detection.label) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (lowered == "normal" || lowered == "background" || lowered == "ok") {
        return false;
    }
    return true;
}

simplejson::JsonValue detectionToJson(const app::DetectionResult& detection) {
    simplejson::JsonValue value = simplejson::makeObject();
    auto& obj = value.asObject();
    obj["label"] = detection.label;
    simplejson::JsonValue region = simplejson::makeArray();
    auto& arr = region.asArray();
    arr.push_back(detection.region.x1);
    arr.push_back(detection.region.y1);
    arr.push_back(detection.region.x2);
    arr.push_back(detection.region.y2);
    obj["region"] = region;
    obj["confidence"] = detection.confidence;
    obj["filtered"] = detection.filtered;
    return value;
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
        std::filesystem::path root = std::filesystem::current_path();
        app::AppConfig config = app::loadConfig(configPath);
        app::ConfigStore store(root.string());

        app::ProcessingPipeline pipeline(config, &store);
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

            simplejson::JsonValue resultsValue = simplejson::makeArray();
            auto& resultsArray = resultsValue.asArray();

            for (const auto& command : commands) {
                simplejson::JsonValue commandResult = simplejson::makeObject();
                auto& commandObj = commandResult.asObject();

                simplejson::JsonValue scenarioIds = simplejson::makeArray();
                auto& scenarioArray = scenarioIds.asArray();
                for (const auto& scenarioId : command.scenario_id) {
                    scenarioArray.push_back(scenarioId);
                }

                commandObj["scenario_ids"] = scenarioIds;
                commandObj["threshold"] = command.threshold;
                commandObj["fps"] = command.fps;

                if (!command.activation_code.empty()) {
                    commandObj["activation_code"] = command.activation_code;
                }

                if (!command.detection_regions.empty()) {
                    simplejson::JsonValue regionsValue = simplejson::makeArray();
                    auto& regionsArray = regionsValue.asArray();
                    for (const auto& region : command.detection_regions) {
                        simplejson::JsonValue regionValue = simplejson::makeArray();
                        auto& regionArray = regionValue.asArray();
                        regionArray.push_back(region.x1);
                        regionArray.push_back(region.y1);
                        regionArray.push_back(region.x2);
                        regionArray.push_back(region.y2);
                        regionsArray.push_back(regionValue);
                    }
                    commandObj["detection_regions"] = regionsValue;
                }

                if (!command.filter_regions.empty()) {
                    simplejson::JsonValue regionsValue = simplejson::makeArray();
                    auto& regionsArray = regionsValue.asArray();
                    for (const auto& region : command.filter_regions) {
                        simplejson::JsonValue regionValue = simplejson::makeArray();
                        auto& regionArray = regionValue.asArray();
                        regionArray.push_back(region.x1);
                        regionArray.push_back(region.y1);
                        regionArray.push_back(region.x2);
                        regionArray.push_back(region.y2);
                        regionsArray.push_back(regionValue);
                    }
                    commandObj["filter_regions"] = regionsValue;
                }

                commandObj["extra"] = command.extra;

                simplejson::JsonValue scenarioResults = simplejson::makeArray();
                auto& scenarioResultsArray = scenarioResults.asArray();
                auto analyses = pipeline.process(command);

                for (const auto& analysis : analyses) {
                    scenarioResultsArray.push_back(app::toJson(analysis));
                }
                commandObj["results"] = scenarioResults;

                resultsArray.push_back(commandResult);
            }
            obj["results"] = resultsValue;
            std::cout << output.dump(prettyPrint ? 2 : -1) << std::endl;
            return 0;
        }

        std::string localIp = detectLocalIp();
        auto statusBuilder = [effectiveConfig, localIp]() mutable {
            return buildServiceSnapshot(effectiveConfig, localIp);
        };

        std::string localMac = detectLocalMac();

        std::mutex session_mutex;
        std::condition_variable session_cv;
        std::shared_ptr<MonitoringSession> active_session;
        std::atomic<std::uint64_t> session_version{0};
        std::atomic<bool> monitor_stop{false};

        auto processor = [&pipeline,
                          &effectiveConfig,
                          &session_mutex,
                          &session_cv,
                          &active_session,
                          &session_version](const simplejson::JsonValue& payload, std::string& responseTopic) {
            
            std::cout << "=== processor ===" << std::endl;
            std::cout << payload.dump(4) << std::endl;

            const simplejson::JsonValue* commandSource = &payload;
            std::string requestId;
            simplejson::JsonValue commandMetadata;
            bool hasMetadata = false;

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

                auto requestIt = obj.find("request_id");
                if (requestIt != obj.end()) {
                    try {
                        requestId = requestIt->second.asString();
                    } catch (const std::exception&) {
                        requestId.clear();
                    }
                }
                auto extraIt = obj.find("extra");
                if (extraIt != obj.end()) {
                    commandMetadata = extraIt->second;
                    hasMetadata = true;
                }
            }

            std::vector<app::Command> commands = app::parseCommandList(*commandSource);
            for (const auto& command: commands) {
                if (command.action == "enable") {
                    pipeline.add_missing(command.scenario_id);
                } else if (command.action == "disable") {
                    pipeline.remove_inactive(command.scenario_id);
                }
            }

            if (commands.empty()) {
                {
                    std::lock_guard<std::mutex> lock(session_mutex);
                    active_session.reset();
                    session_version.fetch_add(1, std::memory_order_relaxed);
                }
                session_cv.notify_all();

                simplejson::JsonValue response = simplejson::makeObject();
                auto& root = response.asObject();
                root["type"] = "analysis_result";
                root["service_name"] = effectiveConfig.service.name;
                root["timestamp"] = currentIsoTimestamp();
                root["command_count"] = 0;
                root["mode"] = "continuous";
                root["status"] = "monitoring_stopped";
                root["results"] = simplejson::makeArray();
                if (!requestId.empty()) {
                    root["request_id"] = requestId;
                }
                if (hasMetadata) {
                    root["command_metadata"] = commandMetadata;
                }
                return response;
            }

            auto session = std::make_shared<MonitoringSession>();
            session->commands = commands;
            session->request_id = requestId;
            session->response_topic = responseTopic;

            {
                std::lock_guard<std::mutex> lock(session_mutex);
                active_session = session;
                session_version.fetch_add(1, std::memory_order_relaxed);
            }

            session_cv.notify_all();

            simplejson::JsonValue response = simplejson::makeObject();
            auto& root = response.asObject();
            root["type"] = "analysis_result";
            root["service_name"] = effectiveConfig.service.name;
            root["timestamp"] = currentIsoTimestamp();
            root["command_count"] = static_cast<int>(commands.size());
            root["mode"] = "continuous";
            root["status"] = "monitoring_started";
            root["results"] = simplejson::makeArray();

            if (!requestId.empty()) {
                root["request_id"] = requestId;
            }
            if (hasMetadata) {
                root["command_metadata"] = commandMetadata;
            }

            simplejson::JsonValue commandArray = simplejson::makeArray();
            auto& array = commandArray.asArray();
            
            for (const auto& command : commands) {
                simplejson::JsonValue commandJson = simplejson::makeObject();
                auto& commandObj = commandJson.asObject();

                simplejson::JsonValue scenarioIds = simplejson::makeArray();
                auto& scenarioArray = scenarioIds.asArray();
                scenarioArray.push_back(command.scenario_id);

                commandObj["scenario_ids"] = scenarioIds;
                commandObj["threshold"] = command.threshold;
                commandObj["fps"] = command.fps;

                if (!command.activation_code.empty()) {
                    commandObj["activation_code"] = command.activation_code;
                }

                if (!command.detection_regions.empty()) {
                    simplejson::JsonValue regionsValue = simplejson::makeArray();
                    auto& regionsArray = regionsValue.asArray();
                    for (const auto& region : command.detection_regions) {
                        simplejson::JsonValue regionValue = simplejson::makeArray();
                        auto& regionArray = regionValue.asArray();
                        regionArray.push_back(region.x1);
                        regionArray.push_back(region.y1);
                        regionArray.push_back(region.x2);
                        regionArray.push_back(region.y2);
                        regionsArray.push_back(regionValue);
                    }
                    commandObj["detection_regions"] = regionsValue;
                }

                if (!command.filter_regions.empty()) {
                    simplejson::JsonValue regionsValue = simplejson::makeArray();
                    auto& regionsArray = regionsValue.asArray();
                    for (const auto& region : command.filter_regions) {
                        simplejson::JsonValue regionValue = simplejson::makeArray();
                        auto& regionArray = regionValue.asArray();
                        regionArray.push_back(region.x1);
                        regionArray.push_back(region.y1);
                        regionArray.push_back(region.x2);
                        regionArray.push_back(region.y2);
                        regionsArray.push_back(regionValue);
                    }
                    commandObj["filter_regions"] = regionsValue;
                }

                commandObj["extra"] = command.extra;
                array.push_back(commandJson);
            }

            root["commands"] = commandArray;
            return response;
        };

        app::MqttService service(effectiveConfig, processor, statusBuilder);

        std::thread monitorThread([&]() {
            try {
                while (!monitor_stop.load()) {
                    std::shared_ptr<MonitoringSession> session;
                    std::uint64_t version = 0;
                    {
                        std::unique_lock<std::mutex> lock(session_mutex);
                        session_cv.wait(lock, [&]() {
                            return monitor_stop.load() || active_session != nullptr;
                        });
                        if (monitor_stop.load()) {
                            break;
                        }
                        session = active_session;
                        version = session_version.load();
                    }

                    if (!session) {
                        continue;
                    }

                    while (!monitor_stop.load() && session_version.load() == version) {
                        for (const auto& command : session->commands) {
                            if (monitor_stop.load() || session_version.load() != version) {
                                break;
                            }
                            
                            if (command.scenario_id.empty()) {
                                continue;
                            }

                            continue;

                            std::vector<app::AnalysisResult> analyses;

                            try {
                                analyses = pipeline.process(command);
                            } catch (const std::exception& ex) {
                                simplejson::JsonValue error = simplejson::makeObject();
                                auto& obj = error.asObject();
                                obj["type"] = "analysis_error";
                                obj["service_name"] = effectiveConfig.service.name;
                                obj["client_id"] = effectiveConfig.mqtt.client_id;
                                obj["timestamp"] = currentIsoTimestamp();
                                obj["error"] = ex.what();
                                if (!session->request_id.empty()) {
                                    obj["request_id"] = session->request_id;
                                }
                                try {
                                    service.publish(error, session->response_topic);
                                } catch (const std::exception& pubEx) {
                                    std::cerr << "MQTT publish error: " << pubEx.what() << std::endl;
                                }
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                continue;
                            }

                            for (const auto& analysis : analyses) {
                                if (monitor_stop.load() || session_version.load() != version) {
                                    break;
                                }
                                for (const auto& frame : analysis.frames) {
                                    if (monitor_stop.load() || session_version.load() != version) {
                                        break;
                                    }

                                    simplejson::JsonValue detectionArray = simplejson::makeArray();
                                    auto& detectionVec = detectionArray.asArray();
                                    for (const auto& detection : frame.detections) {
                                        if (isAnomalousDetection(detection, command.threshold)) {
                                            detectionVec.push_back(detectionToJson(detection));
                                        }
                                    }

                                    if (detectionVec.empty()) {
                                        continue;
                                    }

                                    simplejson::JsonValue frameJson = simplejson::makeObject();
                                    auto& frameObj = frameJson.asObject();
                                    frameObj["timestamp"] = frame.timestamp;
                                    if (!frame.image_path.empty()) {
                                        frameObj["image_path"] = frame.image_path;
                                    }
                                    frameObj["detections"] = detectionArray;

                                    simplejson::JsonValue event = simplejson::makeObject();
                                    auto& eventObj = event.asObject();
                                    eventObj["type"] = "analysis_anomaly";
                                    eventObj["timestamp"] = currentIsoTimestamp();
                                    eventObj["service_name"] = effectiveConfig.service.name;
                                    eventObj["client_id"] = effectiveConfig.mqtt.client_id;
                                    eventObj["scenario_id"] = analysis.scenario_id;
                                    simplejson::JsonValue modelJson = simplejson::makeObject();
                                    auto& modelObj = modelJson.asObject();
                                    modelObj["id"] = analysis.model.id;
                                    modelObj["type"] = analysis.model.type;
                                    modelObj["path"] = analysis.model.path;
                                    eventObj["model"] = modelJson;
                                    eventObj["frame"] = frameJson;
                                    eventObj["threshold"] = command.threshold;
                                    if (!session->request_id.empty()) {
                                        eventObj["request_id"] = session->request_id;
                                    }
                                    if (command.fps > 0.0) {
                                        eventObj["fps"] = command.fps;
                                    }

                                    try {
                                        service.publish(event, session->response_topic);
                                    } catch (const std::exception& ex) {
                                        std::cerr << "MQTT publish error: " << ex.what() << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (const std::exception& ex) {
                std::cerr << "Monitoring loop error: " << ex.what() << std::endl;
            }
        });

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

        monitor_stop.store(true);
        session_cv.notify_all();
        if (monitorThread.joinable()) {
            monitorThread.join();
        }

        worker.join();
        runFuture.get();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
