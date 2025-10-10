#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "app/command.hpp"
#include "app/config.hpp"
#include "app/pipeline.hpp"

namespace {

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
              << " STDIN or a file, and emits simulated analysis results as JSON." << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string configPath = "local.config.json";
    std::string commandPath;
    bool prettyPrint = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--command" && i + 1 < argc) {
            commandPath = argv[++i];
        } else if (arg == "--compact") {
            prettyPrint = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        app::AppConfig config = app::loadConfig(configPath);
        app::ProcessingPipeline pipeline(config);

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
            simplejson::JsonValue info = simplejson::makeObject();
            auto& obj = info.asObject();
            obj["service_name"] = config.service.name;
            obj["mqtt_server"] = config.mqtt.server;
            obj["mqtt_topic"] = config.mqtt.subscribe_topic;
            obj["scenarios"] = simplejson::makeArray();
            auto& scenarios = obj["scenarios"].asArray();
            for (const auto& entry : config.scenario_list) {
                simplejson::JsonValue scenario = simplejson::makeObject();
                auto& map = scenario.asObject();
                map["id"] = entry.id;
                map["model"] = simplejson::makeObject();
                auto& modelMap = map["model"].asObject();
                modelMap["id"] = entry.model.id;
                modelMap["type"] = entry.model.type;
                modelMap["path"] = entry.model.path;
                scenarios.push_back(scenario);
            }
            std::cout << info.dump(prettyPrint ? 2 : -1) << std::endl;
            return 0;
        }

        simplejson::JsonValue commandsJson = simplejson::parse(commandData);
        std::vector<app::Command> commands = app::parseCommandList(commandsJson);

        simplejson::JsonValue output = simplejson::makeArray();
        auto& outputArray = output.asArray();
        for (const auto& command : commands) {
            app::AnalysisResult result = pipeline.process(command);
            outputArray.push_back(app::toJson(result));
        }

        std::cout << output.dump(prettyPrint ? 2 : -1) << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

