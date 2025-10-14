#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "app/json.hpp"


namespace app {

struct Region {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    bool operator==(const Region& other) const {
        return x1 == other.x1 && y1 == other.y1 && x2 == other.x2 && y2 == other.y2;
    }
};

struct ModelInfo {
    std::string id;
    std::string type;
    std::string path;
};

Region parseRegion(const simplejson::JsonValue& value);
std::vector<Region> parseRegions(const simplejson::JsonValue& value);

std::vector<std::string> parseLabels(const simplejson::JsonValue& value);

std::string detectLocalIp();
std::string detectLocalMac();

using namespace cv; 

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;

struct PreprocessInfo {
    std::vector<float> input_tensor;
    float scale;
    int pad_x;
    int pad_y;
    cv::Mat resized_image;
};

struct InferResult {
    int num_boxes;
    int classid;
};

PreprocessInfo preprocess_letterbox(const cv::Mat& img, int input_w, int input_h);

} // namespace app