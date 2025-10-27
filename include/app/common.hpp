#pragma once
#include <string>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "app/json.hpp"


namespace app {

struct Region {
    int x = 0;     // left x
    int y = 0;     // left y
    int width = 0;     // width
    int height = 0;     // height

    bool operator==(const Region& other) const {
        return x == other.x && y == other.y && width == other.width && height == other.height;
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

struct CapturedFrame;
cv::Mat decodeFrameToMat(const CapturedFrame& frame);
cv::Mat extractROI(const cv::Mat& image, int x, int y, int width, int height);

std::vector<int> NMS(const std::vector<cv::Rect2f>& boxes,
                     const std::vector<float>& scores,
                     float iouThreshold = 0.45f);
float IoU(const cv::Rect2f& a, const cv::Rect2f& b);


} // namespace app