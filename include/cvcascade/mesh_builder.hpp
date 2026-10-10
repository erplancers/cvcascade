#pragma once
#include <opencv2/core.hpp>
#include <string>
namespace cvcascade {
struct MeshOptions {
    double width = 100.0;
    double max_height = 20.0;
    bool invert = false;
    int max_dimension = 512;
};
struct MeshStats {
    int rows = 0;
    int cols = 0;
    std::size_t triangles = 0;
    double min_height = 0.0;
    double max_height = 0.0;
};
cv::Mat make_height_map(const cv::Mat& grayscale, const MeshOptions& options);
MeshStats write_stl(const cv::Mat& height_map, const MeshOptions& options, const std::string& output_path);
}
