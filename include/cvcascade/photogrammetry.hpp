#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>

namespace cvcascade {

struct CameraModel {
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Size image_size;
    double reprojection_rms = 0.0;
};

struct PhotogrammetryOptions {
    int max_dimension = 640;
    int num_disparities = 128;
    int block_size = 5;
    int min_inliers = 30;
    double baseline_mm = 0.0;
    double max_depth = 10000.0;
    double max_relative_depth_jump = 0.20;
};

struct ReconstructionStats {
    std::size_t images = 0;
    std::size_t registered_pairs = 0;
    std::size_t feature_matches = 0;
    std::size_t pose_inliers = 0;
    std::size_t vertices = 0;
    std::size_t triangles = 0;
    bool metric_scale = false;
};

std::vector<std::filesystem::path> read_image_list(const std::filesystem::path& list_path);

CameraModel calibrate_camera_from_list(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_output,
    int board_cols,
    int board_rows,
    double square_size_mm);

CameraModel load_camera_model(const std::filesystem::path& calibration_path);

ReconstructionStats reconstruct_multiview(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_path,
    const std::filesystem::path& output_stl,
    const PhotogrammetryOptions& options);

} // namespace cvcascade
