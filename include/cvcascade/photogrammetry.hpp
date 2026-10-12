#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <vector>

namespace cvcascade {

struct CameraModel {
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Size image_size;
    double reprojection_rms = 0.0;
    std::size_t calibration_views_used = 0;
    std::size_t calibration_views_rejected = 0;
};

struct PhotogrammetryOptions {
    int max_dimension = 640;
    int num_disparities = 128;
    int block_size = 5;
    int min_inliers = 30;
    double baseline_mm = 0.0;
    double scale_reference_mm = 0.0;
    double scale_reference_units = 0.0;
    double max_depth = 10000.0;
    double max_relative_depth_jump = 0.20;
    double left_right_tolerance = 1.5;
    double fusion_voxel_size = 2.0;
    std::size_t min_fusion_observations = 1;
    int speckle_window_size = 100;
    int speckle_range = 2;
};

struct ColoredPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::uint8_t red = 255;
    std::uint8_t green = 255;
    std::uint8_t blue = 255;
};

struct ReconstructionStats {
    std::size_t images = 0;
    std::size_t registered_pairs = 0;
    std::size_t feature_matches = 0;
    std::size_t shared_tracks = 0;
    std::size_t bundle_observations = 0;
    std::size_t bundle_iterations = 0;
    double bundle_initial_rmse = 0.0;
    double bundle_final_rmse = 0.0;
    std::size_t pose_inliers = 0;
    std::size_t vertices = 0;
    std::size_t triangles = 0;
    std::size_t mesh_boundary_edges = 0;
    std::size_t mesh_non_manifold_edges = 0;
    std::size_t mesh_degenerate_faces_removed = 0;
    std::size_t mesh_duplicate_faces_removed = 0;
    std::size_t stereo_candidates = 0;
    std::size_t stereo_consistent = 0;
    std::size_t stereo_rejected = 0;
    std::size_t fused_points = 0;
    std::size_t fusion_voxels_rejected = 0;
    double applied_scale_factor = 1.0;
    double mean_pose_inlier_ratio = 0.0;
    double mean_valid_depth_ratio = 0.0;
    bool metric_scale = false;
};

std::vector<std::filesystem::path> read_image_list(const std::filesystem::path& list_path);

CameraModel calibrate_camera_from_list(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_output,
    int board_cols,
    int board_rows,
    double square_size_mm,
    const std::atomic_bool* cancel_requested = nullptr);

CameraModel load_camera_model(const std::filesystem::path& calibration_path);

std::vector<ColoredPoint> fuse_point_cloud_voxels(
    const std::vector<ColoredPoint>& points,
    double voxel_size,
    std::size_t minimum_observations = 1);

void write_ply_point_cloud(
    const std::filesystem::path& output_path,
    const std::vector<ColoredPoint>& points);

ReconstructionStats reconstruct_multiview(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_path,
    const std::filesystem::path& output_stl,
    const PhotogrammetryOptions& options,
    const std::filesystem::path& output_ply = {},
    const std::atomic_bool* cancel_requested = nullptr);

} // namespace cvcascade
