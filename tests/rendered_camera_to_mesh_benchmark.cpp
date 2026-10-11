#include "cvcascade/photogrammetry.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    fs::path root;
    try {
        // Render a deterministic, fronto-parallel textured plane at known metric
        // depth from three calibrated pinhole cameras. This exercises image loading,
        // feature tracking, global BA, dense stereo, mesh export, and PLY geometry.
        root = fs::temp_directory_path() / "cvcascade_rendered_camera_to_mesh_benchmark";
        fs::remove_all(root);
        fs::create_directories(root);

        constexpr int width = 640, height = 480;
        constexpr double focal = 600.0, plane_z_mm = 5000.0, baseline_mm = 400.0;
        const cv::Mat k = (cv::Mat_<double>(3, 3) <<
            focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0);
        cv::Mat texture(1200, 1680, CV_8UC1);
        cv::RNG rng(0x5eed1234);
        rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
        // Add deterministic, larger-scale texture so SGBM has support beyond noise.
        for (int y = 20; y < texture.rows; y += 47) {
            for (int x = 20; x < texture.cols; x += 53) {
                const int radius = 3 + ((x * 7 + y * 11) % 8);
                cv::circle(texture, cv::Point(x, y), radius,
                           cv::Scalar((x * 13 + y * 5) % 256), -1, cv::LINE_8);
            }
        }

        std::ofstream list(root / "images.txt");
        require(static_cast<bool>(list), "could not create rendered image list");
        const double x_min = -3500.0, x_span = 7000.0;
        const double y_min = -2500.0, y_span = 5000.0;
        for (int view = 0; view < 3; ++view) {
            const double camera_x = view * baseline_mm;
            const double sx = x_span / (texture.cols - 1);
            const double sy = y_span / (texture.rows - 1);
            cv::Mat h = (cv::Mat_<double>(3, 3) <<
                focal * sx / plane_z_mm, 0.0,
                focal * (x_min - camera_x) / plane_z_mm + width / 2.0,
                0.0, focal * sy / plane_z_mm,
                focal * y_min / plane_z_mm + height / 2.0,
                0.0, 0.0, 1.0);
            cv::Mat rendered;
            cv::warpPerspective(texture, rendered, h, cv::Size(width, height),
                                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(12));
            const fs::path image_path = root / ("view_" + std::to_string(view) + ".png");
            require(cv::imwrite(image_path.string(), rendered), "failed to write rendered camera image");
            list << image_path.string() << '\n';
        }
        list.close();

        cv::FileStorage calibration((root / "camera.yml").string(), cv::FileStorage::WRITE);
        require(calibration.isOpened(), "could not create rendered-scene calibration");
        calibration << "camera_matrix" << k;
        calibration << "distortion_coefficients" << cv::Mat::zeros(8, 1, CV_64F);
        calibration << "image_width" << width;
        calibration << "image_height" << height;
        calibration << "reprojection_rms" << 0.0;
        calibration << "calibration_views_used" << 3;
        calibration << "calibration_views_rejected" << 0;
        calibration.release();

        cvcascade::PhotogrammetryOptions options;
        options.max_dimension = width;
        options.num_disparities = 128;
        options.block_size = 5;
        options.min_inliers = 20;
        options.baseline_mm = baseline_mm;
        options.max_depth = 9000.0;
        options.max_relative_depth_jump = 0.30;
        options.left_right_tolerance = 2.0;
        options.fusion_voxel_size = 8.0;
        options.min_fusion_observations = 1;
        options.speckle_window_size = 80;
        options.speckle_range = 2;

        const fs::path stl = root / "reconstructed.stl";
        const fs::path ply = root / "reconstructed.ply";
        const auto stats = cvcascade::reconstruct_multiview(
            root / "images.txt", root / "camera.yml", stl, options, ply);
        require(fs::exists(stl) && fs::file_size(stl) > 84,
                "end-to-end rendered-scene reconstruction did not produce an STL mesh");
        require(stats.triangles > 100, "rendered-scene mesh has too few triangles");
        require(stats.shared_tracks >= 8, "global BA used too few shared multi-view tracks");
        require(stats.bundle_observations >= 24, "global BA used too few image observations");
        require(fs::exists(ply), "rendered-scene reconstruction did not produce a PLY point cloud");

        std::ifstream cloud(ply);
        require(static_cast<bool>(cloud), "could not read reconstructed point cloud");
        std::string line;
        while (std::getline(cloud, line) && line != "end_header") {}
        double sum_z = 0.0, sum_z2 = 0.0;
        std::size_t count = 0;
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -min_x;
        double min_y = min_x, max_y = -min_x;
        while (std::getline(cloud, line)) {
            std::istringstream row(line);
            double x = 0.0, y = 0.0, z = 0.0;
            unsigned red = 0, green = 0, blue = 0;
            if (!(row >> x >> y >> z >> red >> green >> blue)) continue;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            sum_z += z;
            sum_z2 += z * z;
            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);
            ++count;
        }
        require(count > 100, "too few fused points in rendered-scene PLY");
        const double mean_z = sum_z / count;
        const double variance_z = std::max(0.0, sum_z2 / count - mean_z * mean_z);
        const double z_stddev = std::sqrt(variance_z);
        const double depth_error = std::abs(mean_z - plane_z_mm);
        const double x_span = max_x - min_x, y_span = max_y - min_y;
        std::cout << "Deterministic rendered camera-to-mesh benchmark\n"
                  << "images=" << stats.images << "\n"
                  << "shared_tracks=" << stats.shared_tracks << "\n"
                  << "bundle_reprojection_rmse_px=" << stats.bundle_final_rmse << "\n"
                  << "mesh_triangles=" << stats.triangles << "\n"
                  << "fused_points=" << count << "\n"
                  << "known_plane_depth_mm=" << plane_z_mm << "\n"
                  << "reconstructed_mean_depth_mm=" << mean_z << "\n"
                  << "depth_error_mm=" << depth_error << "\n"
                  << "plane_depth_stddev_mm=" << z_stddev << "\n"
                  << "point_cloud_xy_span_mm=" << x_span << " x " << y_span << "\n";
        require(depth_error < 1200.0, "reconstructed plane mean depth error exceeded 1200 mm");
        require(z_stddev < 1200.0, "reconstructed plane depth spread exceeded 1200 mm");
        require(x_span > 1500.0 && y_span > 1000.0,
                "reconstructed plane extent is implausibly small");
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& e) {
        if (!root.empty()) {
            std::error_code ignored;
            fs::remove_all(root, ignored);
        }
        std::cerr << "RENDERED CAMERA-TO-MESH BENCHMARK FAILURE: " << e.what() << '\n';
        return 1;
    }
}
