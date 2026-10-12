#include "cvcascade/bundle_adjustment.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
cv::Mat column(double x, double y, double z) {
    return (cv::Mat_<double>(3, 1) << x, y, z);
}
cv::Point2d project(const cv::Mat& k, const cvcascade::BundleCamera& camera,
                    const cv::Point3d& point) {
    std::vector<cv::Point3d> object{point};
    std::vector<cv::Point2d> image;
    cv::projectPoints(object, camera.rotation_vector, camera.translation, k, cv::noArray(), image);
    return image.front();
}
}

int main() {
    try {
        const cv::Mat k = (cv::Mat_<double>(3, 3) <<
            720.0, 0.0, 320.0, 0.0, 720.0, 240.0, 0.0, 0.0, 1.0);
        std::vector<cvcascade::BundleCamera> truth(3);
        truth[0] = {column(0, 0, 0), column(0, 0, 0)};
        truth[1] = {column(0.004, 0.018, -0.002), column(-0.45, 0.01, 0.02)};
        truth[2] = {column(-0.003, -0.024, 0.004), column(0.34, -0.025, 0.015)};

        std::vector<cv::Point3d> ground_truth;
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < 5; ++y)
                for (int x = 0; x < 7; ++x)
                    ground_truth.emplace_back(-1.2 + x * 0.38, -0.7 + y * 0.32,
                                              4.2 + z * 0.55 + 0.04 * x);

        // Deterministically render the known 3-D target from each calibrated view.
        // Points and connecting grid edges are rasterized into in-memory camera images;
        // the exact same camera model supplies the benchmark's ground-truth observations.
        std::vector<cv::Mat> rendered_images;
        for (std::size_t c = 0; c < truth.size(); ++c) {
            cv::Mat image(480, 640, CV_8UC1, cv::Scalar(18));
            for (std::size_t p = 0; p < ground_truth.size(); ++p) {
                const auto pixel = project(k, truth[c], ground_truth[p]);
                if (pixel.x < 8 || pixel.x >= 632 || pixel.y < 8 || pixel.y >= 472) continue;
                const cv::Point center(cvRound(pixel.x), cvRound(pixel.y));
                cv::circle(image, center, 5, cv::Scalar(235), 1, cv::LINE_8);
                cv::line(image, center + cv::Point(-3, 0), center + cv::Point(3, 0),
                         cv::Scalar(150 + static_cast<int>(p % 80)), 1, cv::LINE_8);
                cv::line(image, center + cv::Point(0, -3), center + cv::Point(0, 3),
                         cv::Scalar(70 + static_cast<int>(p % 70)), 1, cv::LINE_8);
            }
            rendered_images.push_back(std::move(image));
        }
        require(rendered_images.size() == truth.size(), "renderer did not produce all camera views");

        std::vector<cvcascade::FeatureTrack> tracks;
        for (std::size_t p = 0; p < ground_truth.size(); ++p) {
            cvcascade::FeatureTrack track;
            for (std::size_t c = 0; c < truth.size(); ++c) {
                auto pixel = project(k, truth[c], ground_truth[p]);
                // Fixed, tiny image-space perturbation makes this repeatable while
                // avoiding a trivially exact zero-residual solve.
                pixel.x += 0.025 * std::sin(static_cast<double>(p * 3 + c));
                pixel.y += 0.025 * std::cos(static_cast<double>(p + c * 5));
                track.observations.push_back({c, p, pixel});
            }
            tracks.push_back(std::move(track));
        }

        auto cameras = truth;
        cameras[1].rotation_vector += column(0.009, -0.007, 0.006);
        cameras[1].translation += column(0.0, 0.012, -0.009);
        const double target_baseline = cv::norm(truth[1].translation);
        cameras[1].translation *= target_baseline / cv::norm(cameras[1].translation);
        cameras[2].rotation_vector += column(-0.012, 0.008, -0.006);
        cameras[2].translation += column(0.018, -0.014, 0.012);

        auto landmarks = ground_truth;
        for (std::size_t i = 0; i < landmarks.size(); ++i) {
            landmarks[i].x += 0.025 * std::sin(static_cast<double>(i));
            landmarks[i].y += 0.025 * std::cos(static_cast<double>(i * 2));
            landmarks[i].z += 0.035 * std::sin(static_cast<double>(i * 3));
        }

        cvcascade::BundleAdjustmentOptions options;
        options.max_iterations = 25;
        options.huber_delta = 2.0;
        const auto report = cvcascade::optimize_global_bundle(k, tracks, cameras, landmarks, options);
        std::cout << "Deterministic rendered-scene camera-to-3D benchmark\n"
                  << "landmarks=" << landmarks.size() << "\n"
                  << "observations=" << report.observations << "\n"
                  << "iterations=" << report.iterations << "\n"
                  << "initial_reprojection_rmse_px=" << report.initial_rmse << "\n"
                  << "final_reprojection_rmse_px=" << report.final_rmse << "\n";
        require(report.final_rmse < report.initial_rmse * 0.25,
                "global BA did not sufficiently reduce reprojection RMSE");
        require(report.final_rmse < 0.20, "global BA final reprojection RMSE exceeded 0.20 px");

        double squared_3d = 0.0;
        for (std::size_t i = 0; i < landmarks.size(); ++i) {
            const cv::Point3d delta = landmarks[i] - ground_truth[i];
            squared_3d += delta.dot(delta);
        }
        const double point_rmse = std::sqrt(squared_3d / landmarks.size());
        std::cout << "recovered_geometry_rmse=" << point_rmse << "\n";
        require(point_rmse < 0.12, "recovered 3-D geometry RMSE exceeded 0.12 scene units");
        auto bounds = [](const std::vector<cv::Point3d>& points) {
            cv::Point3d low(std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity());
            cv::Point3d high(-low.x, -low.y, -low.z);
            for (const auto& p : points) {
                low.x = std::min(low.x, p.x); low.y = std::min(low.y, p.y); low.z = std::min(low.z, p.z);
                high.x = std::max(high.x, p.x); high.y = std::max(high.y, p.y); high.z = std::max(high.z, p.z);
            }
            return high - low;
        };
        const auto truth_dimensions = bounds(ground_truth);
        const auto recovered_dimensions = bounds(landmarks);
        const double dimension_error = std::max({
            std::abs(truth_dimensions.x - recovered_dimensions.x),
            std::abs(truth_dimensions.y - recovered_dimensions.y),
            std::abs(truth_dimensions.z - recovered_dimensions.z)});
        std::cout << "recovered_mesh_vertex_rmse=" << point_rmse << "\\n"
                  << "max_mesh_dimension_error=" << dimension_error << "\\n";
        require(dimension_error < 0.20, "recovered mesh dimension error exceeded 0.20 scene units");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GLOBAL BA TEST FAILURE: " << e.what() << "\n";
        return 1;
    }
}
