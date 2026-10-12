#pragma once

#include <opencv2/core.hpp>
#include <cstddef>
#include <vector>

namespace cvcascade {

struct TrackObservation {
    std::size_t camera_index = 0;
    std::size_t landmark_index = 0;
    cv::Point2d pixel;
};

struct FeatureTrack {
    std::vector<TrackObservation> observations;
};

struct BundleCamera {
    cv::Mat rotation_vector; // world-to-camera Rodrigues vector, CV_64F 3x1
    cv::Mat translation;     // world-to-camera translation, CV_64F 3x1
};

struct BundleAdjustmentOptions {
    int max_iterations = 15;
    double huber_delta = 2.0;
    double initial_damping = 1e-3;
    double minimum_track_length = 2.0;
};

struct BundleAdjustmentReport {
    std::size_t tracks = 0;
    std::size_t observations = 0;
    std::size_t rejected_observations = 0;
    int iterations = 0;
    double initial_rmse = 0.0;
    double final_rmse = 0.0;
    bool improved = false;
};

// Matches ORB features across all image pairs, merges pairwise correspondences
// into persistent tracks, and guarantees at most one observation per image/track.
std::vector<FeatureTrack> build_shared_feature_tracks(
    const std::vector<cv::Mat>& grayscale_images,
    double ratio_threshold = 0.75,
    std::size_t minimum_track_length = 2);

// Jointly minimizes robust reprojection error over all camera poses and 3-D
// landmarks with a damped Schur-complement Gauss-Newton/LM solver. Camera 0 is
// fixed to remove the global rigid-motion gauge; caller must supply an initial
// reconstruction in a consistent coordinate frame.
BundleAdjustmentReport optimize_global_bundle(
    const cv::Mat& camera_matrix,
    const std::vector<FeatureTrack>& tracks,
    std::vector<BundleCamera>& cameras,
    std::vector<cv::Point3d>& landmarks,
    const BundleAdjustmentOptions& options = {});

} // namespace cvcascade
