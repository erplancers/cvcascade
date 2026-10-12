#include "cvcascade/bundle_adjustment.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace cvcascade {
namespace {

class DisjointSet {
public:
    explicit DisjointSet(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }
    std::size_t find(std::size_t x) {
        if (parent_[x] != x) parent_[x] = find(parent_[x]);
        return parent_[x];
    }
    void join(std::size_t a, std::size_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned char> rank_;
};

cv::Mat as_column(const cv::Mat& source) {
    cv::Mat result;
    source.reshape(1, 3).convertTo(result, CV_64F);
    return result.clone();
}

cv::Point2d project(const cv::Mat& k, const BundleCamera& camera,
                    const cv::Point3d& point) {
    const cv::Mat rvec = as_column(camera.rotation_vector);
    const cv::Mat tvec = as_column(camera.translation);
    const std::vector<cv::Point3d> object{point};
    std::vector<cv::Point2d> image;
    cv::projectPoints(object, rvec, tvec, k, cv::noArray(), image);
    return image.front();
}

double robust_cost(double squared, double delta) {
    const double residual = std::sqrt(std::max(0.0, squared));
    return residual <= delta ? 0.5 * squared : delta * (residual - 0.5 * delta);
}

struct PointNormal {
    cv::Mat h = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat g = cv::Mat::zeros(3, 1, CV_64F);
    std::map<std::size_t, cv::Mat> cross; // camera block -> 6x3
};

double total_cost(const cv::Mat& k, const std::vector<FeatureTrack>& tracks,
                  const std::vector<BundleCamera>& cameras,
                  const std::vector<cv::Point3d>& points, double huber,
                  double* squared_sum = nullptr, std::size_t* count = nullptr) {
    double cost = 0.0, squared = 0.0;
    std::size_t n = 0;
    for (const auto& track : tracks) {
        for (const auto& observation : track.observations) {
            if (observation.camera_index >= cameras.size() ||
                observation.landmark_index >= points.size()) continue;
            const cv::Point2d predicted = project(k, cameras[observation.camera_index],
                                                  points[observation.landmark_index]);
            const cv::Point2d residual = predicted - observation.pixel;
            const double e2 = residual.dot(residual);
            if (!std::isfinite(e2)) continue;
            cost += robust_cost(e2, huber);
            squared += e2;
            ++n;
        }
    }
    if (squared_sum) *squared_sum = squared;
    if (count) *count = n;
    return cost;
}

} // namespace

std::vector<FeatureTrack> build_shared_feature_tracks(
    const std::vector<cv::Mat>& images, double ratio_threshold,
    std::size_t minimum_track_length) {
    if (images.size() < 2) throw std::invalid_argument("shared feature tracks need at least two images");
    if (!(ratio_threshold > 0.0 && ratio_threshold < 1.0))
        throw std::invalid_argument("feature ratio threshold must be in (0,1)");
    if (minimum_track_length < 2) throw std::invalid_argument("minimum track length must be at least two");

    const auto detector = cv::ORB::create(6000);
    std::vector<std::vector<cv::KeyPoint>> keypoints(images.size());
    std::vector<cv::Mat> descriptors(images.size());
    std::vector<std::size_t> offsets(images.size() + 1, 0);
    for (std::size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty() || images[i].channels() != 1)
            throw std::invalid_argument("track builder expects non-empty grayscale images");
        detector->detectAndCompute(images[i], cv::noArray(), keypoints[i], descriptors[i]);
        offsets[i + 1] = offsets[i] + keypoints[i].size();
    }
    if (offsets.back() == 0) return {};
    DisjointSet sets(offsets.back());
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Match every overlapping pair, not only adjacent frames, so tracks can
    // bridge a missed detection in one image and provide loop-closure constraints.
    for (std::size_t i = 0; i < images.size(); ++i) {
        if (descriptors[i].empty()) continue;
        for (std::size_t j = i + 1; j < images.size(); ++j) {
            if (descriptors[j].empty()) continue;
            std::vector<std::vector<cv::DMatch>> candidates;
            matcher.knnMatch(descriptors[i], descriptors[j], candidates, 2);
            std::vector<cv::DMatch> ratio_matches;
            std::vector<cv::Point2f> source_points, target_points;
            for (const auto& pair : candidates) {
                if (pair.size() < 2 || pair[0].distance >= ratio_threshold * pair[1].distance) continue;
                ratio_matches.push_back(pair[0]);
                source_points.push_back(keypoints[i][pair[0].queryIdx].pt);
                target_points.push_back(keypoints[j][pair[0].trainIdx].pt);
            }
            if (ratio_matches.size() < 8) continue;
            cv::Mat inlier_mask;
            cv::findFundamentalMat(source_points, target_points, cv::FM_RANSAC, 2.0, 0.995, inlier_mask);
            if (inlier_mask.empty()) continue;
            for (std::size_t m = 0; m < ratio_matches.size(); ++m) {
                if (inlier_mask.at<unsigned char>(static_cast<int>(m)) == 0) continue;
                const auto& match = ratio_matches[m];
                sets.join(offsets[i] + static_cast<std::size_t>(match.queryIdx),
                          offsets[j] + static_cast<std::size_t>(match.trainIdx));
            }
        }
    }

    std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> groups;
    for (std::size_t image = 0; image < images.size(); ++image) {
        for (std::size_t key = 0; key < keypoints[image].size(); ++key) {
            groups[sets.find(offsets[image] + key)].push_back({image, key});
        }
    }
    std::vector<FeatureTrack> tracks;
    for (const auto& entry : groups) {
        std::map<std::size_t, cv::Point2d> unique_views;
        for (const auto& node : entry.second) {
            const auto image = node.first;
            const auto& pt = keypoints[image][node.second].pt;
            // Conflicting duplicate detections in one view are ambiguous; retain
            // the first deterministically and keep one observation per camera.
            unique_views.emplace(image, cv::Point2d(pt.x, pt.y));
        }
        if (unique_views.size() < minimum_track_length) continue;
        FeatureTrack track;
        const std::size_t landmark_id = tracks.size();
        for (const auto& view : unique_views)
            track.observations.push_back({view.first, landmark_id, view.second});
        tracks.push_back(std::move(track));
    }
    return tracks;
}

BundleAdjustmentReport optimize_global_bundle(
    const cv::Mat& camera_matrix, const std::vector<FeatureTrack>& tracks,
    std::vector<BundleCamera>& cameras, std::vector<cv::Point3d>& landmarks,
    const BundleAdjustmentOptions& options) {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3)
        throw std::invalid_argument("bundle adjustment requires a 3x3 camera matrix");
    if (cameras.size() < 2 || landmarks.empty())
        throw std::invalid_argument("bundle adjustment requires at least two cameras and one landmark");
    if (options.max_iterations < 1 || !(options.huber_delta > 0.0) ||
        !(options.initial_damping > 0.0))
        throw std::invalid_argument("invalid bundle-adjustment options");

    cv::Mat k;
    camera_matrix.convertTo(k, CV_64F);
    for (auto& camera : cameras) {
        camera.rotation_vector = as_column(camera.rotation_vector);
        camera.translation = as_column(camera.translation);
    }
    BundleAdjustmentReport report;
    report.tracks = tracks.size();
    double initial_squared = 0.0;
    std::size_t initial_count = 0;
    double current_cost = total_cost(k, tracks, cameras, landmarks, options.huber_delta,
                                     &initial_squared, &initial_count);
    if (initial_count < 8) throw std::runtime_error("bundle adjustment has fewer than eight usable observations");
    report.observations = initial_count;
    report.initial_rmse = std::sqrt(initial_squared / initial_count);
    double damping = options.initial_damping;
    const std::size_t variable_cameras = cameras.size() - 1; // camera 0 anchors the gauge
    const int camera_dofs = static_cast<int>(variable_cameras * 6);
    const double fixed_baseline = cv::norm(cameras[1].translation);
    if (!(fixed_baseline > 1e-12))
        throw std::invalid_argument("camera 1 must have a non-zero baseline to fix monocular scale");

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        cv::Mat hcc = cv::Mat::zeros(camera_dofs, camera_dofs, CV_64F);
        cv::Mat gc = cv::Mat::zeros(camera_dofs, 1, CV_64F);
        std::vector<PointNormal> point_normals(landmarks.size());
        std::size_t usable = 0;

        for (const auto& track : tracks) {
            for (const auto& obs : track.observations) {
                if (obs.camera_index >= cameras.size() || obs.landmark_index >= landmarks.size()) continue;
                const auto ci = obs.camera_index, pi = obs.landmark_index;
                const cv::Point2d prediction = project(k, cameras[ci], landmarks[pi]);
                cv::Mat residual = (cv::Mat_<double>(2, 1) <<
                    prediction.x - obs.pixel.x, prediction.y - obs.pixel.y);
                const double norm = cv::norm(residual);
                if (!std::isfinite(norm)) { ++report.rejected_observations; continue; }
                const double weight = norm <= options.huber_delta || norm == 0.0
                    ? 1.0 : options.huber_delta / norm;

                cv::Mat jp = cv::Mat::zeros(2, 3, CV_64F);
                for (int axis = 0; axis < 3; ++axis) {
                    cv::Point3d shifted = landmarks[pi];
                    double* coordinates[] = {&shifted.x, &shifted.y, &shifted.z};
                    const double step = 1e-6 * std::max(1.0, std::abs(*coordinates[axis]));
                    *coordinates[axis] += step;
                    const auto p = project(k, cameras[ci], shifted);
                    jp.at<double>(0, axis) = (p.x - prediction.x) / step;
                    jp.at<double>(1, axis) = (p.y - prediction.y) / step;
                }

                cv::Mat jc = cv::Mat::zeros(2, 6, CV_64F);
                if (ci != 0) {
                    for (int axis = 0; axis < 6; ++axis) {
                        BundleCamera shifted = cameras[ci];
                        cv::Mat& parameter = axis < 3 ? shifted.rotation_vector : shifted.translation;
                        const int component = axis < 3 ? axis : axis - 3;
                        const double old_value = parameter.at<double>(component);
                        const double step = 1e-6 * std::max(1.0, std::abs(old_value));
                        parameter.at<double>(component) += step;
                        const auto p = project(k, shifted, landmarks[pi]);
                        jc.at<double>(0, axis) = (p.x - prediction.x) / step;
                        jc.at<double>(1, axis) = (p.y - prediction.y) / step;
                    }
                    const int base = static_cast<int>((ci - 1) * 6);
                    cv::Mat jc_t = jc.t();
                    cv::Mat weighted_jc = weight * jc;
                    cv::Mat weighted_r = weight * residual;
                    hcc(cv::Range(base, base + 6), cv::Range(base, base + 6)) += jc_t * weighted_jc;
                    gc.rowRange(base, base + 6) += jc_t * weighted_r;
                    auto& cross = point_normals[pi].cross[ci];
                    if (cross.empty()) cross = cv::Mat::zeros(6, 3, CV_64F);
                    cross += jc_t * (weight * jp);
                }
                auto& pn = point_normals[pi];
                pn.h += jp.t() * (weight * jp);
                pn.g += jp.t() * (weight * residual);
                ++usable;
            }
        }
        if (usable < 8) throw std::runtime_error("bundle adjustment lost too many valid observations");

        std::vector<cv::Mat> point_inverse(landmarks.size());
        cv::Mat schur = hcc.clone();
        cv::Mat rhs = -gc;
        for (std::size_t pi = 0; pi < point_normals.size(); ++pi) {
            auto& pn = point_normals[pi];
            if (cv::norm(pn.h) == 0.0) continue;
            pn.h += cv::Mat::eye(3, 3, CV_64F) * damping;
            if (!cv::invert(pn.h, point_inverse[pi], cv::DECOMP_SVD)) continue;
            const cv::Mat inv_g = point_inverse[pi] * pn.g;
            for (const auto& left : pn.cross) {
                const int li = static_cast<int>((left.first - 1) * 6);
                rhs.rowRange(li, li + 6) += left.second * inv_g;
                for (const auto& right : pn.cross) {
                    const int ri = static_cast<int>((right.first - 1) * 6);
                    schur(cv::Range(li, li + 6), cv::Range(ri, ri + 6)) -=
                        left.second * point_inverse[pi] * right.second.t();
                }
            }
        }

        for (int i = 0; i < camera_dofs; ++i) schur.at<double>(i, i) += damping;
        cv::Mat dc;
        if (camera_dofs > 0 && !cv::solve(schur, rhs, dc, cv::DECOMP_SVD)) break;
        auto candidate_cameras = cameras;
        auto candidate_points = landmarks;
        for (std::size_t ci = 1; ci < candidate_cameras.size(); ++ci) {
            const int base = static_cast<int>((ci - 1) * 6);
            for (int axis = 0; axis < 3; ++axis) {
                candidate_cameras[ci].rotation_vector.at<double>(axis) += dc.at<double>(base + axis);
                candidate_cameras[ci].translation.at<double>(axis) += dc.at<double>(base + 3 + axis);
            }
        }
        for (std::size_t pi = 0; pi < point_normals.size(); ++pi) {
            if (point_inverse[pi].empty()) continue;
            cv::Mat gradient = point_normals[pi].g.clone();
            for (const auto& cross : point_normals[pi].cross) {
                const int base = static_cast<int>((cross.first - 1) * 6);
                gradient += cross.second.t() * dc.rowRange(base, base + 6);
            }
            const cv::Mat dp = -point_inverse[pi] * gradient;
            candidate_points[pi].x += dp.at<double>(0);
            candidate_points[pi].y += dp.at<double>(1);
            candidate_points[pi].z += dp.at<double>(2);
        }
        // Monocular reprojection has a scale gauge. Apply one common scale to
        // every free camera translation and landmark, preserving projections
        // while pinning the first baseline to its initial length.
        const double candidate_baseline = cv::norm(candidate_cameras[1].translation);
        if (candidate_baseline > 1e-12) {
            const double gauge_scale = fixed_baseline / candidate_baseline;
            for (std::size_t ci = 1; ci < candidate_cameras.size(); ++ci)
                candidate_cameras[ci].translation *= gauge_scale;
            for (auto& point : candidate_points) {
                point.x *= gauge_scale;
                point.y *= gauge_scale;
                point.z *= gauge_scale;
            }
        }

        const double candidate_cost = total_cost(k, tracks, candidate_cameras,
                                                  candidate_points, options.huber_delta);
        report.iterations = iteration + 1;
        if (std::isfinite(candidate_cost) && candidate_cost < current_cost) {
            cameras = std::move(candidate_cameras);
            landmarks = std::move(candidate_points);
            const double improvement = current_cost - candidate_cost;
            current_cost = candidate_cost;
            damping = std::max(1e-9, damping * 0.35);
            report.improved = true;
            if (improvement < 1e-10 * std::max(1.0, current_cost)) break;
        } else {
            damping = std::min(1e12, damping * 10.0);
            if (damping >= 1e12) break;
        }
    }

    double final_squared = 0.0;
    std::size_t final_count = 0;
    total_cost(k, tracks, cameras, landmarks, options.huber_delta, &final_squared, &final_count);
    report.final_rmse = final_count ? std::sqrt(final_squared / final_count) :
        std::numeric_limits<double>::infinity();
    return report;
}

} // namespace cvcascade
