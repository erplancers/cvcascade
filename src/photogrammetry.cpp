#include "cvcascade/photogrammetry.hpp"

#include <BRep_Builder.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <StlAPI_Writer.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Pnt.hxx>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cvcascade {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

cv::Mat read_gray_image(const std::filesystem::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("cannot read image: " + path.string());
    return image;
}

cv::Mat resize_image(const cv::Mat& input, int maximum_dimension) {
    const int largest = std::max(input.cols, input.rows);
    if (largest <= maximum_dimension) return input;
    const double scale = static_cast<double>(maximum_dimension) / largest;
    cv::Mat resized;
    cv::resize(input, resized,
               cv::Size(std::max(2, cvRound(input.cols * scale)),
                        std::max(2, cvRound(input.rows * scale))),
               0.0, 0.0, cv::INTER_AREA);
    return resized;
}

CameraModel camera_for_size(const CameraModel& source, cv::Size target_size) {
    if (source.image_size.width <= 0 || source.image_size.height <= 0)
        throw std::runtime_error("calibration file has invalid image dimensions");
    const double sx = static_cast<double>(target_size.width) / source.image_size.width;
    const double sy = static_cast<double>(target_size.height) / source.image_size.height;
    if (std::abs(sx - sy) > 0.02 * std::max(sx, sy))
        throw std::runtime_error("calibration and reconstruction images have different aspect ratios");

    CameraModel scaled = source;
    source.camera_matrix.convertTo(scaled.camera_matrix, CV_64F);
    scaled.camera_matrix = scaled.camera_matrix.clone();
    source.distortion_coefficients.convertTo(scaled.distortion_coefficients, CV_64F);
    scaled.camera_matrix.at<double>(0, 0) *= sx;
    scaled.camera_matrix.at<double>(0, 2) *= sx;
    scaled.camera_matrix.at<double>(1, 1) *= sy;
    scaled.camera_matrix.at<double>(1, 2) *= sy;
    scaled.image_size = target_size;
    return scaled;
}

struct RelativePose {
    cv::Mat rotation;      // X_next = R * X_current + t
    cv::Mat translation;   // Unit translation unless baseline-mm is supplied
    std::size_t good_matches = 0;
    std::size_t inliers = 0;
};

RelativePose estimate_relative_pose(
    const cv::Mat& first,
    const cv::Mat& second,
    const CameraModel& camera,
    int minimum_inliers) {

    const auto detector = cv::ORB::create(6000);
    std::vector<cv::KeyPoint> keypoints_first, keypoints_second;
    cv::Mat descriptors_first, descriptors_second;
    detector->detectAndCompute(first, cv::noArray(), keypoints_first, descriptors_first);
    detector->detectAndCompute(second, cv::noArray(), keypoints_second, descriptors_second);
    if (descriptors_first.empty() || descriptors_second.empty())
        throw std::runtime_error("no usable ORB features in an adjacent image pair");

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher.knnMatch(descriptors_first, descriptors_second, candidates, 2);
    std::vector<cv::Point2f> points_first, points_second;
    points_first.reserve(candidates.size());
    points_second.reserve(candidates.size());

    for (const auto& pair : candidates) {
        if (pair.size() < 2 || pair[0].distance >= 0.75F * pair[1].distance) continue;
        points_first.push_back(keypoints_first[pair[0].queryIdx].pt);
        points_second.push_back(keypoints_second[pair[0].trainIdx].pt);
    }
    if (points_first.size() < static_cast<std::size_t>(std::max(8, minimum_inliers)))
        throw std::runtime_error("too few feature matches; use textured images with substantial overlap");

    std::vector<cv::Point2f> undistorted_first, undistorted_second;
    cv::undistortPoints(points_first, undistorted_first, camera.camera_matrix,
                        camera.distortion_coefficients, cv::noArray(), camera.camera_matrix);
    cv::undistortPoints(points_second, undistorted_second, camera.camera_matrix,
                        camera.distortion_coefficients, cv::noArray(), camera.camera_matrix);

    cv::Mat mask;
    const cv::Mat essential = cv::findEssentialMat(
        undistorted_first, undistorted_second, camera.camera_matrix,
        cv::RANSAC, 0.999, 1.5, mask);
    if (essential.empty())
        throw std::runtime_error("essential-matrix estimation failed for an adjacent image pair");

    RelativePose pose;
    pose.good_matches = points_first.size();
    const int recovered = cv::recoverPose(
        essential, undistorted_first, undistorted_second, camera.camera_matrix,
        pose.rotation, pose.translation, mask);
    if (recovered < minimum_inliers)
        throw std::runtime_error("too few geometrically consistent matches to recover camera motion");

    pose.inliers = static_cast<std::size_t>(recovered);
    pose.rotation.convertTo(pose.rotation, CV_64F);
    pose.translation.convertTo(pose.translation, CV_64F);
    return pose;
}

bool valid_depth(float a, float b, double relative_jump) {
    if (!(a > 0.0F) || !(b > 0.0F) || !std::isfinite(a) || !std::isfinite(b)) return false;
    return std::abs(static_cast<double>(a) - b) <= relative_jump * std::min(a, b);
}

std::size_t append_dense_stereo_mesh(
    const cv::Mat& first,
    const cv::Mat& second,
    const CameraModel& camera,
    const RelativePose& relative_pose,
    const cv::Mat& world_to_camera_rotation,
    const cv::Mat& world_to_camera_translation,
    const PhotogrammetryOptions& options,
    BRep_Builder& builder,
    TopoDS_Compound& compound,
    std::size_t& vertex_count) {

    cv::Mat r1, r2, p1, p2, q;
    cv::stereoRectify(camera.camera_matrix, camera.distortion_coefficients,
                      camera.camera_matrix, camera.distortion_coefficients,
                      first.size(), relative_pose.rotation, relative_pose.translation,
                      r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0.0, first.size());

    cv::Mat map11, map12, map21, map22, rectified_first, rectified_second;
    cv::initUndistortRectifyMap(camera.camera_matrix, camera.distortion_coefficients,
                                r1, p1, first.size(), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(camera.camera_matrix, camera.distortion_coefficients,
                                r2, p2, second.size(), CV_32FC1, map21, map22);
    cv::remap(first, rectified_first, map11, map12, cv::INTER_LINEAR);
    cv::remap(second, rectified_second, map21, map22, cv::INTER_LINEAR);

    const int block = options.block_size;
    auto stereo = cv::StereoSGBM::create(
        0, options.num_disparities, block,
        8 * block * block, 32 * block * block,
        1, 31, 10, 100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat disparity_fixed, disparity, xyz;
    stereo->compute(rectified_first, rectified_second, disparity_fixed);
    disparity_fixed.convertTo(disparity, CV_32F, 1.0 / 16.0);
    cv::reprojectImageTo3D(disparity, xyz, q, false, CV_32F);

    const int rows = xyz.rows;
    const int cols = xyz.cols;
    std::vector<Standard_Integer> node_index(static_cast<std::size_t>(rows) * cols, 0);
    std::vector<float> depth(static_cast<std::size_t>(rows) * cols, 0.0F);
    std::vector<gp_Pnt> vertices;
    vertices.reserve(static_cast<std::size_t>(rows) * cols / 2);

    const cv::Mat rectified_to_camera = r1.t();
    const cv::Mat camera_to_world_rotation = world_to_camera_rotation.t();
    const cv::Mat camera_to_world_translation = -camera_to_world_rotation * world_to_camera_translation;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y) * cols + x;
            const float d = disparity.at<float>(y, x);
            const cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
            if (!(d > 0.5F) || !std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                !std::isfinite(point[2]) || !(point[2] > 0.0F) || point[2] > options.max_depth)
                continue;

            const cv::Mat rectified_point = (cv::Mat_<double>(3, 1) << point[0], point[1], point[2]);
            const cv::Mat camera_point = rectified_to_camera * rectified_point;
            const cv::Mat world_point = camera_to_world_rotation * camera_point + camera_to_world_translation;
            const double wx = world_point.at<double>(0), wy = world_point.at<double>(1), wz = world_point.at<double>(2);
            if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz)) continue;

            vertices.emplace_back(wx, wy, wz);
            node_index[offset] = static_cast<Standard_Integer>(vertices.size());
            depth[offset] = point[2];
        }
    }

    std::vector<std::array<Standard_Integer, 3>> triangles;
    triangles.reserve(static_cast<std::size_t>(rows - 1) * (cols - 1));
    for (int y = 0; y < rows - 1; ++y) {
        for (int x = 0; x < cols - 1; ++x) {
            const std::size_t a0 = static_cast<std::size_t>(y) * cols + x;
            const std::size_t b0 = a0 + 1;
            const std::size_t c0 = a0 + cols;
            const std::size_t d0 = c0 + 1;
            const auto a = node_index[a0], b = node_index[b0], c = node_index[c0], d = node_index[d0];
            if (a == 0 || b == 0 || c == 0 || d == 0) continue;
            if (!valid_depth(depth[a0], depth[b0], options.max_relative_depth_jump) ||
                !valid_depth(depth[a0], depth[c0], options.max_relative_depth_jump) ||
                !valid_depth(depth[d0], depth[b0], options.max_relative_depth_jump) ||
                !valid_depth(depth[d0], depth[c0], options.max_relative_depth_jump)) continue;
            triangles.push_back({a, d, b});
            triangles.push_back({a, c, d});
        }
    }
    if (triangles.empty()) return 0;

    Handle(Poly_Triangulation) triangulation = new Poly_Triangulation(
        static_cast<Standard_Integer>(vertices.size()),
        static_cast<Standard_Integer>(triangles.size()), Standard_False);
    for (std::size_t i = 0; i < vertices.size(); ++i)
        triangulation->SetNode(static_cast<Standard_Integer>(i + 1), vertices[i]);
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        const auto& triangle = triangles[i];
        triangulation->SetTriangle(static_cast<Standard_Integer>(i + 1),
            Poly_Triangle(triangle[0], triangle[1], triangle[2]));
    }

    TopoDS_Face face;
    builder.MakeFace(face, triangulation);
    builder.Add(compound, face);
    vertex_count += vertices.size();
    return triangles.size();
}

} // namespace

std::vector<std::filesystem::path> read_image_list(const std::filesystem::path& list_path) {
    std::ifstream input(list_path);
    if (!input) throw std::runtime_error("cannot open image list: " + list_path.string());

    std::vector<std::filesystem::path> paths;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        std::filesystem::path path(line);
        if (path.is_relative() && !std::filesystem::exists(path)) {
            const auto relative_to_list = list_path.parent_path() / path;
            if (std::filesystem::exists(relative_to_list)) path = relative_to_list;
        }
        paths.push_back(path.lexically_normal());
    }
    if (paths.empty()) throw std::runtime_error("image list contains no paths: " + list_path.string());
    return paths;
}

CameraModel calibrate_camera_from_list(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_output,
    int board_cols,
    int board_rows,
    double square_size_mm) {

    if (board_cols < 2 || board_rows < 2)
        throw std::invalid_argument("chessboard inner-corner columns and rows must be at least 2");
    if (!(square_size_mm > 0.0) || !std::isfinite(square_size_mm))
        throw std::invalid_argument("chessboard square size must be finite and positive");
    if (calibration_output.empty()) throw std::invalid_argument("calibration output path must not be empty");

    const auto paths = read_image_list(image_list);
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    cv::Size image_size;

    for (const auto& path : paths) {
        const cv::Mat gray = read_gray_image(path);
        if (image_size.empty()) image_size = gray.size();
        if (gray.size() != image_size)
            throw std::runtime_error("calibration images must all have the same dimensions");

        std::vector<cv::Point2f> corners;
        if (!cv::findChessboardCorners(gray, cv::Size(board_cols, board_rows), corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE)) continue;
        cv::Mat refined = gray.clone();
        cv::cornerSubPix(refined, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));

        std::vector<cv::Point3f> object;
        object.reserve(static_cast<std::size_t>(board_cols) * board_rows);
        for (int y = 0; y < board_rows; ++y)
            for (int x = 0; x < board_cols; ++x)
                object.emplace_back(static_cast<float>(x * square_size_mm),
                                     static_cast<float>(y * square_size_mm), 0.0F);
        object_points.push_back(std::move(object));
        image_points.push_back(std::move(corners));
    }

    if (object_points.size() < 3)
        throw std::runtime_error("camera calibration needs at least 3 detected chessboard views; use 5-15 varied views for accuracy");

    CameraModel model;
    model.image_size = image_size;
    model.camera_matrix = cv::initCameraMatrix2D(object_points, image_points, image_size);
    model.distortion_coefficients = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rotations, translations;
    model.reprojection_rms = cv::calibrateCamera(
        object_points, image_points, image_size, model.camera_matrix,
        model.distortion_coefficients, rotations, translations);

    cv::FileStorage output(calibration_output.string(), cv::FileStorage::WRITE);
    if (!output.isOpened()) throw std::runtime_error("cannot write calibration file: " + calibration_output.string());
    output << "camera_matrix" << model.camera_matrix;
    output << "distortion_coefficients" << model.distortion_coefficients;
    output << "image_width" << image_size.width;
    output << "image_height" << image_size.height;
    output << "reprojection_rms" << model.reprojection_rms;
    output << "board_cols" << board_cols;
    output << "board_rows" << board_rows;
    output << "square_size_mm" << square_size_mm;
    output.release();
    return model;
}

CameraModel load_camera_model(const std::filesystem::path& calibration_path) {
    cv::FileStorage input(calibration_path.string(), cv::FileStorage::READ);
    if (!input.isOpened()) throw std::runtime_error("cannot open calibration file: " + calibration_path.string());

    CameraModel model;
    input["camera_matrix"] >> model.camera_matrix;
    input["distortion_coefficients"] >> model.distortion_coefficients;
    input["image_width"] >> model.image_size.width;
    input["image_height"] >> model.image_size.height;
    input["reprojection_rms"] >> model.reprojection_rms;
    if (model.camera_matrix.rows != 3 || model.camera_matrix.cols != 3 ||
        model.distortion_coefficients.empty() || model.image_size.width <= 0 ||
        model.image_size.height <= 0)
        throw std::runtime_error("invalid calibration file (camera matrix, distortion, or image dimensions missing)");

    model.camera_matrix.convertTo(model.camera_matrix, CV_64F);
    model.distortion_coefficients.convertTo(model.distortion_coefficients, CV_64F);
    return model;
}

ReconstructionStats reconstruct_multiview(
    const std::filesystem::path& image_list,
    const std::filesystem::path& calibration_path,
    const std::filesystem::path& output_stl,
    const PhotogrammetryOptions& options) {

    if (options.max_dimension < 64) throw std::invalid_argument("max-dimension must be at least 64");
    if (options.num_disparities < 16 || options.num_disparities % 16 != 0)
        throw std::invalid_argument("num-disparities must be a positive multiple of 16");
    if (options.block_size < 3 || options.block_size % 2 == 0)
        throw std::invalid_argument("block-size must be odd and at least 3");
    if (options.min_inliers < 8) throw std::invalid_argument("min-inliers must be at least 8");
    if (!(options.baseline_mm >= 0.0) || !std::isfinite(options.baseline_mm))
        throw std::invalid_argument("baseline-mm must be finite and non-negative");
    if (!(options.max_depth > 0.0) || !std::isfinite(options.max_depth))
        throw std::invalid_argument("max-depth must be finite and positive");
    if (!(options.max_relative_depth_jump > 0.0 && options.max_relative_depth_jump <= 1.0))
        throw std::invalid_argument("depth-jump must be in the interval (0, 1]");
    if (output_stl.empty()) throw std::invalid_argument("output STL path must not be empty");

    const auto paths = read_image_list(image_list);
    if (paths.size() < 2) throw std::runtime_error("multi-view reconstruction needs at least two images");

    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (const auto& path : paths)
        images.push_back(resize_image(read_gray_image(path), options.max_dimension));
    for (std::size_t i = 1; i < images.size(); ++i) {
        if (images[i].size() != images[0].size())
            throw std::runtime_error("all reconstruction images must have the same size/aspect ratio");
    }
    if (options.num_disparities >= images[0].cols)
        throw std::invalid_argument("num-disparities must be smaller than the reconstructed image width");

    const CameraModel calibration = load_camera_model(calibration_path);
    const CameraModel camera = camera_for_size(calibration, images[0].size());

    std::vector<RelativePose> pair_poses;
    pair_poses.reserve(images.size() - 1);
    std::vector<cv::Mat> camera_rotations(images.size());
    std::vector<cv::Mat> camera_translations(images.size());
    camera_rotations[0] = cv::Mat::eye(3, 3, CV_64F);
    camera_translations[0] = cv::Mat::zeros(3, 1, CV_64F);

    ReconstructionStats stats;
    stats.images = images.size();
    stats.metric_scale = options.baseline_mm > 0.0;

    for (std::size_t i = 1; i < images.size(); ++i) {
        RelativePose pose = estimate_relative_pose(images[i - 1], images[i], camera, options.min_inliers);
        if (options.baseline_mm > 0.0) {
            const double norm = cv::norm(pose.translation);
            if (!(norm > 0.0)) throw std::runtime_error("recovered camera translation has zero length");
            pose.translation *= options.baseline_mm / norm;
        }
        camera_rotations[i] = pose.rotation * camera_rotations[i - 1];
        camera_translations[i] = pose.rotation * camera_translations[i - 1] + pose.translation;
        stats.feature_matches += pose.good_matches;
        stats.pose_inliers += pose.inliers;
        pair_poses.push_back(std::move(pose));
    }

    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (std::size_t i = 0; i < pair_poses.size(); ++i) {
        const std::size_t triangles = append_dense_stereo_mesh(
            images[i], images[i + 1], camera, pair_poses[i],
            camera_rotations[i], camera_translations[i], options,
            builder, compound, stats.vertices);
        if (triangles > 0) {
            stats.triangles += triangles;
            ++stats.registered_pairs;
        }
    }
    if (stats.triangles == 0)
        throw std::runtime_error("dense stereo produced no valid triangles; check overlap, lighting, texture and disparity range");

    StlAPI_Writer writer;
    if (!writer.Write(compound, output_stl.string().c_str()))
        throw std::runtime_error("Open CASCADE failed to write STL: " + output_stl.string());
    return stats;
}

} // namespace cvcascade
