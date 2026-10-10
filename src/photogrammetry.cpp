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
#include <functional>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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

cv::Mat read_color_image(const std::filesystem::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot decode color image: " + path.string());
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
    cv::Mat rotation;
    cv::Mat translation;
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
    const cv::Mat& color_first,
    const CameraModel& camera,
    const RelativePose& relative_pose,
    const cv::Mat& world_to_camera_rotation,
    const cv::Mat& world_to_camera_translation,
    const PhotogrammetryOptions& options,
    BRep_Builder& builder,
    TopoDS_Compound& compound,
    std::size_t& vertex_count,
    std::vector<ColoredPoint>* cloud_samples,
    std::size_t& stereo_candidates,
    std::size_t& stereo_consistent,
    std::size_t& stereo_rejected,
    double& valid_depth_ratio) {

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

    cv::Mat rectified_color_first;
    if (cloud_samples != nullptr)
        cv::remap(color_first, rectified_color_first, map11, map12, cv::INTER_LINEAR);

    const int block = options.block_size;
    auto make_stereo = [&](int min_disparity) {
        return cv::StereoSGBM::create(
            min_disparity, options.num_disparities, block,
            8 * block * block, 32 * block * block,
            1, 31, 10, options.speckle_window_size, options.speckle_range,
            cv::StereoSGBM::MODE_SGBM_3WAY);
    };

    cv::Mat disparity_fixed, reverse_disparity_fixed, disparity, reverse_disparity, xyz;
    make_stereo(0)->compute(rectified_first, rectified_second, disparity_fixed);
    make_stereo(-options.num_disparities)->compute(
        rectified_second, rectified_first, reverse_disparity_fixed);

    if (options.speckle_window_size > 0) {
        cv::filterSpeckles(disparity_fixed, -16, options.speckle_window_size,
                           std::max(1, options.speckle_range) * 16);
        cv::filterSpeckles(reverse_disparity_fixed,
                           (-options.num_disparities - 1) * 16,
                           options.speckle_window_size,
                           std::max(1, options.speckle_range) * 16);
    }

    disparity_fixed.convertTo(disparity, CV_32F, 1.0 / 16.0);
    reverse_disparity_fixed.convertTo(reverse_disparity, CV_32F, 1.0 / 16.0);
    cv::reprojectImageTo3D(disparity, xyz, q, false, CV_32F);

    const int rows = xyz.rows;
    const int cols = xyz.cols;
    std::vector<Standard_Integer> node_index(static_cast<std::size_t>(rows) * cols, 0);
    std::vector<float> depth(static_cast<std::size_t>(rows) * cols, 0.0F);
    std::vector<gp_Pnt> vertices;
    vertices.reserve(static_cast<std::size_t>(rows) * cols / 2);
    std::size_t valid_depth_samples = 0;

    const cv::Mat rectified_to_camera = r1.t();
    const cv::Mat camera_to_world_rotation = world_to_camera_rotation.t();
    const cv::Mat camera_to_world_translation = -camera_to_world_rotation * world_to_camera_translation;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y) * cols + x;
            const float d = disparity.at<float>(y, x);
            if (!(d > 0.5F) || !std::isfinite(d)) continue;
            ++stereo_candidates;

            const int xr = cvRound(static_cast<double>(x) - d);
            if (xr < 0 || xr >= cols) {
                ++stereo_rejected;
                continue;
            }
            const float reverse_d = reverse_disparity.at<float>(y, xr);
            if (!std::isfinite(reverse_d) || !(reverse_d < -0.5F) ||
                std::abs(static_cast<double>(d) + reverse_d) > options.left_right_tolerance) {
                ++stereo_rejected;
                continue;
            }
            ++stereo_consistent;

            const cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
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
            ++valid_depth_samples;

            if (cloud_samples != nullptr) {
                const cv::Vec3b bgr = rectified_color_first.at<cv::Vec3b>(y, x);
                cloud_samples->push_back({wx, wy, wz, bgr[2], bgr[1], bgr[0]});
            }
        }
    }
    valid_depth_ratio = static_cast<double>(valid_depth_samples) /
        static_cast<double>(std::max<std::size_t>(1, static_cast<std::size_t>(rows) * cols));

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

struct VoxelKey {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;
    bool operator==(const VoxelKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& key) const noexcept {
        const auto h1 = std::hash<std::int64_t>{}(key.x);
        const auto h2 = std::hash<std::int64_t>{}(key.y);
        const auto h3 = std::hash<std::int64_t>{}(key.z);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U)) ^
               (h3 + 0x9e3779b97f4a7c15ULL + (h2 << 6U) + (h2 >> 2U));
    }
};

struct VoxelAccum {
    double x = 0.0, y = 0.0, z = 0.0;
    std::uint64_t red = 0, green = 0, blue = 0, count = 0;
};

} // namespace

std::vector<ColoredPoint> fuse_point_cloud_voxels(
    const std::vector<ColoredPoint>& points, double voxel_size) {
    if (!(voxel_size > 0.0) || !std::isfinite(voxel_size))
        throw std::invalid_argument("fusion voxel size must be finite and positive");

    std::unordered_map<VoxelKey, VoxelAccum, VoxelHash> voxels;
    voxels.reserve(points.size());
    const long double low = static_cast<long double>(std::numeric_limits<std::int64_t>::min()) + 1.0L;
    const long double high = static_cast<long double>(std::numeric_limits<std::int64_t>::max()) - 1.0L;

    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::invalid_argument("point cloud contains a non-finite coordinate");
        const long double vx = std::floor(static_cast<long double>(point.x) / voxel_size);
        const long double vy = std::floor(static_cast<long double>(point.y) / voxel_size);
        const long double vz = std::floor(static_cast<long double>(point.z) / voxel_size);
        if (vx < low || vx > high || vy < low || vy > high || vz < low || vz > high)
            throw std::out_of_range("point coordinates exceed voxel grid index range");

        auto& voxel = voxels[VoxelKey{static_cast<std::int64_t>(vx),
                                      static_cast<std::int64_t>(vy),
                                      static_cast<std::int64_t>(vz)}];
        voxel.x += point.x;
        voxel.y += point.y;
        voxel.z += point.z;
        voxel.red += point.red;
        voxel.green += point.green;
        voxel.blue += point.blue;
        ++voxel.count;
    }

    std::vector<ColoredPoint> fused;
    fused.reserve(voxels.size());
    for (const auto& entry : voxels) {
        const auto& voxel = entry.second;
        const double count = static_cast<double>(voxel.count);
        fused.push_back({
            voxel.x / count, voxel.y / count, voxel.z / count,
            static_cast<std::uint8_t>(std::lround(static_cast<double>(voxel.red) / count)),
            static_cast<std::uint8_t>(std::lround(static_cast<double>(voxel.green) / count)),
            static_cast<std::uint8_t>(std::lround(static_cast<double>(voxel.blue) / count))
        });
    }
    std::sort(fused.begin(), fused.end(), [](const ColoredPoint& a, const ColoredPoint& b) {
        return std::tie(a.x, a.y, a.z, a.red, a.green, a.blue) <
               std::tie(b.x, b.y, b.z, b.red, b.green, b.blue);
    });
    return fused;
}

void write_ply_point_cloud(
    const std::filesystem::path& output_path,
    const std::vector<ColoredPoint>& points) {
    if (output_path.empty()) throw std::invalid_argument("PLY output path must not be empty");
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::invalid_argument("cannot write a point cloud containing non-finite coordinates");
    }
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot write PLY point cloud: " + output_path.string());
    output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
           << "\nproperty double x\nproperty double y\nproperty double z"
           << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    output << std::setprecision(12);
    for (const auto& point : points) {
        output << point.x << ' ' << point.y << ' ' << point.z << ' '
               << static_cast<unsigned>(point.red) << ' '
               << static_cast<unsigned>(point.green) << ' '
               << static_cast<unsigned>(point.blue) << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing PLY point cloud: " + output_path.string());
}

std::vector<std::filesystem::path> read_image_list(const std::filesystem::path& list_path) {
    std::ifstream input(list_path);
    if (!input) throw std::runtime_error("cannot open image list: " + list_path.string());

    std::vector<std::filesystem::path> paths;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        std::filesystem::path path(line);
        if (path.is_relative()) {
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
    const PhotogrammetryOptions& options,
    const std::filesystem::path& output_ply) {

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
    if (!(options.left_right_tolerance > 0.0) || !std::isfinite(options.left_right_tolerance))
        throw std::invalid_argument("left-right tolerance must be finite and positive");
    if (!(options.fusion_voxel_size > 0.0) || !std::isfinite(options.fusion_voxel_size))
        throw std::invalid_argument("fusion voxel size must be finite and positive");
    if (options.speckle_window_size < 0 || options.speckle_range < 0)
        throw std::invalid_argument("speckle window and range must be non-negative");
    if (output_stl.empty()) throw std::invalid_argument("output STL path must not be empty");
    if (!output_ply.empty() && output_ply.lexically_normal() == output_stl.lexically_normal())
        throw std::invalid_argument("STL mesh and PLY point cloud must use different output paths");

    const auto paths = read_image_list(image_list);
    if (paths.size() < 2) throw std::runtime_error("multi-view reconstruction needs at least two images");

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> color_images;
    images.reserve(paths.size());
    color_images.reserve(paths.size());
    for (const auto& path : paths) {
        images.push_back(resize_image(read_gray_image(path), options.max_dimension));
        if (!output_ply.empty())
            color_images.push_back(resize_image(read_color_image(path), options.max_dimension));
    }
    for (std::size_t i = 1; i < images.size(); ++i) {
        if (images[i].size() != images[0].size())
            throw std::runtime_error("all reconstruction images must have the same size/aspect ratio");
        if (!color_images.empty() && color_images[i].size() != color_images[0].size())
            throw std::runtime_error("all reconstruction color images must have the same dimensions");
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
    double inlier_ratio_sum = 0.0;

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
        inlier_ratio_sum += static_cast<double>(pose.inliers) /
            static_cast<double>(std::max<std::size_t>(1, pose.good_matches));
        pair_poses.push_back(std::move(pose));
    }
    stats.mean_pose_inlier_ratio = inlier_ratio_sum / static_cast<double>(pair_poses.size());

    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    std::vector<ColoredPoint> cloud_samples;
    if (!output_ply.empty()) cloud_samples.reserve(
        static_cast<std::size_t>(images.size() - 1) * images[0].total() / 2);
    double valid_ratio_sum = 0.0;

    for (std::size_t i = 0; i < pair_poses.size(); ++i) {
        const cv::Mat& color_first = output_ply.empty() ? images[i] : color_images[i];
        double valid_ratio = 0.0;
        const std::size_t triangles = append_dense_stereo_mesh(
            images[i], images[i + 1], color_first, camera, pair_poses[i],
            camera_rotations[i], camera_translations[i], options,
            builder, compound, stats.vertices,
            output_ply.empty() ? nullptr : &cloud_samples,
            stats.stereo_candidates, stats.stereo_consistent, stats.stereo_rejected,
            valid_ratio);
        valid_ratio_sum += valid_ratio;
        stats.triangles += triangles;
        if (triangles > 0) ++stats.registered_pairs;
    }
    stats.mean_valid_depth_ratio = valid_ratio_sum / static_cast<double>(pair_poses.size());

    if (stats.triangles == 0)
        throw std::runtime_error("dense stereo produced no valid triangles; check overlap, lighting, texture and disparity range");

    StlAPI_Writer writer;
    if (!writer.Write(compound, output_stl.string().c_str()))
        throw std::runtime_error("Open CASCADE failed to write STL: " + output_stl.string());

    if (!output_ply.empty()) {
        const auto fused = fuse_point_cloud_voxels(cloud_samples, options.fusion_voxel_size);
        if (fused.empty()) throw std::runtime_error("no points remain after dense point-cloud fusion");
        write_ply_point_cloud(output_ply, fused);
        stats.fused_points = fused.size();
    }
    return stats;
}

} // namespace cvcascade
