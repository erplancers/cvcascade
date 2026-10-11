#include "cvcascade/mesh_builder.hpp"
#include "cvcascade/photogrammetry.hpp"

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#ifdef CVCASCADE_HAS_OPENMP
#include <omp.h>
#endif

namespace {
std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) escaped += '?';
                else escaped += static_cast<char>(ch);
        }
    }
    return escaped;
}

void write_reconstruction_report(const std::string& path,
                                 const cvcascade::ReconstructionStats& stats,
                                 const std::string& mesh_path,
                                 const std::string& cloud_path) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write reconstruction report: " + path);
    out << std::setprecision(10)
        << "{\n"
        << "  \"mesh\": \"" << json_escape(mesh_path) << "\",\n"
        << "  \"point_cloud\": \"" << json_escape(cloud_path) << "\",\n"
        << "  \"images\": " << stats.images << ",\n"
        << "  \"registered_pairs\": " << stats.registered_pairs << ",\n"
        << "  \"feature_matches\": " << stats.feature_matches << ",\n"
        << "  \"pose_inliers\": " << stats.pose_inliers << ",\n"
        << "  \"mean_pose_inlier_ratio\": " << stats.mean_pose_inlier_ratio << ",\n"
        << "  \"mean_valid_depth_ratio\": " << stats.mean_valid_depth_ratio << ",\n"
        << "  \"stereo_candidates\": " << stats.stereo_candidates << ",\n"
        << "  \"stereo_consistent\": " << stats.stereo_consistent << ",\n"
        << "  \"stereo_rejected\": " << stats.stereo_rejected << ",\n"
        << "  \"vertices\": " << stats.vertices << ",\n"
        << "  \"triangles\": " << stats.triangles << ",\n"
        << "  \"mesh_boundary_edges\": " << stats.mesh_boundary_edges << ",\n"
        << "  \"mesh_non_manifold_edges\": " << stats.mesh_non_manifold_edges << ",\n"
        << "  \"mesh_degenerate_faces_removed\": " << stats.mesh_degenerate_faces_removed << ",\n"
        << "  \"mesh_duplicate_faces_removed\": " << stats.mesh_duplicate_faces_removed << ",\n"
        << "  \"fused_points\": " << stats.fused_points << ",\n"
        << "  \"metric_scale\": " << (stats.metric_scale ? "true" : "false") << ",\n"
        << "  \"applied_scale_factor\": " << stats.applied_scale_factor << "\n"
        << "}\n";
    if (!out) throw std::runtime_error("failed while writing reconstruction report: " + path);
}

void usage(std::ostream& out) {
    out << "cvcascade - OpenCV + Open CASCADE image reconstruction\n"
        << "\nRelief from one photograph:\n"
        << "  cvcascade --input photo.jpg --output relief.stl [--width MM] [--max-height MM]\n"
        << "            [--max-dimension N] [--invert] [--threads N]\n"
        << "\nCamera calibration from chessboard views:\n"
        << "  cvcascade --calibrate --images calibration-images.txt --calibration camera.yml\n"
        << "            --board-cols N --board-rows N --square-size-mm MM\n"
        << "\nMulti-view reconstruction:\n"
        << "  cvcascade --reconstruct --images views.txt --calibration camera.yml --output model.stl\n"
        << "            [--point-cloud cloud.ply] [--report report.json] [--fusion-voxel N] [--min-fusion-observations N] [--lr-tolerance PX]\n"
        << "            [--scale-reference-mm MM --scale-reference-units UNITS]\n"
        << "            [--speckle-window N] [--speckle-range N] [--max-dimension N]\n"
        << "            [--num-disparities N] [--block-size N] [--min-inliers N]\n"
        << "            [--baseline-mm MM] [--max-depth N] [--depth-jump RATIO]\n"
        << "\nMulti-view uses ORB/RANSAC poses, left-right checked SGBM, speckle filtering and depth-edge rejection.\n"
        << "--point-cloud exports a voxel-fused RGB PLY in addition to the STL mesh.\n"
        << "Without --baseline-mm, model units and fusion voxel units are arbitrary.\n";
}
}

int main(int argc, char** argv) {
    try {
        std::string input, output, image_list, calibration_path, point_cloud_path, report_path;
        cvcascade::MeshOptions mesh_options;
        cvcascade::PhotogrammetryOptions photogrammetry_options;
        bool calibrate = false, reconstruct = false;
        int threads = 0;
        int board_cols = 9, board_rows = 6;
        double square_size_mm = 25.0;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(std::cout);
                return 0;
            }
            auto value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++i];
            };

            if (arg == "--input") input = value("--input");
            else if (arg == "--output") output = value("--output");
            else if (arg == "--images") image_list = value("--images");
            else if (arg == "--calibration") calibration_path = value("--calibration");
            else if (arg == "--point-cloud") point_cloud_path = value("--point-cloud");
            else if (arg == "--report") report_path = value("--report");
            else if (arg == "--calibrate") calibrate = true;
            else if (arg == "--reconstruct") reconstruct = true;
            else if (arg == "--width") mesh_options.width = std::stod(value("--width"));
            else if (arg == "--max-height") mesh_options.max_height = std::stod(value("--max-height"));
            else if (arg == "--max-dimension") {
                const int dimension = std::stoi(value("--max-dimension"));
                mesh_options.max_dimension = dimension;
                photogrammetry_options.max_dimension = dimension;
            }
            else if (arg == "--invert") mesh_options.invert = true;
            else if (arg == "--threads") threads = std::stoi(value("--threads"));
            else if (arg == "--board-cols") board_cols = std::stoi(value("--board-cols"));
            else if (arg == "--board-rows") board_rows = std::stoi(value("--board-rows"));
            else if (arg == "--square-size-mm") square_size_mm = std::stod(value("--square-size-mm"));
            else if (arg == "--num-disparities") photogrammetry_options.num_disparities = std::stoi(value("--num-disparities"));
            else if (arg == "--block-size") photogrammetry_options.block_size = std::stoi(value("--block-size"));
            else if (arg == "--min-inliers") photogrammetry_options.min_inliers = std::stoi(value("--min-inliers"));
            else if (arg == "--baseline-mm") photogrammetry_options.baseline_mm = std::stod(value("--baseline-mm"));
            else if (arg == "--scale-reference-mm") photogrammetry_options.scale_reference_mm = std::stod(value("--scale-reference-mm"));
            else if (arg == "--scale-reference-units") photogrammetry_options.scale_reference_units = std::stod(value("--scale-reference-units"));
            else if (arg == "--max-depth") photogrammetry_options.max_depth = std::stod(value("--max-depth"));
            else if (arg == "--depth-jump") photogrammetry_options.max_relative_depth_jump = std::stod(value("--depth-jump"));
            else if (arg == "--lr-tolerance") photogrammetry_options.left_right_tolerance = std::stod(value("--lr-tolerance"));
            else if (arg == "--fusion-voxel") photogrammetry_options.fusion_voxel_size = std::stod(value("--fusion-voxel"));
            else if (arg == "--min-fusion-observations") photogrammetry_options.min_fusion_observations = static_cast<std::size_t>(std::stoul(value("--min-fusion-observations")));
            else if (arg == "--speckle-window") photogrammetry_options.speckle_window_size = std::stoi(value("--speckle-window"));
            else if (arg == "--speckle-range") photogrammetry_options.speckle_range = std::stoi(value("--speckle-range"));
            else throw std::invalid_argument("unknown argument: " + arg);
        }

        if (calibrate && reconstruct)
            throw std::invalid_argument("--calibrate and --reconstruct are mutually exclusive");
        if (threads < 0) throw std::invalid_argument("--threads must be zero (auto) or positive");
#ifdef CVCASCADE_HAS_OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif

        if (calibrate) {
            if (image_list.empty() || calibration_path.empty())
                throw std::invalid_argument("--calibrate requires --images and --calibration");
            const auto camera = cvcascade::calibrate_camera_from_list(
                image_list, calibration_path, board_cols, board_rows, square_size_mm);
            std::cout << "Wrote calibration: " << calibration_path
                      << "\nImage size: " << camera.image_size.width << " x " << camera.image_size.height
                      << "\nReprojection RMS: " << camera.reprojection_rms << " px"
                      << "\nCalibration views used: " << camera.calibration_views_used
                      << "\nCalibration views rejected as outliers: " << camera.calibration_views_rejected
                      << "\n";
            return 0;
        }

        if (reconstruct) {
            if (image_list.empty() || calibration_path.empty() || output.empty())
                throw std::invalid_argument("--reconstruct requires --images, --calibration and --output");
            const auto stats = cvcascade::reconstruct_multiview(
                image_list, calibration_path, output, photogrammetry_options, point_cloud_path);
            const double consistent_percent = stats.stereo_candidates > 0
                ? 100.0 * static_cast<double>(stats.stereo_consistent) / stats.stereo_candidates : 0.0;
            std::cout << "Wrote mesh: " << output
                      << "\nImages: " << stats.images
                      << "\nDense stereo pairs with triangles: " << stats.registered_pairs
                      << "\nRatio-filtered feature matches: " << stats.feature_matches
                      << "\nPose inliers: " << stats.pose_inliers
                      << "\nMean pose inlier ratio: " << stats.mean_pose_inlier_ratio
                      << "\nMean valid depth ratio: " << stats.mean_valid_depth_ratio
                      << "\nStereo candidates: " << stats.stereo_candidates
                      << "\nLeft-right consistent pixels: " << stats.stereo_consistent
                      << "\nLeft-right consistency: " << consistent_percent << "%"
                      << "\nRejected by stereo consistency: " << stats.stereo_rejected
                      << "\nMesh vertices: " << stats.vertices
                      << "\nMesh triangles: " << stats.triangles
                      << "\nMesh boundary edges (per reconstructed patch): " << stats.mesh_boundary_edges
                      << "\nMesh non-manifold edges (per reconstructed patch): " << stats.mesh_non_manifold_edges
                      << "\nDegenerate faces rejected: " << stats.mesh_degenerate_faces_removed
                      << "\nDuplicate faces rejected: " << stats.mesh_duplicate_faces_removed
                      << "\nScale: " << (stats.metric_scale ? "metric scale requested (validate against a known dimension)" : "arbitrary monocular scale")
                      << "\nApplied scale factor: " << stats.applied_scale_factor
                      << "\n";
            if (!point_cloud_path.empty())
                std::cout << "Wrote fused RGB point cloud: " << point_cloud_path
                          << "\nFused points: " << stats.fused_points << "\n";
            if (!report_path.empty()) {
                write_reconstruction_report(report_path, stats, output, point_cloud_path);
                std::cout << "Wrote reconstruction report: " << report_path << "\n";
            }
            return 0;
        }

        if (input.empty() || output.empty()) {
            usage(std::cerr);
            return 2;
        }
        const cv::Mat image = cv::imread(input, cv::IMREAD_GRAYSCALE);
        if (image.empty()) throw std::runtime_error("cannot read input image: " + input);
        const cv::Mat heights = cvcascade::make_height_map(image, mesh_options);
        const auto stats = cvcascade::write_stl(heights, mesh_options, output);
        std::cout << "Wrote " << output << "\nResolution: " << stats.cols << " x " << stats.rows
                  << "\nTriangles: " << stats.triangles << "\nHeight range: " << stats.min_height
                  << " .. " << stats.max_height << " mm\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
