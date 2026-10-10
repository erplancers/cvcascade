#include "cvcascade/photogrammetry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <cmath>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Exception, typename Fn>
void require_throws(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "cvcascade_photogrammetry_tests";
    try {
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "images");
        std::ofstream(root / "images" / "first.jpg") << "test";
        std::ofstream(root / "images" / "second.jpg") << "test";

        const auto list = root / "views.txt";
        {
            std::ofstream out(list);
            out << "# comment line\n\nimages/first.jpg\n images/second.jpg \n";
        }
        const auto paths = cvcascade::read_image_list(list);
        require(paths.size() == 2, "image-list parser should ignore blanks and comments");
        require(paths[0] == (root / "images" / "first.jpg").lexically_normal(),
                "relative paths should resolve against the list directory");
        require(paths[1] == (root / "images" / "second.jpg").lexically_normal(),
                "whitespace around paths should be trimmed");

        const auto empty_list = root / "empty.txt";
        std::ofstream(empty_list) << "# no images here\n";
        require_throws<std::runtime_error>(
            [&] { (void)cvcascade::read_image_list(empty_list); },
            "empty image lists should be rejected");

        require_throws<std::invalid_argument>(
            [&] { (void)cvcascade::calibrate_camera_from_list(list, root / "camera.yml", 1, 6, 25.0); },
            "invalid chessboard dimensions should be rejected before image decoding");

        require_throws<std::runtime_error>(
            [&] { (void)cvcascade::load_camera_model(root / "missing-camera.yml"); },
            "missing camera calibration should be rejected");

        cvcascade::PhotogrammetryOptions bad_options;
        bad_options.num_disparities = 15;
        require_throws<std::invalid_argument>(
            [&] { (void)cvcascade::reconstruct_multiview(list, root / "missing-camera.yml",
                                                         root / "mesh.stl", bad_options); },
            "SGBM disparity count must be a multiple of 16");

        const std::vector<cvcascade::ColoredPoint> samples{
            {0.10, 0.10, 0.10, 255, 0, 0},
            {0.20, 0.20, 0.20, 0, 0, 255},
            {1.10, 0.10, 0.10, 0, 255, 0}
        };
        const auto fused = cvcascade::fuse_point_cloud_voxels(samples, 1.0);
        require(fused.size() == 2, "points in each voxel should be fused to one point");
        require(std::abs(fused[0].x - 0.15) < 1e-9, "voxel fusion should average point coordinates");
        require(fused[0].red == 128 && fused[0].green == 0 && fused[0].blue == 128,
                "voxel fusion should average RGB colors");
        require(fused[1].green == 255, "separate voxels should keep their own color");

        require_throws<std::invalid_argument>(
            [&] { (void)cvcascade::fuse_point_cloud_voxels(samples, 0.0); },
            "zero voxel size must be rejected");

        const auto ply_path = root / "fused.ply";
        cvcascade::write_ply_point_cloud(ply_path, fused);
        require(std::filesystem::exists(ply_path) && std::filesystem::file_size(ply_path) > 0,
                "PLY export should create a non-empty file");
        std::ifstream ply_file(ply_path);
        const std::string ply((std::istreambuf_iterator<char>(ply_file)), std::istreambuf_iterator<char>());
        require(ply.find("element vertex 2") != std::string::npos,
                "PLY header should report the fused point count");
        require(ply.find("property uchar red") != std::string::npos,
                "PLY should expose color properties");
        require(ply.find("128 0 128") != std::string::npos,
                "PLY should contain the fused point color");

        require_throws<std::invalid_argument>(
            [&] {
                const std::vector<cvcascade::ColoredPoint> bad{{NAN, 0.0, 0.0, 0, 0, 0}};
                (void)cvcascade::fuse_point_cloud_voxels(bad, 1.0);
            },
            "non-finite point coordinates must be rejected");

        std::filesystem::remove_all(root);
        std::cout << "All photogrammetry tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILURE: " << e.what() << "\n";
        std::filesystem::remove_all(root);
        return 1;
    }
}
