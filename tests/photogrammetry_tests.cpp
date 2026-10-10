#include "cvcascade/photogrammetry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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

        std::filesystem::remove_all(root);
        std::cout << "All photogrammetry tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILURE: " << e.what() << "\n";
        std::filesystem::remove_all(root);
        return 1;
    }
}
