#include "cvcascade/mesh_builder.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <stdexcept>
namespace { void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); } }
int main() {
    try {
        cv::Mat image = (cv::Mat_<unsigned char>(2, 3) << 0, 127, 255, 255, 127, 0);
        cvcascade::MeshOptions options; options.max_height = 10.0;
        const cv::Mat heights = cvcascade::make_height_map(image, options);
        require(heights.type() == CV_32F, "height map must be float");
        require(heights.rows == 2 && heights.cols == 3, "small image dimensions must be preserved");
        require(std::abs(heights.at<float>(0, 0)) < 1e-6, "black maps to zero");
        require(std::abs(heights.at<float>(0, 2) - 10.0F) < 1e-5, "white maps to max height");
        options.invert = true;
        const cv::Mat inverted = cvcascade::make_height_map(image, options);
        require(std::abs(inverted.at<float>(0, 0) - 10.0F) < 1e-5, "invert maps black to max");
        require(std::abs(inverted.at<float>(0, 2)) < 1e-5, "invert maps white to zero");
        bool rejected = false;
        try { options.width = -1; (void)cvcascade::make_height_map(image, options); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "negative width must be rejected");
        cv::Mat large(80, 160, CV_8U, cv::Scalar(128));
        options.width = 100; options.max_dimension = 32; options.invert = false;
        const cv::Mat reduced = cvcascade::make_height_map(large, options);
        require(std::max(reduced.rows, reduced.cols) <= 32, "large images must be downsampled");

        const auto stl_path = std::filesystem::temp_directory_path() / "cvcascade_mesh_builder_test.stl";
        std::filesystem::remove(stl_path);
        const auto mesh_stats = cvcascade::write_stl(heights, options, stl_path.string());
        require(mesh_stats.triangles == 4, "2x3 height map should create four triangles");
        require(std::filesystem::exists(stl_path) && std::filesystem::file_size(stl_path) > 0,
                "Open CASCADE should write a non-empty STL file");
        std::filesystem::remove(stl_path);

        std::cout << "All mesh builder tests passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "TEST FAILURE: " << e.what() << "\n"; return 1; }
}
