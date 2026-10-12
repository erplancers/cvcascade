#include "cvcascade/mesh_builder.hpp"
#include <BRep_Builder.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <StlAPI_Writer.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Pnt.hxx>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace cvcascade {
cv::Mat make_height_map(const cv::Mat& grayscale, const MeshOptions& options) {
    if (grayscale.empty()) throw std::invalid_argument("input image is empty");
    if (grayscale.channels() != 1) throw std::invalid_argument("height map requires a single-channel grayscale image");
    if (!(options.width > 0.0) || !(options.max_height >= 0.0) ||
        !std::isfinite(options.width) || !std::isfinite(options.max_height))
        throw std::invalid_argument("width must be positive and max-height must be finite and non-negative");
    if (options.max_dimension < 2) throw std::invalid_argument("max-dimension must be at least 2");
    cv::Mat resized;
    const int largest = std::max(grayscale.cols, grayscale.rows);
    if (largest > options.max_dimension) {
        const double scale = static_cast<double>(options.max_dimension) / largest;
        cv::resize(grayscale, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    } else resized = grayscale;
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    if (options.invert) normalized = 1.0F - normalized;
    normalized *= static_cast<float>(options.max_height);
    return normalized;
}
MeshStats write_stl(const cv::Mat& height_map, const MeshOptions& options, const std::string& output_path) {
    if (height_map.empty() || height_map.type() != CV_32F || height_map.rows < 2 || height_map.cols < 2)
        throw std::invalid_argument("height map must be a non-empty CV_32F matrix at least 2x2");
    if (!(options.width > 0.0) || !std::isfinite(options.width))
        throw std::invalid_argument("width must be finite and positive");
    if (output_path.empty()) throw std::invalid_argument("output path must not be empty");
    const int rows = height_map.rows, cols = height_map.cols;
    const double dx = options.width / static_cast<double>(cols - 1);
    const double dy = options.width * static_cast<double>(rows - 1) / static_cast<double>(cols - 1);
    auto triangulation = new Poly_Triangulation(rows * cols, 2 * (rows - 1) * (cols - 1), Standard_False);
    double min_h = std::numeric_limits<double>::max(), max_h = std::numeric_limits<double>::lowest();
    // Compute the range serially for consistent support across OpenMP implementations.
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const double h = static_cast<double>(height_map.at<float>(y, x));
            min_h = std::min(min_h, h);
            max_h = std::max(max_h, h);
        }
    }
    // Each iteration writes a unique triangulation node.
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) for (int x = 0; x < cols; ++x) {
        const double h = static_cast<double>(height_map.at<float>(y, x));
        const double px = x * dx;
        const double py = (rows - 1 - y) * dy / static_cast<double>(rows - 1);
        triangulation->SetNode(y * cols + x + 1, gp_Pnt(px, py, h));
    }
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows - 1; ++y) for (int x = 0; x < cols - 1; ++x) {
        const int a = y * cols + x + 1, b = a + 1, c = (y + 1) * cols + x + 1, d = c + 1;
        const int first = 2 * (y * (cols - 1) + x) + 1;
        triangulation->SetTriangle(first, Poly_Triangle(a, b, d));
        triangulation->SetTriangle(first + 1, Poly_Triangle(a, d, c));
    }
    TopoDS_Face face;
    BRep_Builder builder;
    builder.MakeFace(face, triangulation);
    StlAPI_Writer writer;
    if (!writer.Write(face, output_path.c_str())) throw std::runtime_error("Open CASCADE failed to write STL: " + output_path);
    return {rows, cols, static_cast<std::size_t>(2) * (rows - 1) * (cols - 1), min_h, max_h};
}
}
