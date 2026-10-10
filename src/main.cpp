#include "cvcascade/mesh_builder.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef CVCASCADE_HAS_OPENMP
#include <omp.h>
#endif
namespace {
void usage(std::ostream& out) {
    out << "cvcascade - image-to-3D relief mesh\n"
        << "Usage: cvcascade --input photo.jpg --output relief.stl [options]\n"
        << "  --width MM          Physical X width in mm (default 100)\n"
        << "  --max-height MM     Maximum relief height in mm (default 20)\n"
        << "  --max-dimension N   Maximum image dimension (default 512)\n"
        << "  --invert            Dark pixels become high points\n"
        << "  --threads N         OpenMP thread count, when available\n"
        << "  --help              Show this help\n"
        << "A single photo cannot reveal hidden geometry; output is a brightness-driven relief.\n";
}
}
int main(int argc, char** argv) {
    try {
        std::string input, output;
        cvcascade::MeshOptions options;
        int threads = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(std::cout); return 0; }
            auto value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--input") input = value("--input");
            else if (arg == "--output") output = value("--output");
            else if (arg == "--width") options.width = std::stod(value("--width"));
            else if (arg == "--max-height") options.max_height = std::stod(value("--max-height"));
            else if (arg == "--max-dimension") options.max_dimension = std::stoi(value("--max-dimension"));
            else if (arg == "--threads") threads = std::stoi(value("--threads"));
            else if (arg == "--invert") options.invert = true;
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (input.empty() || output.empty()) { usage(std::cerr); return 2; }
        if (threads < 0) throw std::invalid_argument("--threads must be zero (auto) or positive");
#ifdef CVCASCADE_HAS_OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif
        const cv::Mat image = cv::imread(input, cv::IMREAD_GRAYSCALE);
        if (image.empty()) throw std::runtime_error("cannot read input image: " + input);
        const cv::Mat heights = cvcascade::make_height_map(image, options);
        const auto stats = cvcascade::write_stl(heights, options, output);
        std::cout << "Wrote " << output << "\nResolution: " << stats.cols << " x " << stats.rows
                  << "\nTriangles: " << stats.triangles << "\nHeight range: " << stats.min_height
                  << " .. " << stats.max_height << " mm\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
