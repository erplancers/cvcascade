#include "cvcascade/photogrammetry.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
double nearest_distance(const cvcascade::ColoredPoint& p,
                        const std::vector<cvcascade::ColoredPoint>& cloud) {
    double best = 1e100;
    for (const auto& q : cloud) {
        const double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
        best = std::min(best, std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    return best;
}
}

int main() {
    try {
        // Deterministic synthetic planar target with two noisy observations per
        // reference point, plus isolated outliers that should fail support filtering.
        std::vector<cvcascade::ColoredPoint> observations;
        std::vector<cvcascade::ColoredPoint> reference;
        for (int y = 0; y < 9; ++y) {
            for (int x = 0; x < 9; ++x) {
                const cvcascade::ColoredPoint truth{double(x), double(y), 0.0, 180, 180, 180};
                reference.push_back(truth);
                observations.push_back({x + 0.01, y - 0.01, 0.015, 180, 180, 180});
                observations.push_back({x - 0.01, y + 0.01, -0.015, 180, 180, 180});
            }
        }
        observations.push_back({50.0, 50.0, 50.0, 255, 0, 0});
        observations.push_back({-40.0, 20.0, 70.0, 0, 255, 0});

        const auto fused = cvcascade::fuse_point_cloud_voxels(observations, 0.25, 2);
        require(fused.size() == reference.size(),
                "minimum-support fusion must remove isolated one-observation outliers");

        double squared = 0.0;
        for (const auto& truth : reference) {
            const double d = nearest_distance(truth, fused);
            squared += d * d;
        }
        const double rmse = std::sqrt(squared / reference.size());
        double min_x = 1e100, max_x = -1e100, min_y = 1e100, max_y = -1e100;
        for (const auto& p : fused) {
            min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        }
        const double dimension_error = std::max(std::abs((max_x - min_x) - 8.0),
                                                std::abs((max_y - min_y) - 8.0));
        std::cout << "Synthetic fusion benchmark\n"
                  << "reference_points=" << reference.size() << "\n"
                  << "fused_points=" << fused.size() << "\n"
                  << "chamfer_one_way_rmse=" << rmse << "\n"
                  << "max_dimension_error=" << dimension_error << "\n";
        require(rmse < 0.05, "synthetic point-cloud RMSE exceeded 0.05 model units");
        require(dimension_error < 0.05, "synthetic target dimension error exceeded 0.05 units");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "BENCHMARK FAILURE: " << e.what() << "\n";
        return 1;
    }
}
