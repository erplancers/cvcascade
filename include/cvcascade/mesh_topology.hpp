#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace cvcascade {

using MeshTriangle = std::array<std::size_t, 3>;

struct MeshTopologyReport {
    std::size_t vertices = 0;
    std::size_t faces = 0;
    std::size_t degenerate_faces = 0;
    std::size_t duplicate_faces = 0;
    std::size_t boundary_edges = 0;
    std::size_t non_manifold_edges = 0;
    std::size_t connected_components = 0;
    bool watertight() const noexcept {
        return faces > 0 && boundary_edges == 0 && non_manifold_edges == 0 &&
               degenerate_faces == 0 && duplicate_faces == 0;
    }
};

MeshTopologyReport analyze_mesh_topology(
    const std::vector<cv::Point3d>& vertices,
    const std::vector<MeshTriangle>& triangles,
    double area_epsilon = 1e-12);

// Removes invalid-index, zero-area, and duplicate faces, then compacts unused vertices.
// It deliberately does not invent geometry to fill holes or guarantee watertightness.
MeshTopologyReport repair_mesh_topology(
    std::vector<cv::Point3d>& vertices,
    std::vector<MeshTriangle>& triangles,
    double area_epsilon = 1e-12);

} // namespace cvcascade
