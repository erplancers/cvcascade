#include "cvcascade/mesh_topology.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace cvcascade {
namespace {
using Edge = std::pair<std::size_t, std::size_t>;

Edge edge_key(std::size_t a, std::size_t b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

double twice_area(const cv::Point3d& a, const cv::Point3d& b, const cv::Point3d& c) {
    const cv::Vec3d ab = b - a;
    const cv::Vec3d ac = c - a;
    return cv::norm(ab.cross(ac));
}

bool invalid_face(const std::vector<cv::Point3d>& vertices, const MeshTriangle& face,
                  double area_epsilon) {
    if (face[0] >= vertices.size() || face[1] >= vertices.size() || face[2] >= vertices.size())
        return true;
    if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) return true;
    const auto& a = vertices[face[0]];
    const auto& b = vertices[face[1]];
    const auto& c = vertices[face[2]];
    if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z) ||
        !std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.z) ||
        !std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) return true;
    return twice_area(a, b, c) <= area_epsilon;
}
}

MeshTopologyReport analyze_mesh_topology(const std::vector<cv::Point3d>& vertices,
                                         const std::vector<MeshTriangle>& triangles,
                                         double area_epsilon) {
    if (!(area_epsilon >= 0.0) || !std::isfinite(area_epsilon))
        throw std::invalid_argument("area epsilon must be finite and non-negative");
    MeshTopologyReport report;
    report.vertices = vertices.size();
    report.faces = triangles.size();
    std::map<Edge, std::size_t> edge_uses;
    std::set<std::array<std::size_t, 3>> unique_faces;
    std::vector<std::size_t> parent(vertices.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<bool> used(vertices.size(), false);
    auto root = [&](std::size_t value) {
        std::size_t r = value;
        while (parent[r] != r) r = parent[r];
        while (parent[value] != value) {
            const std::size_t next = parent[value];
            parent[value] = r;
            value = next;
        }
        return r;
    };
    for (const auto& face : triangles) {
        if (invalid_face(vertices, face, area_epsilon)) {
            ++report.degenerate_faces;
            continue;
        }
        auto canonical = face;
        std::sort(canonical.begin(), canonical.end());
        if (!unique_faces.insert(canonical).second) ++report.duplicate_faces;
        for (const auto vertex : face) used[vertex] = true;
        const Edge edges[] = {edge_key(face[0], face[1]), edge_key(face[1], face[2]),
                              edge_key(face[2], face[0])};
        for (const auto& edge : edges) ++edge_uses[edge];
        const std::size_t a = root(face[0]), b = root(face[1]), c = root(face[2]);
        parent[b] = a;
        parent[c] = a;
    }
    for (const auto& entry : edge_uses) {
        if (entry.second == 1) ++report.boundary_edges;
        else if (entry.second > 2) ++report.non_manifold_edges;
    }
    std::set<std::size_t> components;
    for (std::size_t i = 0; i < used.size(); ++i)
        if (used[i]) components.insert(root(i));
    report.connected_components = components.size();
    return report;
}

MeshTopologyReport repair_mesh_topology(std::vector<cv::Point3d>& vertices,
                                        std::vector<MeshTriangle>& triangles,
                                        double area_epsilon) {
    if (!(area_epsilon >= 0.0) || !std::isfinite(area_epsilon))
        throw std::invalid_argument("area epsilon must be finite and non-negative");
    std::set<std::array<std::size_t, 3>> seen;
    std::vector<MeshTriangle> cleaned;
    cleaned.reserve(triangles.size());
    for (const auto& face : triangles) {
        if (invalid_face(vertices, face, area_epsilon)) continue;
        auto canonical = face;
        std::sort(canonical.begin(), canonical.end());
        if (seen.insert(canonical).second) cleaned.push_back(face);
    }
    std::vector<bool> used(vertices.size(), false);
    for (const auto& face : cleaned) for (const auto index : face) used[index] = true;
    std::vector<std::size_t> remap(vertices.size(), static_cast<std::size_t>(-1));
    std::vector<cv::Point3d> compact;
    compact.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        if (!used[i]) continue;
        remap[i] = compact.size();
        compact.push_back(vertices[i]);
    }
    for (auto& face : cleaned) for (auto& index : face) index = remap[index];
    vertices = std::move(compact);
    triangles = std::move(cleaned);
    return analyze_mesh_topology(vertices, triangles, area_epsilon);
}
} // namespace cvcascade
