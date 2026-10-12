#include "cvcascade/mesh_topology.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

int main() {
    try {
        std::vector<cv::Point3d> vertices{
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {9,9,9}
        };
        std::vector<cvcascade::MeshTriangle> faces{
            {0,1,2}, {0,2,3}, {0,1,2}, {0,0,1}, {0,1,99}
        };
        const auto before = cvcascade::analyze_mesh_topology(vertices, faces);
        require(before.duplicate_faces == 1, "duplicate faces must be counted");
        require(before.degenerate_faces == 2, "degenerate and invalid-index faces must be counted");
        require(before.boundary_edges == 4, "two-triangle square should have four boundary edges");
        const auto after = cvcascade::repair_mesh_topology(vertices, faces);
        require(faces.size() == 2, "repair must remove duplicate and invalid faces");
        require(vertices.size() == 4, "repair must compact unused vertices");
        require(after.boundary_edges == 4, "repair must not pretend open surfaces are watertight");
        require(!after.watertight(), "open square is not watertight");
        std::cout << "All mesh topology tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILURE: " << e.what() << "\n";
        return 1;
    }
}
