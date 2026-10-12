#include "cvcascade/mesh_viewer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}
int main() {
    const auto dir = std::filesystem::temp_directory_path() / "cvcascade-mesh-viewer-tests";
    std::filesystem::create_directories(dir);
    const auto ascii = dir / "square-ascii.stl";
    {
        std::ofstream out(ascii);
        out << "solid square\n"
            << "facet normal 0 0 1\nouter loop\n"
            << "vertex 0 0 0\nvertex 1 0 0\nvertex 1 1 0\n"
            << "endloop\nendfacet\n"
            << "facet normal 0 0 1\nouter loop\n"
            << "vertex 0 0 0\nvertex 1 1 0\nvertex 0 1 0\n"
            << "endloop\nendfacet\nendsolid square\n";
    }
    const auto mesh = cvcascade::inspect_stl_mesh(ascii);
    require(mesh.vertices.size() == 4, "ASCII STL vertices should be deduplicated");
    require(mesh.faces.size() == 2, "ASCII STL should contain two triangles");
    require(mesh.topology.boundary_edges == 4, "square patch should have four boundary edges");
    require(mesh.topology.connected_components == 1, "square patch should have one component");
    require(!mesh.topology.watertight(), "open square patch must not be reported watertight");

    const auto ply = dir / "square.ply";
    cvcascade::export_mesh_ply(mesh, ply);
    std::ifstream ply_in(ply);
    std::string contents((std::istreambuf_iterator<char>(ply_in)), std::istreambuf_iterator<char>());
    require(contents.find("element vertex 4") != std::string::npos, "PLY export should preserve indexed vertices");
    require(contents.find("element face 2") != std::string::npos, "PLY export should preserve triangle faces");

    const auto binary = dir / "single-binary.stl";
    {
        std::ofstream out(binary, std::ios::binary);
        char header[80]{};
        out.write(header, sizeof(header));
        const std::uint32_t count = 1;
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        const float values[12] = {0,0,1, 0,0,0, 1,0,0, 0,1,0};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
        const std::uint16_t attributes = 0;
        out.write(reinterpret_cast<const char*>(&attributes), sizeof(attributes));
    }
    const auto binary_mesh = cvcascade::inspect_stl_mesh(binary);
    require(binary_mesh.faces.size() == 1, "binary STL should parse one triangle");
    require(binary_mesh.vertices.size() == 3, "binary STL vertices should be indexed");
    std::filesystem::remove_all(dir);
    std::cout << "Mesh viewer parser/export tests passed\n";
    return 0;
}
