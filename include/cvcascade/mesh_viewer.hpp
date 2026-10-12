#pragma once

#include <wx/panel.h>
#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>
#include "cvcascade/mesh_topology.hpp"

namespace cvcascade {
struct MeshViewerTriangle {
    cv::Point3d a, b, c;
};
struct MeshInspection {
    std::vector<cv::Point3d> vertices;
    std::vector<MeshTriangle> faces;
    MeshTopologyReport topology;
};
MeshInspection inspect_stl_mesh(const std::filesystem::path& path);
void export_mesh_ply(const MeshInspection& mesh, const std::filesystem::path& path);

class MeshViewer final : public wxPanel {
public:
    explicit MeshViewer(wxWindow* parent);
    void SetMesh(std::vector<MeshViewerTriangle> triangles);
private:
    std::vector<MeshViewerTriangle> triangles_;
    double yaw_ = 0.65;
    double pitch_ = -0.45;
    double zoom_ = 1.0;
    wxPoint last_mouse_;
    bool dragging_ = false;
    void OnPaint(wxPaintEvent& event);
    void OnMouseDown(wxMouseEvent& event);
    void OnMouseMove(wxMouseEvent& event);
    void OnMouseUp(wxMouseEvent& event);
    void OnMouseWheel(wxMouseEvent& event);
};
}
