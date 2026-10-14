#pragma once

#include <wx/panel.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace cvcascade::gui {

struct MeshInspection {
    std::size_t triangles = 0;
    std::size_t vertices = 0;
    std::size_t degenerate_triangles = 0;
    std::size_t non_finite_vertices = 0;
    double min_x = 0.0, min_y = 0.0, min_z = 0.0;
    double max_x = 0.0, max_y = 0.0, max_z = 0.0;
    std::uintmax_t file_bytes = 0;
    bool binary_stl = false;
    bool bounds_valid = false;
    std::string summary;
};

class MeshViewerPanel final : public wxPanel {
public:
    explicit MeshViewerPanel(wxWindow* parent);
    bool LoadStl(const std::filesystem::path& path, std::string& error);
    const MeshInspection& Inspection() const { return inspection_; }
    bool ExportInspectionJson(const std::filesystem::path& path, std::string& error) const;

private:
    using Triangle = std::array<std::array<double, 3>, 3>;
    std::vector<Triangle> triangles_;
    MeshInspection inspection_;
    double yaw_ = -0.55;
    double pitch_ = 0.45;
    double zoom_ = 1.0;
    wxPoint last_mouse_;
    bool dragging_ = false;

    void OnPaint(wxPaintEvent& event);
    void OnMouseDown(wxMouseEvent& event);
    void OnMouseMove(wxMouseEvent& event);
    void OnMouseUp(wxMouseEvent& event);
    void OnMouseWheel(wxMouseEvent& event);
};

} // namespace cvcascade::gui
