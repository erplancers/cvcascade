#pragma once

#include <wx/wx.h>
#include <wx/dcbuffer.h>
#include <wx/filedlg.h>
#include <wx/filename.h>
#include <wx/spinctrl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class StlViewerPanel final : public wxPanel {
public:
    struct Triangle { std::array<std::array<double, 3>, 3> p{}; };

    explicit StlViewerPanel(wxWindow* parent)
        : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxSize(520, 360),
                  wxBORDER_SIMPLE | wxWANTS_CHARS) {
        SetBackgroundStyle(wxBG_STYLE_PAINT);
        Bind(wxEVT_PAINT, &StlViewerPanel::OnPaint, this);
        Bind(wxEVT_LEFT_DOWN, &StlViewerPanel::OnMouseDown, this);
        Bind(wxEVT_LEFT_UP, &StlViewerPanel::OnMouseUp, this);
        Bind(wxEVT_MOTION, &StlViewerPanel::OnMouseMove, this);
        Bind(wxEVT_MOUSEWHEEL, &StlViewerPanel::OnMouseWheel, this);
        Bind(wxEVT_SIZE, [this](wxSizeEvent&) { Refresh(false); });
    }

    bool LoadStl(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) { error_ = "Cannot open STL: " + filename; Refresh(); return false; }
        triangles_.clear();
        in.seekg(0, std::ios::end);
        const auto bytes = in.tellg();
        in.seekg(0, std::ios::beg);
        bool binary = false;
        if (bytes >= 84) {
            std::uint32_t count = 0;
            in.seekg(80);
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            binary = in.good() && static_cast<std::uint64_t>(bytes) == 84ULL + 50ULL * count;
        }
        in.clear();
        in.seekg(0, std::ios::beg);
        if (binary) {
            in.seekg(80);
            std::uint32_t count = 0;
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (count > 100000000U) { error_ = "STL triangle count is unreasonable."; return false; }
            triangles_.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                float normal[3], v[9]; std::uint16_t attribute = 0;
                in.read(reinterpret_cast<char*>(normal), sizeof(normal));
                in.read(reinterpret_cast<char*>(v), sizeof(v));
                in.read(reinterpret_cast<char*>(&attribute), sizeof(attribute));
                if (!in) { error_ = "Truncated binary STL."; triangles_.clear(); return false; }
                Triangle t;
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k) t.p[j][k] = v[j * 3 + k];
                triangles_.push_back(t);
            }
        } else {
            std::string word;
            std::array<double, 3> vertex{};
            int have = 0;
            while (in >> word) {
                if (word != "vertex") continue;
                if (!(in >> vertex[0] >> vertex[1] >> vertex[2])) {
                    error_ = "Malformed ASCII STL vertex."; triangles_.clear(); return false;
                }
                if (have == 0) pending_.p[0] = vertex;
                else if (have == 1) pending_.p[1] = vertex;
                else {
                    pending_.p[2] = vertex;
                    triangles_.push_back(pending_);
                }
                have = (have + 1) % 3;
            }
            if (have != 0) { error_ = "ASCII STL ends with an incomplete triangle."; triangles_.clear(); return false; }
        }
        if (triangles_.empty()) { error_ = "No triangles found in STL."; Refresh(); return false; }
        filename_ = filename;
        error_.clear();
        ComputeBounds();
        Refresh();
        return true;
    }

    std::string ValidationSummary() const {
        if (triangles_.empty()) return "No STL loaded.";
        using PointKey = std::array<long long, 3>;
        using EdgeKey = std::pair<PointKey, PointKey>;
        std::map<EdgeKey, int> edges;
        std::size_t degenerate = 0;
        auto key = [](const std::array<double, 3>& p) {
            return PointKey{std::llround(p[0] * 1000000.0), std::llround(p[1] * 1000000.0),
                            std::llround(p[2] * 1000000.0)};
        };
        for (const auto& t : triangles_) {
            const auto a = key(t.p[0]), b = key(t.p[1]), c = key(t.p[2]);
            const auto ux=t.p[1][0]-t.p[0][0], uy=t.p[1][1]-t.p[0][1], uz=t.p[1][2]-t.p[0][2];
            const auto vx=t.p[2][0]-t.p[0][0], vy=t.p[2][1]-t.p[0][1], vz=t.p[2][2]-t.p[0][2];
            const double cx=uy*vz-uz*vy, cy=uz*vx-ux*vz, cz=ux*vy-uy*vx;
            if (cx*cx+cy*cy+cz*cz < 1e-20) ++degenerate;
            auto add = [&](PointKey x, PointKey y) { if (y < x) std::swap(x,y); ++edges[{x,y}]; };
            add(a,b); add(b,c); add(c,a);
        }
        std::size_t boundary=0, nonmanifold=0;
        for (const auto& e : edges) {
            if (e.second == 1) ++boundary;
            else if (e.second > 2) ++nonmanifold;
        }
        std::ostringstream out;
        out << "File: " << filename_ << "\nTriangles: " << triangles_.size()
            << "\nBoundary edges: " << boundary << "\nNon-manifold edges: " << nonmanifold
            << "\nDegenerate triangles: " << degenerate
            << "\nBounds X: " << min_[0] << " .. " << max_[0]
            << "\nBounds Y: " << min_[1] << " .. " << max_[1]
            << "\nBounds Z: " << min_[2] << " .. " << max_[2]
            << "\nWatertight heuristic: " << ((boundary == 0 && nonmanifold == 0 && degenerate == 0) ? "yes" : "no");
        return out.str();
    }

    bool ExportPly(const std::string& filename) const {
        if (triangles_.empty()) return false;
        std::ofstream out(filename, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << "ply\nformat ascii 1.0\nelement vertex " << triangles_.size() * 3
            << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
            << triangles_.size() << "\nproperty list uchar int vertex_indices\nend_header\n";
        out << std::setprecision(12);
        for (const auto& t : triangles_) for (const auto& p : t.p)
            out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        for (std::size_t i = 0; i < triangles_.size(); ++i)
            out << "3 " << i*3 << ' ' << i*3+1 << ' ' << i*3+2 << '\n';
        return static_cast<bool>(out);
    }

    const std::string& Filename() const { return filename_; }
    std::size_t TriangleCount() const { return triangles_.size(); }

private:
    std::vector<Triangle> triangles_;
    Triangle pending_{};
    std::string filename_;
    std::string error_;
    std::array<double,3> min_{}, max_{};
    double yaw_ = 0.65, pitch_ = -0.45, zoom_ = 1.0;
    wxPoint last_mouse_;
    bool dragging_ = false;

    void ComputeBounds() {
        min_.fill(std::numeric_limits<double>::max());
        max_.fill(std::numeric_limits<double>::lowest());
        for (const auto& t : triangles_) for (const auto& p : t.p)
            for (int k=0;k<3;++k) { min_[k]=std::min(min_[k],p[k]); max_[k]=std::max(max_[k],p[k]); }
    }

    wxPoint Project(const std::array<double,3>& p, int w, int h) const {
        const double cx=(min_[0]+max_[0])*0.5, cy=(min_[1]+max_[1])*0.5, cz=(min_[2]+max_[2])*0.5;
        const double scale=std::max({max_[0]-min_[0],max_[1]-min_[1],max_[2]-min_[2],1e-9});
        double x=(p[0]-cx)/scale, y=(p[1]-cy)/scale, z=(p[2]-cz)/scale;
        const double x1=std::cos(yaw_)*x-std::sin(yaw_)*y;
        const double y1=std::sin(yaw_)*x+std::cos(yaw_)*y;
        const double y2=std::cos(pitch_)*y1-std::sin(pitch_)*z;
        const double margin=24.0, factor=std::min(w,h)*0.78*zoom_;
        return wxPoint(static_cast<int>(w*0.5+factor*x1),static_cast<int>(h*0.5-factor*y2));
    }

    void OnPaint(wxPaintEvent&) {
        wxAutoBufferedPaintDC dc(this);
        const auto size=GetClientSize();
        dc.SetBackground(wxBrush(wxColour(23,29,38))); dc.Clear();
        if (!error_.empty()) { dc.SetTextForeground(wxColour(255,180,150)); dc.DrawText(error_,10,10); return; }
        if (triangles_.empty()) { dc.SetTextForeground(*wxWHITE); dc.DrawText("Open an STL to inspect the mesh. Drag to rotate; wheel to zoom.",10,10); return; }
        dc.SetPen(wxPen(wxColour(110,190,220),1));
        for (const auto& t : triangles_) {
            wxPoint a=Project(t.p[0],size.x,size.y), b=Project(t.p[1],size.x,size.y), c=Project(t.p[2],size.x,size.y);
            dc.DrawLine(a,b); dc.DrawLine(b,c); dc.DrawLine(c,a);
        }
        dc.SetTextForeground(wxColour(220,230,240));
        dc.DrawText(wxString::Format("%zu triangles  |  zoom %.2fx",triangles_.size(),zoom_),10,10);
    }
    void OnMouseDown(wxMouseEvent& e) { dragging_=true; last_mouse_=e.GetPosition(); CaptureMouse(); }
    void OnMouseUp(wxMouseEvent&) { dragging_=false; if (HasCapture()) ReleaseMouse(); }
    void OnMouseMove(wxMouseEvent& e) {
        if (!dragging_ || !e.Dragging()) return;
        const auto p=e.GetPosition(); yaw_+=(p.x-last_mouse_.x)*0.012; pitch_+=(p.y-last_mouse_.y)*0.012;
        last_mouse_=p; Refresh(false);
    }
    void OnMouseWheel(wxMouseEvent& e) { zoom_=std::clamp(zoom_*(e.GetWheelRotation()>0?1.12:0.89),0.1,12.0); Refresh(false); }
};
