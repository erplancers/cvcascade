#include "cvcascade/mesh_viewer.hpp"

#include <wx/dcbuffer.h>
#include <wx/dc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>

namespace cvcascade {
namespace {
float read_f32(std::istream& in) {
    std::array<unsigned char, 4> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), 4);
    if (!in) throw std::runtime_error("truncated binary STL");
    float value;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}
cv::Point3d read_binary_vertex(std::istream& in) {
    const double x = read_f32(in), y = read_f32(in), z = read_f32(in);
    return {x, y, z};
}
}

MeshInspection inspect_stl_mesh(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open mesh: " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    in.seekg(0);
    if (size < 0) throw std::runtime_error("Cannot determine mesh file size.");
    MeshInspection mesh;
    std::map<std::tuple<double, double, double>, std::size_t> index;
    auto add_vertex = [&](const cv::Point3d& p) {
        const auto key = std::make_tuple(p.x, p.y, p.z);
        auto found = index.find(key);
        if (found != index.end()) return found->second;
        const auto id = mesh.vertices.size();
        mesh.vertices.push_back(p);
        index.emplace(key, id);
        return id;
    };
    bool binary = false;
    if (size >= 84) {
        in.seekg(80);
        std::uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        binary = in && static_cast<std::uint64_t>(size) == 84ULL + 50ULL * count;
    }
    in.clear();
    in.seekg(0);
    if (binary) {
        in.seekg(80);
        std::uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (count > 5000000U) throw std::runtime_error("Mesh exceeds the 5-million-triangle inspection safety limit.");
        for (std::uint32_t i = 0; i < count; ++i) {
            (void)read_f32(in); (void)read_f32(in); (void)read_f32(in);
            cv::Point3d p[3] = {read_binary_vertex(in), read_binary_vertex(in), read_binary_vertex(in)};
            in.seekg(2, std::ios::cur);
            if (!in) throw std::runtime_error("Truncated binary STL triangle record.");
            mesh.faces.push_back({add_vertex(p[0]), add_vertex(p[1]), add_vertex(p[2])});
        }
    } else {
        std::string token;
        std::vector<cv::Point3d> triangle;
        while (in >> token) {
            if (token != "vertex") continue;
            cv::Point3d p;
            if (!(in >> p.x >> p.y >> p.z)) throw std::runtime_error("Invalid ASCII STL vertex record.");
            triangle.push_back(p);
            if (triangle.size() == 3) {
                mesh.faces.push_back({add_vertex(triangle[0]), add_vertex(triangle[1]), add_vertex(triangle[2])});
                triangle.clear();
                if (mesh.faces.size() > 5000000U) throw std::runtime_error("Mesh exceeds the 5-million-triangle inspection safety limit.");
            }
        }
        if (!triangle.empty()) throw std::runtime_error("ASCII STL ends with an incomplete triangle.");
    }
    if (mesh.faces.empty()) throw std::runtime_error("No triangles found; only STL meshes are supported by the inspection panel.");
    mesh.topology = analyze_mesh_topology(mesh.vertices, mesh.faces);
    return mesh;
}

void export_mesh_ply(const MeshInspection& mesh, const std::filesystem::path& path) {
    if (mesh.vertices.empty() || mesh.faces.empty()) throw std::runtime_error("No inspected mesh is loaded.");
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot create PLY export: " + path.string());
    out << "ply\nformat ascii 1.0\nelement vertex " << mesh.vertices.size()
        << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
        << mesh.faces.size() << "\nproperty list uchar uint vertex_indices\nend_header\n";
    out.precision(12);
    for (const auto& p : mesh.vertices) out << p.x << ' ' << p.y << ' ' << p.z << '\n';
    for (const auto& f : mesh.faces) out << "3 " << f[0] << ' ' << f[1] << ' ' << f[2] << '\n';
    if (!out) throw std::runtime_error("Failed while writing PLY export.");
}

MeshViewer::MeshViewer(wxWindow* parent) : wxPanel(parent, wxID_ANY) {
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    SetBackgroundColour(wxColour(24, 30, 39));
    Bind(wxEVT_PAINT, &MeshViewer::OnPaint, this);
    Bind(wxEVT_LEFT_DOWN, &MeshViewer::OnMouseDown, this);
    Bind(wxEVT_LEFT_UP, &MeshViewer::OnMouseUp, this);
    Bind(wxEVT_MOTION, &MeshViewer::OnMouseMove, this);
    Bind(wxEVT_MOUSEWHEEL, &MeshViewer::OnMouseWheel, this);
}
void MeshViewer::SetMesh(std::vector<MeshViewerTriangle> triangles) {
    triangles_ = std::move(triangles);
    yaw_ = 0.65; pitch_ = -0.45; zoom_ = 1.0;
    Refresh();
}
void MeshViewer::OnPaint(wxPaintEvent&) {
    wxAutoBufferedPaintDC dc(this);
    dc.SetBackground(wxBrush(wxColour(24, 30, 39)));
    dc.Clear();
    const wxSize size = GetClientSize();
    if (triangles_.empty()) {
        dc.SetTextForeground(wxColour(205, 215, 226));
        dc.DrawText("Load an STL mesh to inspect it. Drag to orbit; use the wheel to zoom.", 18, 18);
        return;
    }
    cv::Point3d center(0,0,0);
    double max_abs = 0.0;
    for (const auto& t : triangles_) for (const auto& p : {t.a,t.b,t.c}) {
        center += p;
        max_abs = std::max({max_abs,std::abs(p.x),std::abs(p.y),std::abs(p.z)});
    }
    center *= 1.0 / static_cast<double>(triangles_.size() * 3);
    const double scale = std::max(1e-9, max_abs > 0.0 ? max_abs : 1.0);
    struct DrawFace { std::array<wxPoint,3> p; double depth; int shade; };
    std::vector<DrawFace> draw;
    draw.reserve(triangles_.size());
    const double cy=std::cos(yaw_), sy=std::sin(yaw_), cp=std::cos(pitch_), sp=std::sin(pitch_);
    auto project = [&](const cv::Point3d& point, double& depth) {
        const double x=(point.x-center.x)/scale, y=(point.y-center.y)/scale, z=(point.z-center.z)/scale;
        const double rx=cy*x-sy*z, rz=sy*x+cy*z, ry=cp*y-sp*rz;
        depth=sp*y+cp*rz;
        const double fit=0.40*std::min(size.x,size.y)*zoom_;
        return wxPoint(static_cast<int>(size.x*0.5+rx*fit),static_cast<int>(size.y*0.5-ry*fit));
    };
    for (const auto& t : triangles_) {
        double da=0,db=0,dc=0;
        DrawFace f{{project(t.a,da),project(t.b,db),project(t.c,dc)},(da+db+dc)/3.0,100};
        const cv::Vec3d normal=(t.b-t.a).cross(t.c-t.a);
        const double n=cv::norm(normal);
        const double light=n>1e-12?std::abs(normal[2]/n):0.0;
        f.shade=static_cast<int>(65+150*light);
        draw.push_back(f);
    }
    std::sort(draw.begin(),draw.end(),[](const auto& a,const auto& b){return a.depth<b.depth;});
    dc.SetPen(wxPen(wxColour(12,17,24),1));
    for (const auto& f : draw) {
        dc.SetBrush(wxBrush(wxColour(f.shade, std::min(255,f.shade+12), std::min(255,f.shade+25))));
        dc.DrawPolygon(3,f.p.data());
    }
    dc.SetTextForeground(wxColour(205,215,226));
    dc.DrawText(wxString::Format("%zu triangles | drag: orbit | wheel: zoom",triangles_.size()),10,10);
}
void MeshViewer::OnMouseDown(wxMouseEvent& e) { dragging_=true; last_mouse_=e.GetPosition(); CaptureMouse(); }
void MeshViewer::OnMouseUp(wxMouseEvent&) { dragging_=false; if (HasCapture()) ReleaseMouse(); }
void MeshViewer::OnMouseMove(wxMouseEvent& e) {
    if (!dragging_ || !e.LeftIsDown()) return;
    const auto p=e.GetPosition();
    yaw_ += (p.x-last_mouse_.x)*0.01;
    pitch_ = std::clamp(pitch_+(p.y-last_mouse_.y)*0.01,-1.5,1.5);
    last_mouse_=p; Refresh(false);
}
void MeshViewer::OnMouseWheel(wxMouseEvent& e) {
    zoom_ *= e.GetWheelRotation()>0?1.12:0.89;
    zoom_=std::clamp(zoom_,0.1,12.0); Refresh(false);
}
}
