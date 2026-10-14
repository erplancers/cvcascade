#include "cvcascade/mesh_viewer.hpp"

#include <wx/dcbuffer.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cvcascade::gui {
namespace {

using Triangle = std::array<std::array<double, 3>, 3>;

double AreaSquared(const Triangle& t) {
    const double ax=t[1][0]-t[0][0], ay=t[1][1]-t[0][1], az=t[1][2]-t[0][2];
    const double bx=t[2][0]-t[0][0], by=t[2][1]-t[0][1], bz=t[2][2]-t[0][2];
    const double x=ay*bz-az*by, y=az*bx-ax*bz, z=ax*by-ay*bx;
    return x*x+y*y+z*z;
}

bool Finite(const std::array<double,3>& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

std::uint32_t ReadU32LE(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

float ReadF32LE(const unsigned char* p) {
    const std::uint32_t bits = ReadU32LE(p);
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

MeshViewerPanel::MeshViewerPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxSize(720, 480),
              wxBORDER_SIMPLE | wxFULL_REPAINT_ON_RESIZE) {
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    Bind(wxEVT_PAINT, &MeshViewerPanel::OnPaint, this);
    Bind(wxEVT_LEFT_DOWN, &MeshViewerPanel::OnMouseDown, this);
    Bind(wxEVT_LEFT_UP, &MeshViewerPanel::OnMouseUp, this);
    Bind(wxEVT_MOTION, &MeshViewerPanel::OnMouseMove, this);
    Bind(wxEVT_MOUSEWHEEL, &MeshViewerPanel::OnMouseWheel, this);
}

bool MeshViewerPanel::LoadStl(const std::filesystem::path& path, std::string& error) {
    triangles_.clear();
    inspection_ = {};
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "Cannot open STL file: " + path.string(); Refresh(); return false; }
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0) { error = "Cannot determine STL file size."; Refresh(); return false; }
    inspection_.file_bytes = static_cast<std::uintmax_t>(end);
    in.seekg(0, std::ios::beg);
    std::array<unsigned char,84> header{};
    in.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    bool binary = false;
    std::uint32_t count = 0;
    if (in.gcount() == static_cast<std::streamsize>(header.size())) {
        count = ReadU32LE(header.data()+80);
        const std::uintmax_t expected = 84ULL + static_cast<std::uintmax_t>(count)*50ULL;
        binary = expected == inspection_.file_bytes && count > 0;
    }
    in.clear();
    in.seekg(0, std::ios::beg);
    if (binary) {
        inspection_.binary_stl = true;
        in.seekg(84, std::ios::beg);
        triangles_.reserve(count);
        std::array<unsigned char,50> record{};
        for (std::uint32_t i=0; i<count; ++i) {
            in.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
            if (!in) { error = "Truncated binary STL triangle record."; triangles_.clear(); Refresh(); return false; }
            Triangle tri{};
            for (std::size_t v=0; v<3; ++v) for (std::size_t a=0; a<3; ++a)
                tri[v][a] = ReadF32LE(record.data()+12+(v*3+a)*4);
            triangles_.push_back(tri);
        }
    } else {
        inspection_.binary_stl = false;
        std::string line;
        std::vector<std::array<double,3>> vertices;
        while (std::getline(in,line)) {
            std::istringstream row(line);
            std::string word;
            row >> word;
            if (word != "vertex") continue;
            std::array<double,3> p{};
            if (!(row >> p[0] >> p[1] >> p[2])) {
                error = "Malformed ASCII STL vertex record."; triangles_.clear(); Refresh(); return false;
            }
            vertices.push_back(p);
            if (vertices.size() == 3) {
                triangles_.push_back({vertices[0],vertices[1],vertices[2]});
                vertices.clear();
            }
        }
        if (!vertices.empty()) { error = "ASCII STL ends with an incomplete triangle."; triangles_.clear(); Refresh(); return false; }
    }
    if (triangles_.empty()) { error = "No STL triangles were found."; Refresh(); return false; }

    double lo[3]={std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    double hi[3]={-lo[0],-lo[1],-lo[2]};
    for (const auto& tri : triangles_) {
        for (const auto& p : tri) {
            if (!Finite(p)) { ++inspection_.non_finite_vertices; continue; }
            for (std::size_t a=0;a<3;++a) { lo[a]=std::min(lo[a],p[a]); hi[a]=std::max(hi[a],p[a]); }
        }
        if (!Finite(tri[0]) || !Finite(tri[1]) || !Finite(tri[2]) || AreaSquared(tri)<1e-24)
            ++inspection_.degenerate_triangles;
    }
    inspection_.triangles=triangles_.size();
    inspection_.vertices=triangles_.size()*3;
    inspection_.bounds_valid=inspection_.non_finite_vertices==0;
    if (inspection_.bounds_valid) {
        inspection_.min_x=lo[0]; inspection_.min_y=lo[1]; inspection_.min_z=lo[2];
        inspection_.max_x=hi[0]; inspection_.max_y=hi[1]; inspection_.max_z=hi[2];
    }
    std::ostringstream summary;
    summary << (binary ? "Binary" : "ASCII") << " STL | " << inspection_.triangles << " triangles | "
            << inspection_.file_bytes << " bytes | degenerate: " << inspection_.degenerate_triangles
            << " | non-finite vertices: " << inspection_.non_finite_vertices;
    if (inspection_.bounds_valid)
        summary << std::setprecision(6) << "\nBounds X ["<<lo[0]<<", "<<hi[0]<<"] Y ["<<lo[1]<<", "<<hi[1]
                <<"] Z ["<<lo[2]<<", "<<hi[2]<<"]";
    inspection_.summary=summary.str();
    error.clear();
    Refresh();
    return true;
}

bool MeshViewerPanel::ExportInspectionJson(const std::filesystem::path& path, std::string& error) const {
    if (triangles_.empty()) { error="Load an STL mesh before exporting an inspection report."; return false; }
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) { error="Cannot create mesh inspection report: "+path.string(); return false; }
    fs << "schema_version" << 1
       << "format" << (inspection_.binary_stl ? "binary_stl" : "ascii_stl")
       << "file_bytes" << static_cast<double>(inspection_.file_bytes)
       << "triangles" << static_cast<double>(inspection_.triangles)
       << "vertices" << static_cast<double>(inspection_.vertices)
       << "degenerate_triangles" << static_cast<double>(inspection_.degenerate_triangles)
       << "non_finite_vertices" << static_cast<double>(inspection_.non_finite_vertices)
       << "bounds_valid" << static_cast<int>(inspection_.bounds_valid);
    if (inspection_.bounds_valid) {
        fs << "bounds" << "{"
           << "min_x" << inspection_.min_x << "min_y" << inspection_.min_y << "min_z" << inspection_.min_z
           << "max_x" << inspection_.max_x << "max_y" << inspection_.max_y << "max_z" << inspection_.max_z
           << "}";
    }
    fs.release();
    error.clear();
    return true;
}

void MeshViewerPanel::OnPaint(wxPaintEvent&) {
    wxAutoBufferedPaintDC dc(this);
    dc.SetBackground(wxBrush(wxColour(20,27,36)));
    dc.Clear();
    const wxSize size=GetClientSize();
    if (triangles_.empty()) {
        dc.SetTextForeground(wxColour(190,202,214));
        dc.DrawLabel("Load an STL mesh to inspect it. Drag to rotate; use the wheel to zoom.",
                     wxRect(10,10,size.x-20,size.y-20),wxALIGN_CENTER);
        return;
    }
    double center[3]={0,0,0}, span=0.0;
    if (inspection_.bounds_valid) {
        center[0]=(inspection_.min_x+inspection_.max_x)/2;
        center[1]=(inspection_.min_y+inspection_.max_y)/2;
        center[2]=(inspection_.min_z+inspection_.max_z)/2;
        span=std::max({inspection_.max_x-inspection_.min_x,inspection_.max_y-inspection_.min_y,inspection_.max_z-inspection_.min_z});
    }
    if (!(span>0.0)) span=1.0;
    struct Face { std::array<wxPoint,3> p; double depth; };
    std::vector<Face> faces;
    const double cy=std::cos(yaw_), sy=std::sin(yaw_), cp=std::cos(pitch_), sp=std::sin(pitch_);
    const double scale=std::min(size.x,size.y)*0.72*zoom_/span;
    const std::size_t stride=std::max<std::size_t>(1,triangles_.size()/12000);
    for (std::size_t i=0;i<triangles_.size();i+=stride) {
        Face face{};
        double depth=0;
        bool good=true;
        for (std::size_t j=0;j<3;++j) {
            const auto& v=triangles_[i][j];
            if (!Finite(v)) { good=false; break; }
            const double x=v[0]-center[0], y=v[1]-center[1], z=v[2]-center[2];
            const double xr=cy*x+sy*z, zr=-sy*x+cy*z;
            const double yr=cp*y-sp*zr, zz=sp*y+cp*zr;
            face.p[j]=wxPoint(static_cast<int>(size.x/2+xr*scale),static_cast<int>(size.y/2-yr*scale));
            depth+=zz;
        }
        if(good){face.depth=depth/3;faces.push_back(face);}
    }
    std::sort(faces.begin(),faces.end(),[](const Face&a,const Face&b){return a.depth<b.depth;});
    dc.SetPen(wxPen(wxColour(13,20,28),1));
    for(const auto& face:faces) {
        wxPoint points[3]={face.p[0],face.p[1],face.p[2]};
        const int shade=static_cast<int>(std::clamp(100.0+face.depth/span*45.0,65.0,175.0));
        dc.SetBrush(wxBrush(wxColour(shade,shade+std::min(18,255-shade),std::min(255,shade+35))));
        dc.DrawPolygon(3,points);
    }
    dc.SetTextForeground(wxColour(220,230,239));
    dc.DrawText(wxString::Format("%zu triangles  |  drag: rotate  |  wheel: zoom",triangles_.size()),10,10);
}
void MeshViewerPanel::OnMouseDown(wxMouseEvent& e){dragging_=true;last_mouse_=e.GetPosition();CaptureMouse();}
void MeshViewerPanel::OnMouseMove(wxMouseEvent& e){
    if(!dragging_||!e.LeftIsDown())return;
    const auto p=e.GetPosition();yaw_+=(p.x-last_mouse_.x)*0.012;pitch_+=(p.y-last_mouse_.y)*0.012;
    pitch_=std::clamp(pitch_,-1.55,1.55);last_mouse_=p;Refresh(false);
}
void MeshViewerPanel::OnMouseUp(wxMouseEvent&){dragging_=false;if(HasCapture())ReleaseMouse();}
void MeshViewerPanel::OnMouseWheel(wxMouseEvent& e){
    zoom_*=e.GetWheelRotation()>0?1.12:1.0/1.12;zoom_=std::clamp(zoom_,0.05,30.0);Refresh(false);
}

} // namespace cvcascade::gui
