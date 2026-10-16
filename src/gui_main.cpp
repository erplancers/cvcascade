#include "cvcascade/mesh_builder.hpp"
#include "cvcascade/photogrammetry.hpp"

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/filedlg.h>
#include <wx/filename.h>
#include <wx/gauge.h>
#include <wx/statbmp.h>
#include <wx/spinctrl.h>
#include <wx/timer.h>
#include <wx/config.h>
#include "stl_viewer_panel.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/persistence.hpp>

#include <atomic>
#include <cmath>
#include <functional>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef CVCASCADE_HAS_OPENMP
#include <omp.h>
#endif

namespace {

enum class PickMode { Open, Save };

void WriteQualityReport(const std::string& filename, const std::string& stl,
                        const std::string& ply, const cvcascade::ReconstructionStats& stats) {
    const std::filesystem::path path(filename);
    if (!path.parent_path().empty() && !std::filesystem::exists(path.parent_path()))
        throw std::runtime_error("Report directory does not exist: " + path.parent_path().string());
    const std::filesystem::path temporary(path.string() + ".tmp");
    cv::FileStorage report(temporary.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!report.isOpened()) throw std::runtime_error("Cannot create quality report: " + temporary.string());
    report << "schema_version" << 1
           << "images" << static_cast<double>(stats.images)
           << "registered_pairs" << static_cast<double>(stats.registered_pairs)
           << "feature_matches" << static_cast<double>(stats.feature_matches)
           << "shared_tracks" << static_cast<double>(stats.shared_tracks)
           << "bundle_observations" << static_cast<double>(stats.bundle_observations)
           << "bundle_iterations" << static_cast<double>(stats.bundle_iterations)
           << "bundle_initial_rmse_px" << stats.bundle_initial_rmse
           << "bundle_final_rmse_px" << stats.bundle_final_rmse
           << "pose_inliers" << static_cast<double>(stats.pose_inliers)
           << "mean_pose_inlier_ratio" << stats.mean_pose_inlier_ratio
           << "mean_valid_depth_ratio" << stats.mean_valid_depth_ratio
           << "stereo_candidates" << static_cast<double>(stats.stereo_candidates)
           << "stereo_consistent" << static_cast<double>(stats.stereo_consistent)
           << "stereo_rejected" << static_cast<double>(stats.stereo_rejected)
           << "vertices" << static_cast<double>(stats.vertices)
           << "triangles" << static_cast<double>(stats.triangles)
           << "boundary_edges" << static_cast<double>(stats.mesh_boundary_edges)
           << "non_manifold_edges" << static_cast<double>(stats.mesh_non_manifold_edges)
           << "degenerate_faces_removed" << static_cast<double>(stats.mesh_degenerate_faces_removed)
           << "duplicate_faces_removed" << static_cast<double>(stats.mesh_duplicate_faces_removed)
           << "fused_points" << static_cast<double>(stats.fused_points)
           << "fusion_voxels_rejected" << static_cast<double>(stats.fusion_voxels_rejected)
           << "applied_scale_factor" << stats.applied_scale_factor
           << "metric_scale" << static_cast<int>(stats.metric_scale)
           << "outputs" << "{"
           << "stl" << stl
           << "ply" << ply
           << "}";
    report.release();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("Cannot finalize quality report: " + ec.message());
    }
}

class MainFrame final : public wxFrame {
public:
    MainFrame()
        : wxFrame(nullptr, wxID_ANY, "CVCascade Studio — Photogrammetry Workbench",
                  wxDefaultPosition, wxSize(1180, 850),
                  wxDEFAULT_FRAME_STYLE | wxCLIP_CHILDREN),
          timer_(this) {
        SetMinSize(wxSize(900, 680));
        wxInitAllImageHandlers();
        if (wxConfigBase::Get(false) == nullptr) wxConfigBase::Set(new wxConfig("CVCascade Studio"));

        auto* root = new wxBoxSizer(wxVERTICAL);
        auto* heading = new wxPanel(this);
        heading->SetBackgroundColour(wxColour(29, 42, 57));
        auto* heading_sizer = new wxBoxSizer(wxVERTICAL);
        auto* title = new wxStaticText(heading, wxID_ANY, "CVCascade Studio");
        title->SetForegroundColour(*wxWHITE);
        title->SetFont(wxFontInfo(20).Bold());
        auto* subtitle = new wxStaticText(
            heading, wxID_ANY,
            "Photo relief  •  Camera calibration  •  Multi-view stereo  •  Colored point clouds");
        subtitle->SetForegroundColour(wxColour(218, 226, 234));
        heading_sizer->Add(title, 0, wxLEFT | wxRIGHT | wxTOP, 16);
        heading_sizer->Add(subtitle, 0, wxLEFT | wxRIGHT | wxBOTTOM, 16);
        heading->SetSizer(heading_sizer);
        root->Add(heading, 0, wxEXPAND);

        notebook_ = new wxNotebook(this, wxID_ANY);
        BuildReliefPage();
        BuildCalibrationPage();
        BuildReconstructionPage();
        BuildMeshToolsPage();
        root->Add(notebook_, 1, wxEXPAND | wxALL, 10);

        auto* status_panel = new wxPanel(this);
        auto* status_sizer = new wxBoxSizer(wxVERTICAL);
        auto* status_row = new wxBoxSizer(wxHORIZONTAL);
        status_ = new wxStaticText(status_panel, wxID_ANY, "Ready");
        status_->SetFont(wxFontInfo(10).Bold());
        gauge_ = new wxGauge(status_panel, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 14));
        status_row->Add(status_, 1, wxALIGN_CENTER_VERTICAL | wxRIGHT, 12);
        status_row->Add(gauge_, 0, wxALIGN_CENTER_VERTICAL | wxEXPAND, 0);
        cancel_button_ = new wxButton(status_panel, wxID_ANY, "Cancel job");
        cancel_button_->Enable(false);
        status_row->Add(cancel_button_, 0, wxLEFT, 8);
        cancel_button_->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RequestCancel(); });
        status_sizer->Add(status_row, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

        auto* log_label = new wxStaticText(status_panel, wxID_ANY, "Activity log");
        log_label->SetFont(wxFontInfo(10).Bold());
        status_sizer->Add(log_label, 0, wxLEFT | wxTOP, 8);
        log_ = new wxTextCtrl(status_panel, wxID_ANY, "",
                              wxDefaultPosition, wxSize(-1, 135),
                              wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH2 | wxHSCROLL);
        status_sizer->Add(log_, 1, wxEXPAND | wxALL, 8);
        status_panel->SetSizer(status_sizer);
        root->Add(status_panel, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 8);
        SetSizer(root);

        auto* menu_file = new wxMenu();
        menu_file->Append(10001, "&Open project...");
        menu_file->Append(10002, "&Save project...");
        menu_file->AppendSeparator();
        menu_file->Append(10003, "Load &preset...");
        menu_file->Append(10004, "Save preset...");
        menu_file->AppendSeparator();
        menu_file->Append(wxID_ABOUT, "&About CVCascade Studio...");
        menu_file->AppendSeparator();
        menu_file->Append(wxID_EXIT, "E&xit");
        auto* menu_bar = new wxMenuBar();
        menu_bar->Append(menu_file, "&File");
        SetMenuBar(menu_bar);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) {
            wxMessageBox(
                "CVCascade Studio\n\nA C++20 desktop front end for the OpenCV + Open CASCADE "
                "photogrammetry pipeline.\n\nThe multi-view reconstruction is incremental and "
                "pairwise; inspect scale and quality metrics before relying on measurements.",
                "About CVCascade Studio", wxOK | wxICON_INFORMATION, this);
        }, wxID_ABOUT);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { OpenProjectDialog(false); }, 10001);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { SaveProjectDialog(false); }, 10002);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { OpenProjectDialog(true); }, 10003);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { SaveProjectDialog(true); }, 10004);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { Close(); }, wxID_EXIT);
        Bind(wxEVT_CLOSE_WINDOW, &MainFrame::OnClose, this);
        Bind(wxEVT_TIMER, [this](wxTimerEvent&) {
            if (busy_) gauge_->Pulse();
        }, timer_.GetId());

        LoadPersistentSettings();
        Log("Ready. Open/save projects and presets from the File menu.");
        Centre();
    }

    ~MainFrame() override {
        timer_.Stop();
        if (worker_.joinable()) worker_.join();
    }

private:
    wxNotebook* notebook_ = nullptr;
    wxStaticText* status_ = nullptr;
    wxGauge* gauge_ = nullptr;
    wxTextCtrl* log_ = nullptr;
    wxTimer timer_;
    std::thread worker_;
    std::atomic_bool busy_{false};
    std::atomic_bool cancel_requested_{false};
    wxButton* cancel_button_ = nullptr;
    StlViewerPanel* mesh_viewer_ = nullptr;
    wxTextCtrl* mesh_validation_ = nullptr;
    wxTextCtrl* mesh_path_ = nullptr;
    wxString last_project_path_;

    // Single-photo relief controls.
    wxTextCtrl* relief_input_ = nullptr;
    wxTextCtrl* relief_output_ = nullptr;
    wxSpinCtrlDouble* relief_width_ = nullptr;
    wxSpinCtrlDouble* relief_height_ = nullptr;
    wxSpinCtrl* relief_dimension_ = nullptr;
    wxSpinCtrl* relief_threads_ = nullptr;
    wxCheckBox* relief_invert_ = nullptr;
    wxStaticBitmap* relief_preview_ = nullptr;
    wxStaticText* relief_image_info_ = nullptr;

    // Camera calibration controls.
    wxTextCtrl* calibration_images_ = nullptr;
    wxTextCtrl* calibration_output_ = nullptr;
    wxSpinCtrl* board_cols_ = nullptr;
    wxSpinCtrl* board_rows_ = nullptr;
    wxSpinCtrlDouble* square_size_ = nullptr;

    // Multi-view reconstruction controls.
    wxTextCtrl* reconstruction_images_ = nullptr;
    wxTextCtrl* reconstruction_calibration_ = nullptr;
    wxTextCtrl* reconstruction_stl_ = nullptr;
    wxTextCtrl* reconstruction_ply_ = nullptr;
    wxTextCtrl* reconstruction_report_ = nullptr;
    wxSpinCtrl* reconstruction_dimension_ = nullptr;
    wxSpinCtrl* disparities_ = nullptr;
    wxSpinCtrl* block_size_ = nullptr;
    wxSpinCtrl* min_inliers_ = nullptr;
    wxSpinCtrlDouble* baseline_mm_ = nullptr;
    wxSpinCtrlDouble* max_depth_ = nullptr;
    wxSpinCtrlDouble* depth_jump_ = nullptr;
    wxSpinCtrlDouble* lr_tolerance_ = nullptr;
    wxSpinCtrlDouble* fusion_voxel_ = nullptr;
    wxSpinCtrl* speckle_window_ = nullptr;
    wxSpinCtrl* speckle_range_ = nullptr;

    static wxPanel* NewPage(wxNotebook* notebook, const wxString& name) {
        auto* page = new wxPanel(notebook);
        page->SetName(name);
        notebook->AddPage(page, name);
        return page;
    }

    static void AddControlRow(wxWindow* parent, wxBoxSizer* root,
                               const wxString& label, wxWindow* control) {
        auto* row = new wxBoxSizer(wxHORIZONTAL);
        auto* text = new wxStaticText(parent, wxID_ANY, label,
                                      wxDefaultPosition, wxSize(190, -1));
        row->Add(text, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
        row->Add(control, 1, wxEXPAND);
        root->Add(row, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);
    }

    wxTextCtrl* AddPathRow(wxWindow* parent, wxBoxSizer* root,
                           const wxString& label, const wxString& initial,
                           PickMode mode, const wxString& wildcard,
                           std::function<void(const wxString&)> on_selected = {}) {
        auto* row = new wxBoxSizer(wxHORIZONTAL);
        auto* text = new wxStaticText(parent, wxID_ANY, label,
                                      wxDefaultPosition, wxSize(190, -1));
        auto* path = new wxTextCtrl(parent, wxID_ANY, initial);
        auto* browse = new wxButton(parent, wxID_ANY, "Browse…", wxDefaultPosition, wxSize(92, -1));
        row->Add(text, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
        row->Add(path, 1, wxEXPAND);
        row->Add(browse, 0, wxLEFT, 6);
        root->Add(row, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

        browse->Bind(wxEVT_BUTTON, [this, parent, path, mode, wildcard, label, on_selected](wxCommandEvent&) {
            const long flags = mode == PickMode::Open
                ? wxFD_OPEN | wxFD_FILE_MUST_EXIST
                : wxFD_SAVE | wxFD_OVERWRITE_PROMPT;
            wxFileDialog dialog(parent, "Choose " + label, "", path->GetValue(), wildcard, flags);
            if (dialog.ShowModal() == wxID_OK) {
                path->SetValue(dialog.GetPath());
                if (on_selected) on_selected(dialog.GetPath());
            }
        });
        return path;
    }

    static wxSpinCtrl* MakeInt(wxWindow* parent, int initial, int minimum, int maximum, int increment = 1) {
        auto* spin = new wxSpinCtrl(parent, wxID_ANY);
        spin->SetRange(minimum, maximum);
        spin->SetIncrement(increment);
        spin->SetValue(initial);
        return spin;
    }

    static wxSpinCtrlDouble* MakeDouble(wxWindow* parent, double initial,
                                        double minimum, double maximum, double increment,
                                        unsigned digits = 3) {
        auto* spin = new wxSpinCtrlDouble(parent, wxID_ANY);
        spin->SetRange(minimum, maximum);
        spin->SetIncrement(increment);
        spin->SetDigits(digits);
        spin->SetValue(initial);
        return spin;
    }

    void RequestCancel() {
        if (!busy_) return;
        cancel_requested_ = true;
        cancel_button_->Enable(false);
        status_->SetLabel("Cancellation requested; waiting for a safe checkpoint...");
        Log("Cancellation requested.");
    }

    void SaveProjectFile(const std::string& filename, bool preset) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) throw std::runtime_error("Cannot write project/preset file: " + filename);
        fs << "schema_version" << 1 << "kind" << (preset ? "preset" : "project");
        auto text = [&](const char* k, wxTextCtrl* c) { fs << k << c->GetValue().ToStdString(); };
        auto integer = [&](const char* k, wxSpinCtrl* c) { fs << k << c->GetValue(); };
        auto real = [&](const char* k, wxSpinCtrlDouble* c) { fs << k << c->GetValue(); };
        auto boolean = [&](const char* k, wxCheckBox* c) { fs << k << static_cast<int>(c->GetValue()); };
        text("relief_input",relief_input_); text("relief_output",relief_output_);
        real("relief_width",relief_width_); real("relief_height",relief_height_);
        integer("relief_dimension",relief_dimension_); integer("relief_threads",relief_threads_); boolean("relief_invert",relief_invert_);
        text("calibration_images",calibration_images_); text("calibration_output",calibration_output_);
        integer("board_cols",board_cols_); integer("board_rows",board_rows_); real("square_size",square_size_);
        text("reconstruction_images",reconstruction_images_); text("reconstruction_calibration",reconstruction_calibration_);
        text("reconstruction_stl",reconstruction_stl_); text("reconstruction_ply",reconstruction_ply_);
        text("reconstruction_report",reconstruction_report_);
        integer("reconstruction_dimension",reconstruction_dimension_); integer("disparities",disparities_);
        integer("block_size",block_size_); integer("min_inliers",min_inliers_);
        real("baseline_mm",baseline_mm_); real("max_depth",max_depth_); real("depth_jump",depth_jump_);
        real("lr_tolerance",lr_tolerance_); real("fusion_voxel",fusion_voxel_);
        integer("speckle_window",speckle_window_); integer("speckle_range",speckle_range_);
        text("mesh_path",mesh_path_); fs.release();
        if (auto* config=wxConfigBase::Get(false)) { config->Write("LastProjectPath",wxString::FromUTF8(filename)); config->Flush(); }
    }

    void LoadProjectFile(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("Cannot open project/preset file: " + filename);
        int schema=0; fs["schema_version"] >> schema;
        if (schema != 1) throw std::runtime_error("Unsupported project/preset schema version.");
        auto text = [&](const char* k, wxTextCtrl* c) { cv::FileNode n=fs[k]; if(!n.empty()){std::string v;n>>v;c->SetValue(wxString::FromUTF8(v));} };
        auto integer = [&](const char* k, wxSpinCtrl* c) { cv::FileNode n=fs[k]; if(!n.empty()){int v=0;n>>v;c->SetValue(v);} };
        auto real = [&](const char* k, wxSpinCtrlDouble* c) { cv::FileNode n=fs[k]; if(!n.empty()){double v=0;n>>v;c->SetValue(v);} };
        auto boolean = [&](const char* k, wxCheckBox* c) { cv::FileNode n=fs[k]; if(!n.empty()){int v=0;n>>v;c->SetValue(v!=0);} };
        text("relief_input",relief_input_); text("relief_output",relief_output_);
        real("relief_width",relief_width_); real("relief_height",relief_height_);
        integer("relief_dimension",relief_dimension_); integer("relief_threads",relief_threads_); boolean("relief_invert",relief_invert_);
        text("calibration_images",calibration_images_); text("calibration_output",calibration_output_);
        integer("board_cols",board_cols_); integer("board_rows",board_rows_); real("square_size",square_size_);
        text("reconstruction_images",reconstruction_images_); text("reconstruction_calibration",reconstruction_calibration_);
        text("reconstruction_stl",reconstruction_stl_); text("reconstruction_ply",reconstruction_ply_);
        text("reconstruction_report",reconstruction_report_);
        integer("reconstruction_dimension",reconstruction_dimension_); integer("disparities",disparities_);
        integer("block_size",block_size_); integer("min_inliers",min_inliers_);
        real("baseline_mm",baseline_mm_); real("max_depth",max_depth_); real("depth_jump",depth_jump_);
        real("lr_tolerance",lr_tolerance_); real("fusion_voxel",fusion_voxel_);
        integer("speckle_window",speckle_window_); integer("speckle_range",speckle_range_); text("mesh_path",mesh_path_);
        last_project_path_=wxString::FromUTF8(filename);
        if (!mesh_path_->GetValue().empty() && std::filesystem::is_regular_file(mesh_path_->GetValue().ToStdString())) {
            if(mesh_viewer_->LoadStl(mesh_path_->GetValue().ToStdString()))
                mesh_validation_->SetValue(wxString::FromUTF8(mesh_viewer_->ValidationSummary()));
        }
        if(auto* config=wxConfigBase::Get(false)){config->Write("LastProjectPath",last_project_path_);config->Flush();}
        Log(wxString("Loaded project/preset: ")+wxString::FromUTF8(filename));
    }

    void SaveProjectDialog(bool preset) {
        wxFileDialog dialog(this,preset?"Save preset":"Save project","","",
            preset?"CVCascade preset (*.cvcpreset)|*.cvcpreset|JSON (*.json)|*.json":"CVCascade project (*.cvcproj)|*.cvcproj|JSON (*.json)|*.json",
            wxFD_SAVE|wxFD_OVERWRITE_PROMPT);
        if(dialog.ShowModal()!=wxID_OK)return;
        try { SaveProjectFile(dialog.GetPath().ToStdString(),preset); last_project_path_=dialog.GetPath(); Log(wxString("Saved: ")+dialog.GetPath()); }
        catch(const std::exception& e){wxMessageBox(wxString::FromUTF8(e.what()),"Save failed",wxOK|wxICON_ERROR,this);}
    }

    void OpenProjectDialog(bool preset) {
        wxString initial;
        if(auto* config=wxConfigBase::Get(false))config->Read("LastProjectPath",&initial);
        wxFileDialog dialog(this,preset?"Load preset":"Open project","",initial,
            preset?"CVCascade preset (*.cvcpreset;*.json)|*.cvcpreset;*.json|All files|*.*":"CVCascade project (*.cvcproj;*.json)|*.cvcproj;*.json|All files|*.*",
            wxFD_OPEN|wxFD_FILE_MUST_EXIST);
        if(dialog.ShowModal()!=wxID_OK)return;
        try { LoadProjectFile(dialog.GetPath().ToStdString()); if(preset)Log("Preset loaded; use Save project to persist the working session."); }
        catch(const std::exception& e){wxMessageBox(wxString::FromUTF8(e.what()),"Open failed",wxOK|wxICON_ERROR,this);}
    }

    void LoadPersistentSettings() {
        if(auto* config=wxConfigBase::Get(false)){
            wxString path;
            if(config->Read("LastProjectPath",&path)&&!path.empty()&&std::filesystem::is_regular_file(path.ToStdString())){
                try{LoadProjectFile(path.ToStdString());}catch(...){Log("Could not restore last project; using defaults.");}
            }
        }
    }

    void BuildReliefPage() {
        auto* page = new wxPanel(notebook_);
        notebook_->AddPage(page, "Photo relief");
        auto* root = new wxBoxSizer(wxVERTICAL);
        auto* intro = new wxStaticText(page, wxID_ANY,
            "Convert brightness into a height map and export a triangulated STL relief.");
        root->Add(intro, 0, wxALL, 10);

        auto* body = new wxBoxSizer(wxHORIZONTAL);
        auto* form = new wxBoxSizer(wxVERTICAL);
        relief_input_ = AddPathRow(page, form, "Input photograph", "",
            PickMode::Open, "Image files (*.png;*.jpg;*.jpeg;*.bmp;*.tif)|*.png;*.jpg;*.jpeg;*.bmp;*.tif|All files|*.*",
            [this](const wxString& path) { UpdatePreview(path); });
        relief_output_ = AddPathRow(page, form, "Output STL", "relief.stl",
            PickMode::Save, "STL mesh (*.stl)|*.stl|All files|*.*");
        relief_width_ = MakeDouble(page, 120.0, 1.0, 100000.0, 1.0, 2);
        relief_height_ = MakeDouble(page, 8.0, 0.0, 100000.0, 0.5, 2);
        relief_dimension_ = MakeInt(page, 768, 64, 8192, 64);
        relief_threads_ = MakeInt(page, 0, 0, 256);
        relief_invert_ = new wxCheckBox(page, wxID_ANY, "Invert brightness-to-height mapping");
        AddControlRow(page, form, "Physical width (mm)", relief_width_);
        AddControlRow(page, form, "Maximum height (mm)", relief_height_);
        AddControlRow(page, form, "Maximum image dimension", relief_dimension_);
        AddControlRow(page, form, "OpenMP threads (0 = default)", relief_threads_);
        form->Add(relief_invert_, 0, wxALL, 10);
        auto* run = new wxButton(page, wxID_ANY, "Generate relief STL");
        run->SetMinSize(wxSize(-1, 38));
        form->Add(run, 0, wxEXPAND | wxALL, 10);
        run->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RunRelief(); });

        auto* preview_panel = new wxPanel(page);
        auto* preview_sizer = new wxBoxSizer(wxVERTICAL);
        auto* preview_title = new wxStaticText(preview_panel, wxID_ANY, "Image preview");
        preview_title->SetFont(wxFontInfo(11).Bold());
        preview_sizer->Add(preview_title, 0, wxBOTTOM, 8);
        relief_preview_ = new wxStaticBitmap(preview_panel, wxID_ANY, wxBitmap(320, 240));
        preview_sizer->Add(relief_preview_, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 4);
        relief_image_info_ = new wxStaticText(preview_panel, wxID_ANY, "No photograph selected",
                                              wxDefaultPosition, wxSize(300, -1), wxST_NO_AUTORESIZE);
        relief_image_info_->Wrap(300);
        preview_sizer->Add(relief_image_info_, 0, wxALL, 4);
        preview_panel->SetSizer(preview_sizer);

        body->Add(form, 3, wxEXPAND | wxRIGHT, 12);
        body->Add(preview_panel, 2, wxEXPAND | wxALL, 10);
        root->Add(body, 1, wxEXPAND);
        page->SetSizer(root);
    }

    void BuildCalibrationPage() {
        auto* page = new wxPanel(notebook_);
        notebook_->AddPage(page, "Camera calibration");
        auto* root = new wxBoxSizer(wxVERTICAL);
        auto* intro = new wxStaticText(page, wxID_ANY,
            "Calibrate the lens from varied chessboard images. The manifest contains one image path per line.");
        intro->Wrap(1000);
        root->Add(intro, 0, wxALL, 10);
        calibration_images_ = AddPathRow(page, root, "Image manifest (.txt)", "calibration-images.txt",
            PickMode::Open, "Text manifest (*.txt)|*.txt|All files|*.*");
        calibration_output_ = AddPathRow(page, root, "Calibration output", "camera.yml",
            PickMode::Save, "YAML calibration (*.yml;*.yaml)|*.yml;*.yaml|All files|*.*");

        auto* settings = new wxStaticBoxSizer(wxVERTICAL, page, "Chessboard pattern");
        board_cols_ = MakeInt(page, 9, 2, 100);
        board_rows_ = MakeInt(page, 6, 2, 100);
        square_size_ = MakeDouble(page, 25.0, 0.001, 100000.0, 1.0, 3);
        AddControlRow(page, settings, "Inner-corner columns", board_cols_);
        AddControlRow(page, settings, "Inner-corner rows", board_rows_);
        AddControlRow(page, settings, "Square size (mm)", square_size_);
        root->Add(settings, 0, wxEXPAND | wxALL, 10);

        auto* note = new wxStaticText(page, wxID_ANY,
            "Tip: use 5–15 sharp images with the same resolution, varied board tilt and distance. "
            "Columns/rows count inner corners, not squares.");
        note->Wrap(1000);
        root->Add(note, 0, wxLEFT | wxRIGHT | wxBOTTOM, 12);
        auto* run = new wxButton(page, wxID_ANY, "Calibrate camera");
        run->SetMinSize(wxSize(-1, 38));
        root->Add(run, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        run->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RunCalibration(); });
        page->SetSizer(root);
    }

    void BuildMeshToolsPage() {
        auto* page = new wxPanel(notebook_);
        notebook_->AddPage(page, "Mesh viewer & validation");
        auto* root = new wxBoxSizer(wxVERTICAL);
        auto* row = new wxBoxSizer(wxHORIZONTAL);
        mesh_path_ = new wxTextCtrl(page, wxID_ANY, "");
        auto* open = new wxButton(page, wxID_ANY, "Open STL...");
        auto* validate = new wxButton(page, wxID_ANY, "Validate mesh");
        auto* export_ply = new wxButton(page, wxID_ANY, "Export PLY...");
        row->Add(mesh_path_, 1, wxEXPAND | wxRIGHT, 6);
        row->Add(open, 0, wxRIGHT, 6); row->Add(validate, 0, wxRIGHT, 6); row->Add(export_ply, 0);
        root->Add(row, 0, wxEXPAND | wxALL, 8);
        auto* body = new wxBoxSizer(wxHORIZONTAL);
        mesh_viewer_ = new StlViewerPanel(page);
        mesh_validation_ = new wxTextCtrl(page, wxID_ANY, "Open an STL to inspect triangle count and topology.",
            wxDefaultPosition, wxSize(340, -1), wxTE_MULTILINE | wxTE_READONLY | wxHSCROLL);
        body->Add(mesh_viewer_, 2, wxEXPAND | wxALL, 4); body->Add(mesh_validation_, 1, wxEXPAND | wxALL, 4);
        root->Add(body, 1, wxEXPAND);
        auto load = [this, page]() {
            const std::string path = mesh_path_->GetValue().ToStdString();
            if (path.empty() || !mesh_viewer_->LoadStl(path)) {
                wxMessageBox("Could not load the selected STL.", "Mesh viewer", wxOK | wxICON_WARNING, page); return;
            }
            mesh_validation_->SetValue(wxString::FromUTF8(mesh_viewer_->ValidationSummary()));
        };
        open->Bind(wxEVT_BUTTON, [this, page, load](wxCommandEvent&) {
            wxFileDialog dialog(page, "Open mesh", "", mesh_path_->GetValue(),
                "STL mesh (*.stl)|*.stl|All files|*.*", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
            if (dialog.ShowModal() == wxID_OK) { mesh_path_->SetValue(dialog.GetPath()); load(); }
        });
        validate->Bind(wxEVT_BUTTON, [load](wxCommandEvent&) { load(); });
        export_ply->Bind(wxEVT_BUTTON, [this, page](wxCommandEvent&) {
            wxFileDialog dialog(page, "Export triangle mesh as PLY", "", "mesh.ply",
                "PLY mesh (*.ply)|*.ply", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
            if (dialog.ShowModal() != wxID_OK) return;
            if (!mesh_viewer_->ExportPly(dialog.GetPath().ToStdString()))
                wxMessageBox("PLY export failed. Load a valid STL first.", "Export failed", wxOK | wxICON_ERROR, page);
            else Log(wxString("Exported triangle-soup PLY: ") + dialog.GetPath());
        });
        page->SetSizer(root);
    }

    void BuildReconstructionPage() {
        auto* page = new wxPanel(notebook_);
        notebook_->AddPage(page, "Multi-view reconstruction");
        auto* page_sizer = new wxBoxSizer(wxVERTICAL);
        auto* scroller = new wxScrolledWindow(page, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scroller->SetScrollRate(0, 14);
        auto* outer = new wxBoxSizer(wxVERTICAL);

        auto* intro = new wxStaticText(scroller, wxID_ANY,
            "Incremental SfM + pairwise stereo. Use ordered, overlapping photographs from the calibrated camera.");
        intro->Wrap(1000);
        outer->Add(intro, 0, wxALL, 10);

        auto* files = new wxStaticBoxSizer(wxVERTICAL, scroller, "Inputs and outputs");
        reconstruction_images_ = AddPathRow(scroller, files, "Image manifest (.txt)", "views.txt",
            PickMode::Open, "Text manifest (*.txt)|*.txt|All files|*.*");
        reconstruction_calibration_ = AddPathRow(scroller, files, "Camera calibration", "camera.yml",
            PickMode::Open, "Calibration files (*.yml;*.yaml;*.xml)|*.yml;*.yaml;*.xml|All files|*.*");
        reconstruction_stl_ = AddPathRow(scroller, files, "Output mesh (.stl)", "reconstruction.stl",
            PickMode::Save, "STL mesh (*.stl)|*.stl|All files|*.*");
        reconstruction_ply_ = AddPathRow(scroller, files, "Optional RGB point cloud (.ply)", "reconstruction.ply",
            PickMode::Save, "PLY point cloud (*.ply)|*.ply|All files|*.*");
        reconstruction_report_ = AddPathRow(scroller, files, "Quality report (.json)", "reconstruction-report.json",
            PickMode::Save, "JSON report (*.json)|*.json|All files|*.*");
        outer->Add(files, 0, wxEXPAND | wxLEFT | wxRIGHT, 8);

        auto* settings = new wxStaticBoxSizer(wxVERTICAL, scroller, "Stereo, filtering and fusion");
        auto* grid = new wxFlexGridSizer(2, 8, 12);
        grid->AddGrowableCol(1, 1);
        auto add_grid = [&](const wxString& label, wxWindow* control) {
            grid->Add(new wxStaticText(scroller, wxID_ANY, label), 0, wxALIGN_CENTER_VERTICAL);
            grid->Add(control, 1, wxEXPAND);
        };

        reconstruction_dimension_ = MakeInt(scroller, 640, 64, 8192, 64);
        disparities_ = MakeInt(scroller, 128, 16, 4096, 16);
        block_size_ = MakeInt(scroller, 5, 3, 51, 2);
        min_inliers_ = MakeInt(scroller, 30, 8, 100000);
        baseline_mm_ = MakeDouble(scroller, 0.0, 0.0, 1000000000.0, 1.0, 3);
        max_depth_ = MakeDouble(scroller, 10000.0, 0.001, 1000000000.0, 100.0, 3);
        depth_jump_ = MakeDouble(scroller, 0.20, 0.001, 1.0, 0.01, 3);
        lr_tolerance_ = MakeDouble(scroller, 1.5, 0.1, 1000.0, 0.1, 2);
        speckle_window_ = MakeInt(scroller, 100, 0, 100000);
        speckle_range_ = MakeInt(scroller, 2, 0, 10000);
        fusion_voxel_ = MakeDouble(scroller, 2.0, 0.001, 1000000000.0, 0.5, 3);

        add_grid("Maximum image dimension", reconstruction_dimension_);
        add_grid("Disparities (multiple of 16)", disparities_);
        add_grid("Block size (odd)", block_size_);
        add_grid("Minimum pose inliers", min_inliers_);
        add_grid("Assumed pair baseline (mm; 0 = unknown)", baseline_mm_);
        add_grid("Maximum depth (model units)", max_depth_);
        add_grid("Depth-edge jump ratio", depth_jump_);
        add_grid("Left/right disparity tolerance (px)", lr_tolerance_);
        add_grid("Speckle window (0 = off)", speckle_window_);
        add_grid("Speckle disparity range", speckle_range_);
        add_grid("Point-cloud fusion voxel size", fusion_voxel_);
        settings->Add(grid, 0, wxEXPAND | wxALL, 10);
        outer->Add(settings, 0, wxEXPAND | wxALL, 8);

        auto* note = new wxStaticText(scroller, wxID_ANY,
            "Without a known baseline, scale and voxel size are arbitrary. Voxel fusion affects PLY; "
            "the STL retains per-pair surfaces. Review bundle-adjustment and mesh-topology diagnostics; conservative repair does not guarantee watertightness.");
        note->Wrap(1000);
        outer->Add(note, 0, wxLEFT | wxRIGHT | wxBOTTOM, 10);
        auto* run = new wxButton(scroller, wxID_ANY, "Reconstruct mesh and point cloud");
        run->SetMinSize(wxSize(-1, 40));
        outer->Add(run, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        run->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RunReconstruction(); });

        scroller->SetSizer(outer);
        page_sizer->Add(scroller, 1, wxEXPAND);
        page->SetSizer(page_sizer);
    }

    void UpdatePreview(const wxString& path) {
        if (path.empty()) return;
        const cv::Mat image = cv::imread(path.ToStdString(), cv::IMREAD_COLOR);
        if (image.empty()) {
            relief_image_info_->SetLabel("Could not decode the selected image.");
            relief_preview_->SetBitmap(wxBitmap(320, 240));
            return;
        }
        wxImage preview(path);
        if (preview.IsOk()) {
            preview.Rescale(320, 240, wxIMAGE_QUALITY_HIGH);
            relief_preview_->SetBitmap(wxBitmap(preview));
        }
        relief_image_info_->SetLabel(wxString::Format(
            "%s\nResolution: %d × %d\nChannels: %d",
            wxFileName(path).GetFullName(), image.cols, image.rows, image.channels()));
        Layout();
    }

    void Log(const wxString& message) {
        if (log_ == nullptr) return;
        log_->AppendText(wxDateTime::Now().FormatISOTime() + "  " + message + "\n");
    }

    void RunJob(const wxString& label, std::function<std::string(const std::atomic_bool&)> operation) {
        if (busy_) {
            wxMessageBox("Another operation is already running.", "CVCascade Studio",
                         wxOK | wxICON_INFORMATION, this);
            return;
        }
        if (worker_.joinable()) worker_.join();

        cancel_requested_ = false;
        busy_ = true;
        cancel_button_->Enable(true);
        notebook_->Enable(false);
        gauge_->Pulse();
        status_->SetLabel(label + "…");
        Log("Started: " + label);
        timer_.Start(100);

        worker_ = std::thread([this, label, operation = std::move(operation)]() mutable {
            bool succeeded = false;
            std::string result;
            try {
                result = operation(cancel_requested_);
                if (cancel_requested_) result = "Job cancelled at a safe checkpoint.";
                else succeeded = true;
            } catch (const std::exception& e) {
                result = e.what();
            } catch (...) {
                result = "Unknown non-standard exception.";
            }
            wxTheApp->CallAfter([this, label, succeeded, result = std::move(result)]() {
                FinishJob(label, succeeded, result);
            });
        });
    }

    void FinishJob(const wxString& label, bool succeeded, const std::string& result) {
        if (worker_.joinable()) worker_.join();
        timer_.Stop();
        gauge_->SetValue(succeeded ? 100 : 0);
        busy_ = false;
        cancel_button_->Enable(false);
        notebook_->Enable(true);
        const bool cancelled = !succeeded && (cancel_requested_ || result.find("cancelled") != std::string::npos);
        status_->SetLabel(succeeded ? label + " completed." : cancelled ? label + " cancelled." : label + " failed.");
        Log((succeeded ? "Completed: " : cancelled ? "Cancelled: " : "Failed: ") + label);
        Log(wxString::FromUTF8(result));
        if (succeeded)
            wxMessageBox(wxString::FromUTF8(result), label + " complete",
                         wxOK | wxICON_INFORMATION, this);
        else
            wxMessageBox(wxString::FromUTF8(result), cancelled ? label + " cancelled" : label + " failed",
                         wxOK | (cancelled ? wxICON_INFORMATION : wxICON_ERROR), this);
    }

    void RunRelief() {
        const std::string input = relief_input_->GetValue().ToStdString();
        const std::string output = relief_output_->GetValue().ToStdString();
        if (input.empty() || output.empty()) {
            wxMessageBox("Choose both an input photograph and an output STL path.", "Missing input",
                         wxOK | wxICON_WARNING, this);
            return;
        }
        cvcascade::MeshOptions options;
        options.width = relief_width_->GetValue();
        options.max_height = relief_height_->GetValue();
        options.max_dimension = relief_dimension_->GetValue();
        options.invert = relief_invert_->GetValue();
        const int threads = relief_threads_->GetValue();

        RunJob("Photo relief", [input, output, options, threads](const std::atomic_bool& cancel) {
            if (cancel) throw std::runtime_error("job cancelled");
#ifdef CVCASCADE_HAS_OPENMP
            if (threads > 0) omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            const cv::Mat image = cv::imread(input, cv::IMREAD_GRAYSCALE);
            if (cancel) throw std::runtime_error("job cancelled");
            if (image.empty()) throw std::runtime_error("Cannot decode input image: " + input);
            const cv::Mat height_map = cvcascade::make_height_map(image, options);
            const auto stats = cvcascade::write_stl(height_map, options, output);
            std::ostringstream text;
            text << "Wrote: " << output << "\nResolution: " << stats.cols << " x " << stats.rows
                 << "\nTriangles: " << stats.triangles << "\nHeight: "
                 << stats.min_height << " .. " << stats.max_height << " mm";
            return text.str();
        });
    }

    void RunCalibration() {
        const std::string images = calibration_images_->GetValue().ToStdString();
        const std::string output = calibration_output_->GetValue().ToStdString();
        const int cols = board_cols_->GetValue();
        const int rows = board_rows_->GetValue();
        const double square = square_size_->GetValue();
        if (images.empty() || output.empty()) {
            wxMessageBox("Choose an image manifest and a calibration output path.", "Missing input",
                         wxOK | wxICON_WARNING, this);
            return;
        }
        RunJob("Camera calibration", [images, output, cols, rows, square](const std::atomic_bool& cancel) {
            if (cancel) throw std::runtime_error("job cancelled");
            const auto model = cvcascade::calibrate_camera_from_list(images, output, cols, rows, square);
            if (cancel) throw std::runtime_error("job cancelled");
            std::ostringstream text;
            text << "Calibration saved: " << output
                 << "\nImage size: " << model.image_size.width << " x " << model.image_size.height
                 << "\nReprojection RMS: " << model.reprojection_rms << " px";
            return text.str();
        });
    }

    void RunReconstruction() {
        const std::string images = reconstruction_images_->GetValue().ToStdString();
        const std::string calibration = reconstruction_calibration_->GetValue().ToStdString();
        const std::string stl = reconstruction_stl_->GetValue().ToStdString();
        const std::string ply = reconstruction_ply_->GetValue().ToStdString();
        const std::string report = reconstruction_report_->GetValue().ToStdString();
        if (images.empty() || calibration.empty() || stl.empty()) {
            wxMessageBox("Choose the image manifest, camera calibration, and output STL paths.",
                         "Missing input", wxOK | wxICON_WARNING, this);
            return;
        }
        if (!std::filesystem::is_regular_file(images) || !std::filesystem::is_regular_file(calibration)) {
            wxMessageBox("The image manifest and calibration must both exist and be readable files.",
                         "Missing input file", wxOK | wxICON_WARNING, this);
            return;
        }
        const int max_dimension = reconstruction_dimension_->GetValue();
        const int disparity_count = disparities_->GetValue();
        const int selected_block_size = block_size_->GetValue();
        if (disparity_count <= 0 || disparity_count % 16 != 0 || disparity_count >= max_dimension ||
            selected_block_size < 3 || selected_block_size % 2 == 0) {
            wxMessageBox("Disparities must be a positive multiple of 16 smaller than the maximum image dimension; block size must be odd and at least 3.",
                         "Invalid stereo settings", wxOK | wxICON_WARNING, this);
            return;
        }
        const auto normalized_stl = std::filesystem::absolute(stl).lexically_normal();
        const auto normalized_ply = ply.empty() ? std::filesystem::path{} : std::filesystem::absolute(ply).lexically_normal();
        const auto normalized_report = report.empty() ? std::filesystem::path{} : std::filesystem::absolute(report).lexically_normal();
        if ((!ply.empty() && normalized_ply == normalized_stl) ||
            (!report.empty() && (normalized_report == normalized_stl || (!ply.empty() && normalized_report == normalized_ply)))) {
            wxMessageBox("STL, PLY, and JSON report output paths must be different.", "Invalid output paths",
                         wxOK | wxICON_WARNING, this);
            return;
        }

        cvcascade::PhotogrammetryOptions options;
        options.max_dimension = reconstruction_dimension_->GetValue();
        options.num_disparities = disparities_->GetValue();
        options.block_size = block_size_->GetValue();
        options.min_inliers = min_inliers_->GetValue();
        options.baseline_mm = baseline_mm_->GetValue();
        options.max_depth = max_depth_->GetValue();
        options.max_relative_depth_jump = depth_jump_->GetValue();
        options.left_right_tolerance = lr_tolerance_->GetValue();
        options.fusion_voxel_size = fusion_voxel_->GetValue();
        options.speckle_window_size = speckle_window_->GetValue();
        options.speckle_range = speckle_range_->GetValue();

        RunJob("Multi-view reconstruction", [images, calibration, stl, ply, report, options](const std::atomic_bool& cancel) {
            const auto stats = cvcascade::reconstruct_multiview(images, calibration, stl, options, ply, [&cancel]() { return cancel.load(); });
            const double consistent = stats.stereo_candidates > 0
                ? 100.0 * static_cast<double>(stats.stereo_consistent) / stats.stereo_candidates : 0.0;
            std::ostringstream text;
            text << "Mesh saved: " << stl
                 << "\nImages: " << stats.images
                 << "\nPairs with triangles: " << stats.registered_pairs
                 << "\nFeature matches: " << stats.feature_matches
                 << "\nShared multi-view tracks: " << stats.shared_tracks
                 << "\nBundle observations / iterations: " << stats.bundle_observations << " / " << stats.bundle_iterations
                 << "\nBundle reprojection RMSE (initial -> final): " << stats.bundle_initial_rmse << " -> " << stats.bundle_final_rmse << " px"
                 << "\nPose inliers: " << stats.pose_inliers
                 << "\nMean pose inlier ratio: " << stats.mean_pose_inlier_ratio
                 << "\nMean valid-depth ratio: " << stats.mean_valid_depth_ratio
                 << "\nStereo consistency: " << consistent << "%"
                 << "\nStereo candidates / consistent / rejected: "
                 << stats.stereo_candidates << " / " << stats.stereo_consistent << " / " << stats.stereo_rejected
                 << "\nMesh vertices / triangles: " << stats.vertices << " / " << stats.triangles
                 << "\nBoundary / non-manifold edges: " << stats.mesh_boundary_edges << " / " << stats.mesh_non_manifold_edges
                 << "\nDegenerate / duplicate faces removed: " << stats.mesh_degenerate_faces_removed << " / " << stats.mesh_duplicate_faces_removed
                 << "\nApplied scale factor: " << stats.applied_scale_factor
                 << "\nScale: " << (stats.metric_scale ? "baseline-scaled (approximate)" : "arbitrary")
                 << "\n";
            if (!ply.empty()) text << "Fused RGB PLY: " << ply << "\nPoints: " << stats.fused_points;
            if (!report.empty()) {
                WriteQualityReport(report, stl, ply, stats);
                text << "\nQuality report: " << report;
            }
            wxTheApp->CallAfter([this, stl]() {
                mesh_path_->SetValue(wxString::FromUTF8(stl));
                if (mesh_viewer_->LoadStl(stl))
                    mesh_validation_->SetValue(wxString::FromUTF8(mesh_viewer_->ValidationSummary()));
            });
            return text.str();
        });
    }

    void OnClose(wxCloseEvent& event) {
        if (busy_) {
            wxMessageBox("A reconstruction task is running in the background. Please wait for it to finish before closing.",
                         "Task in progress", wxOK | wxICON_INFORMATION, this);
            event.Veto();
            return;
        }
        if (worker_.joinable()) worker_.join();
        event.Skip();
    }
};

class CVCascadeApp final : public wxApp {
public:
    bool OnInit() override {
        auto* frame = new MainFrame();
        frame->Show();
        return true;
    }
};

} // namespace

wxIMPLEMENT_APP(CVCascadeApp);
