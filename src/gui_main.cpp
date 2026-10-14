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

#include <opencv2/imgcodecs.hpp>

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
        root->Add(notebook_, 1, wxEXPAND | wxALL, 10);

        auto* status_panel = new wxPanel(this);
        auto* status_sizer = new wxBoxSizer(wxVERTICAL);
        auto* status_row = new wxBoxSizer(wxHORIZONTAL);
        status_ = new wxStaticText(status_panel, wxID_ANY, "Ready");
        status_->SetFont(wxFontInfo(10).Bold());
        gauge_ = new wxGauge(status_panel, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 14));
        status_row->Add(status_, 1, wxALIGN_CENTER_VERTICAL | wxRIGHT, 12);
        status_row->Add(gauge_, 0, wxALIGN_CENTER_VERTICAL | wxEXPAND, 0);
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
        menu_file->Append(2001, "&Open project…");
        menu_file->Append(2002, "&Save project…");
        menu_file->AppendSeparator();
        menu_file->Append(wxID_ABOUT, "&About CVCascade Studio...");
        menu_file->AppendSeparator();
        menu_file->Append(wxID_EXIT, "E&xit");
        auto* menu_bar = new wxMenuBar();
        menu_bar->Append(menu_file, "&File");
        SetMenuBar(menu_bar);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { OpenProject(); }, 2001);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { SaveProject(); }, 2002);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) {
            wxMessageBox(
                "CVCascade Studio\n\nA C++20 desktop front end for the OpenCV + Open CASCADE "
                "photogrammetry pipeline.\n\nThe multi-view reconstruction is incremental and "
                "pairwise; inspect scale and quality metrics before relying on measurements.",
                "About CVCascade Studio", wxOK | wxICON_INFORMATION, this);
        }, wxID_ABOUT);
        Bind(wxEVT_MENU, [this](wxCommandEvent&) { Close(); }, wxID_EXIT);
        Bind(wxEVT_CLOSE_WINDOW, &MainFrame::OnClose, this);
        Bind(wxEVT_TIMER, [this](wxTimerEvent&) {
            if (busy_) gauge_->Pulse();
        }, timer_.GetId());

        Log("Ready. Select a workflow above to begin.");
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
        auto* preset_row = new wxBoxSizer(wxHORIZONTAL);
        auto* save_preset = new wxButton(scroller, wxID_ANY, "Save settings preset…");
        auto* load_preset = new wxButton(scroller, wxID_ANY, "Load settings preset…");
        preset_row->Add(save_preset, 1, wxRIGHT, 6);
        preset_row->Add(load_preset, 1, wxLEFT, 6);
        outer->Add(preset_row, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        save_preset->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { SavePreset(); });
        load_preset->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { LoadPreset(); });
        auto* run = new wxButton(scroller, wxID_ANY, "Reconstruct mesh and point cloud");
        run->SetMinSize(wxSize(-1, 40));
        outer->Add(run, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        run->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RunReconstruction(); });

        scroller->SetSizer(outer);
        page_sizer->Add(scroller, 1, wxEXPAND);
        page->SetSizer(page_sizer);
    }


    void BuildMeshPage() {
        auto* page = new wxPanel(notebook_);
        notebook_->AddPage(page, "Mesh validation & viewer");
        auto* root = new wxBoxSizer(wxVERTICAL);
        auto* intro = new wxStaticText(page, wxID_ANY,
            "Inspect an STL mesh interactively, check basic file/geometry integrity, export a validation report, or save a copy.");
        intro->Wrap(1000);
        root->Add(intro, 0, wxALL, 10);
        mesh_path_ = AddPathRow(page, root, "STL mesh to inspect", "", PickMode::Open,
            "STL mesh (*.stl)|*.stl|All files|*.*",
            [this](const wxString& path) { LoadMeshFromPath(path); });
        auto* actions = new wxBoxSizer(wxHORIZONTAL);
        auto* validate = new wxButton(page, wxID_ANY, "Load / validate mesh");
        auto* report = new wxButton(page, wxID_ANY, "Export validation JSON…");
        auto* copy = new wxButton(page, wxID_ANY, "Export STL copy…");
        actions->Add(validate, 1, wxRIGHT, 6);
        actions->Add(report, 1, wxLEFT | wxRIGHT, 6);
        actions->Add(copy, 1, wxLEFT, 6);
        root->Add(actions, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        mesh_summary_ = new wxStaticText(page, wxID_ANY, "No mesh loaded");
        mesh_summary_->Wrap(1100);
        root->Add(mesh_summary_, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        mesh_viewer_ = new cvcascade::gui::MeshViewerPanel(page);
        root->Add(mesh_viewer_, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        validate->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { LoadMeshFromPath(mesh_path_->GetValue()); });
        report->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) {
            wxFileDialog dialog(this, "Export mesh validation report", "", "mesh-validation.json",
                                "JSON report (*.json)|*.json|All files|*.*", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
            if (dialog.ShowModal() != wxID_OK) return;
            std::string error;
            if (!mesh_viewer_->ExportInspectionJson(dialog.GetPath().ToStdString(), error)) {
                wxMessageBox(wxString::FromUTF8(error), "Export failed", wxOK | wxICON_ERROR, this);
                return;
            }
            Log("Exported mesh validation report: " + dialog.GetPath());
        });
        copy->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) {
            const std::filesystem::path source(mesh_path_->GetValue().ToStdString());
            if (!std::filesystem::is_regular_file(source)) {
                wxMessageBox("Load a valid STL file first.", "No mesh", wxOK | wxICON_WARNING, this);
                return;
            }
            wxFileDialog dialog(this, "Export STL copy", "", source.filename().string(),
                                "STL mesh (*.stl)|*.stl|All files|*.*", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
            if (dialog.ShowModal() != wxID_OK) return;
            try {
                const auto destination = std::filesystem::path(dialog.GetPath().ToStdString());
                if (std::filesystem::absolute(source).lexically_normal() != std::filesystem::absolute(destination).lexically_normal())
                    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
                Log("Exported STL copy: " + destination.string());
            } catch (const std::exception& e) {
                wxMessageBox(wxString::FromUTF8(e.what()), "Export failed", wxOK | wxICON_ERROR, this);
            }
        });
        page->SetSizer(root);
    }

    void LoadMeshFromPath(const wxString& path) {
        if (path.empty() || mesh_viewer_ == nullptr) return;
        std::string error;
        if (!mesh_viewer_->LoadStl(path.ToStdString(), error)) {
            mesh_summary_->SetLabel(wxString::FromUTF8(error));
            wxMessageBox(wxString::FromUTF8(error), "Mesh validation failed", wxOK | wxICON_ERROR, this);
            return;
        }
        mesh_summary_->SetLabel(wxString::FromUTF8(mesh_viewer_->Inspection().summary));
        Layout();
        Log("Loaded and inspected STL: " + path);
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

    void RunJob(const wxString& label, std::function<std::string()> operation) {
        if (busy_) {
            wxMessageBox("Another operation is already running.", "CVCascade Studio",
                         wxOK | wxICON_INFORMATION, this);
            return;
        }
        if (worker_.joinable()) worker_.join();

        cancel_requested_.store(false, std::memory_order_relaxed);
        busy_ = true;
        cancel_button_->Enable(true);
        notebook_->Enable(false);
        gauge_->Pulse();
        status_->SetLabel(label + "…");
        Log("Started: " + label);
        timer_.Start(100);

        worker_ = std::thread([this, label, operation = std::move(operation)]() mutable {
            bool succeeded = false;
            bool cancelled = false;
            std::string result;
            try {
                if (cancel_requested_.load(std::memory_order_relaxed)) throw std::runtime_error("Task cancelled by user.");
                result = operation();
                if (cancel_requested_.load(std::memory_order_relaxed)) throw std::runtime_error("Task cancelled by user.");
                succeeded = true;
            } catch (const std::exception& e) {
                result = e.what();
                cancelled = cancel_requested_.load(std::memory_order_relaxed) || result.find("cancelled by user") != std::string::npos;
            } catch (...) {
                result = "Unknown non-standard exception.";
            }
            wxTheApp->CallAfter([this, label, succeeded, cancelled, result = std::move(result)]() {
                FinishJob(label, succeeded, cancelled, result);
            });
        });
    }

    void FinishJob(const wxString& label, bool succeeded, bool cancelled, const std::string& result) {
        if (worker_.joinable()) worker_.join();
        timer_.Stop();
        gauge_->SetValue(succeeded ? 100 : 0);
        busy_ = false;
        cancel_button_->Enable(false);
        notebook_->Enable(true);
        status_->SetLabel(succeeded ? label + " completed." : (cancelled ? label + " cancelled." : label + " failed."));
        Log((succeeded ? "Completed: " : (cancelled ? "Cancelled: " : "Failed: ")) + label);
        Log(wxString::FromUTF8(result));
        if (succeeded)
            wxMessageBox(wxString::FromUTF8(result), label + " complete",
                         wxOK | wxICON_INFORMATION, this);
        else if (!cancelled)
            wxMessageBox(wxString::FromUTF8(result), label + " failed",
                         wxOK | wxICON_ERROR, this);
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

        RunJob("Photo relief", [input, output, options, threads]() {
#ifdef CVCASCADE_HAS_OPENMP
            if (threads > 0) omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            const cv::Mat image = cv::imread(input, cv::IMREAD_GRAYSCALE);
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
        RunJob("Camera calibration", [images, output, cols, rows, square]() {
            const auto model = cvcascade::calibrate_camera_from_list(images, output, cols, rows, square);
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
        options.cancellation_flag = &cancel_requested_;

        RunJob("Multi-view reconstruction", [images, calibration, stl, ply, report, options]() {
            const auto stats = cvcascade::reconstruct_multiview(images, calibration, stl, options, ply);
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
            return text.str();
        });
    }


    void WriteSettings(cv::FileStorage& fs, bool preset_only) {
        fs << "schema_version" << 1;
        if (!preset_only) {
            fs << "relief_input" << relief_input_->GetValue().ToStdString()
               << "relief_output" << relief_output_->GetValue().ToStdString()
               << "relief_width" << relief_width_->GetValue()
               << "relief_height" << relief_height_->GetValue()
               << "relief_dimension" << relief_dimension_->GetValue()
               << "relief_threads" << relief_threads_->GetValue()
               << "relief_invert" << static_cast<int>(relief_invert_->GetValue())
               << "calibration_images" << calibration_images_->GetValue().ToStdString()
               << "calibration_output" << calibration_output_->GetValue().ToStdString()
               << "board_cols" << board_cols_->GetValue()
               << "board_rows" << board_rows_->GetValue()
               << "square_size" << square_size_->GetValue();
        }
        fs << "reconstruction_images" << reconstruction_images_->GetValue().ToStdString()
           << "reconstruction_calibration" << reconstruction_calibration_->GetValue().ToStdString()
           << "reconstruction_stl" << reconstruction_stl_->GetValue().ToStdString()
           << "reconstruction_ply" << reconstruction_ply_->GetValue().ToStdString()
           << "reconstruction_report" << reconstruction_report_->GetValue().ToStdString()
           << "mesh_path" << mesh_path_->GetValue().ToStdString()
           << "reconstruction_dimension" << reconstruction_dimension_->GetValue()
           << "disparities" << disparities_->GetValue()
           << "block_size" << block_size_->GetValue()
           << "min_inliers" << min_inliers_->GetValue()
           << "baseline_mm" << baseline_mm_->GetValue()
           << "max_depth" << max_depth_->GetValue()
           << "depth_jump" << depth_jump_->GetValue()
           << "lr_tolerance" << lr_tolerance_->GetValue()
           << "fusion_voxel" << fusion_voxel_->GetValue()
           << "speckle_window" << speckle_window_->GetValue()
           << "speckle_range" << speckle_range_->GetValue();
    }

    void ReadSettings(const cv::FileStorage& fs, bool preset_only) {
        auto string_value = [&](const char* key, wxTextCtrl* control) {
            const auto node = fs[key];
            if (!node.empty()) { std::string value; node >> value; control->SetValue(wxString::FromUTF8(value)); }
        };
        auto int_value = [&](const char* key, wxSpinCtrl* control) {
            const auto node = fs[key];
            if (!node.empty()) { int value = control->GetValue(); node >> value; control->SetValue(value); }
        };
        auto double_value = [&](const char* key, wxSpinCtrlDouble* control) {
            const auto node = fs[key];
            if (!node.empty()) { double value = control->GetValue(); node >> value; control->SetValue(value); }
        };
        if (!preset_only) {
            string_value("relief_input", relief_input_);
            string_value("relief_output", relief_output_);
            double_value("relief_width", relief_width_);
            double_value("relief_height", relief_height_);
            int_value("relief_dimension", relief_dimension_);
            int_value("relief_threads", relief_threads_);
            const auto invert = fs["relief_invert"];
            if (!invert.empty()) { int value = 0; invert >> value; relief_invert_->SetValue(value != 0); }
            string_value("calibration_images", calibration_images_);
            string_value("calibration_output", calibration_output_);
            int_value("board_cols", board_cols_);
            int_value("board_rows", board_rows_);
            double_value("square_size", square_size_);
        }
        string_value("reconstruction_images", reconstruction_images_);
        string_value("reconstruction_calibration", reconstruction_calibration_);
        string_value("reconstruction_stl", reconstruction_stl_);
        string_value("reconstruction_ply", reconstruction_ply_);
        string_value("reconstruction_report", reconstruction_report_);
        string_value("mesh_path", mesh_path_);
        int_value("reconstruction_dimension", reconstruction_dimension_);
        int_value("disparities", disparities_);
        int_value("block_size", block_size_);
        int_value("min_inliers", min_inliers_);
        double_value("baseline_mm", baseline_mm_);
        double_value("max_depth", max_depth_);
        double_value("depth_jump", depth_jump_);
        double_value("lr_tolerance", lr_tolerance_);
        double_value("fusion_voxel", fusion_voxel_);
        int_value("speckle_window", speckle_window_);
        int_value("speckle_range", speckle_range_);
    }

    void SaveSettingsDialog(bool preset_only) {
        wxFileDialog dialog(this, preset_only ? "Save reconstruction preset" : "Save project",
                            "", preset_only ? "reconstruction-preset.yml" : "project.cvcascade.yml",
                            "CVCascade YAML (*.yml;*.yaml)|*.yml;*.yaml|All files|*.*",
                            wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() != wxID_OK) return;
        try {
            cv::FileStorage fs(dialog.GetPath().ToStdString(), cv::FileStorage::WRITE);
            if (!fs.isOpened()) throw std::runtime_error("Cannot open settings file for writing.");
            WriteSettings(fs, preset_only);
            fs.release();
            Log(std::string(preset_only ? "Saved preset: " : "Saved project: ") + dialog.GetPath().ToStdString());
        } catch (const std::exception& e) {
            wxMessageBox(wxString::FromUTF8(e.what()), "Save failed", wxOK | wxICON_ERROR, this);
        }
    }

    void LoadSettingsDialog(bool preset_only) {
        wxFileDialog dialog(this, preset_only ? "Load reconstruction preset" : "Open project",
                            "", "", "CVCascade YAML (*.yml;*.yaml)|*.yml;*.yaml|All files|*.*",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() != wxID_OK) return;
        try {
            cv::FileStorage fs(dialog.GetPath().ToStdString(), cv::FileStorage::READ);
            if (!fs.isOpened()) throw std::runtime_error("Cannot open settings file.");
            int version = 0;
            fs["schema_version"] >> version;
            if (version != 1) throw std::runtime_error("Unsupported or missing project schema_version.");
            ReadSettings(fs, preset_only);
            if (!preset_only && !relief_input_->GetValue().empty()) UpdatePreview(relief_input_->GetValue());
            if (!preset_only && !mesh_path_->GetValue().empty()) LoadMeshFromPath(mesh_path_->GetValue());
            Log(std::string(preset_only ? "Loaded preset: " : "Opened project: ") + dialog.GetPath().ToStdString());
        } catch (const std::exception& e) {
            wxMessageBox(wxString::FromUTF8(e.what()), "Load failed", wxOK | wxICON_ERROR, this);
        }
    }

    void SaveProject() { SaveSettingsDialog(false); }
    void OpenProject() { LoadSettingsDialog(false); }
    void SavePreset() { SaveSettingsDialog(true); }
    void LoadPreset() { LoadSettingsDialog(true); }

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
