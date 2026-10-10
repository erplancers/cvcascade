#include "cvcascade/mesh_builder.hpp"
#include "cvcascade/photogrammetry.hpp"

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/filedlg.h>
#include <wx/gauge.h>
#include <wx/statbmp.h>
#include <wx/spinctrl.h>
#include <wx/timer.h>

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cmath>
#include <functional>
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
        auto* outer = new wxBoxSizer(wxVERTICAL);
        auto* intro = new wxStaticText(page, wxID_ANY,
            "Incremental SfM + pairwise stereo. Use ordered, overlapping photographs from the calibrated camera.");
        intro->Wrap(1000);
        outer->Add(intro, 0, wxALL, 10);

        auto* files = new wxStaticBoxSizer(wxVERTICAL, page, "Inputs and outputs");
        reconstruction_images_ = AddPathRow(page, files, "Image manifest (.txt)", "views.txt",
            PickMode::Open, "Text manifest (*.txt)|*.txt|All files|*.*");
        reconstruction_calibration_ = AddPathRow(page, files, "Camera calibration", "camera.yml",
            PickMode::Open, "Calibration files (*.yml;*.yaml;*.xml)|*.yml;*.yaml;*.xml|All files|*.*");
        reconstruction_stl_ = AddPathRow(page, files, "Output mesh (.stl)", "reconstruction.stl",
            PickMode::Save, "STL mesh (*.stl)|*.stl|All files|*.*");
        reconstruction_ply_ = AddPathRow(page, files, "Optional RGB point cloud (.ply)", "reconstruction.ply",
            PickMode::Save, "PLY point cloud (*.ply)|*.ply|All files|*.*");
        outer->Add(files, 0, wxEXPAND | wxLEFT | wxRIGHT, 8);

        auto* settings = new wxStaticBoxSizer(wxVERTICAL, page, "Stereo, filtering and fusion");
        auto* grid = new wxFlexGridSizer(2, 8, 12);
        grid->AddGrowableCol(1, 1);
        auto add_grid = [&](const wxString& label, wxWindow* control) {
            grid->Add(new wxStaticText(page, wxID_ANY, label), 0, wxALIGN_CENTER_VERTICAL);
            grid->Add(control, 1, wxEXPAND);
        };

        reconstruction_dimension_ = MakeInt(page, 640, 64, 8192, 64);
        disparities_ = MakeInt(page, 128, 16, 4096, 16);
        block_size_ = MakeInt(page, 5, 3, 51, 2);
        min_inliers_ = MakeInt(page, 30, 8, 100000);
        baseline_mm_ = MakeDouble(page, 0.0, 0.0, 1000000000.0, 1.0, 3);
        max_depth_ = MakeDouble(page, 10000.0, 0.001, 1000000000.0, 100.0, 3);
        depth_jump_ = MakeDouble(page, 0.20, 0.001, 1.0, 0.01, 3);
        lr_tolerance_ = MakeDouble(page, 1.5, 0.1, 1000.0, 0.1, 2);
        speckle_window_ = MakeInt(page, 100, 0, 100000);
        speckle_range_ = MakeInt(page, 2, 0, 10000);
        fusion_voxel_ = MakeDouble(page, 2.0, 0.001, 1000000000.0, 0.5, 3);

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

        auto* note = new wxStaticText(page, wxID_ANY,
            "Without a known baseline, scale and voxel size are arbitrary. Voxel fusion affects PLY; "
            "the STL retains per-pair surfaces. Global bundle adjustment and watertight repair are not implemented.");
        note->Wrap(1000);
        outer->Add(note, 0, wxLEFT | wxRIGHT | wxBOTTOM, 10);
        auto* run = new wxButton(page, wxID_ANY, "Reconstruct mesh and point cloud");
        run->SetMinSize(wxSize(-1, 40));
        outer->Add(run, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        run->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { RunReconstruction(); });

        auto* scroller = new wxScrolledWindow(page, wxID_ANY, wxDefaultPosition, wxDefaultSize,
                                              wxVSCROLL);
        scroller->SetScrollRate(0, 14);
        scroller->SetSizer(outer);
        auto* page_sizer = new wxBoxSizer(wxVERTICAL);
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

    void RunJob(const wxString& label, std::function<std::string()> operation) {
        if (busy_) {
            wxMessageBox("Another operation is already running.", "CVCascade Studio",
                         wxOK | wxICON_INFORMATION, this);
            return;
        }
        if (worker_.joinable()) worker_.join();

        busy_ = true;
        notebook_->Enable(false);
        gauge_->Pulse();
        status_->SetLabel(label + "…");
        Log("Started: " + label);
        timer_.Start(100);

        worker_ = std::thread([this, label, operation = std::move(operation)]() mutable {
            bool succeeded = false;
            std::string result;
            try {
                result = operation();
                succeeded = true;
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
        notebook_->Enable(true);
        status_->SetLabel(succeeded ? label + " completed." : label + " failed.");
        Log((succeeded ? "Completed: " : "Failed: ") + label);
        Log(wxString::FromUTF8(result));
        if (succeeded)
            wxMessageBox(wxString::FromUTF8(result), label + " complete",
                         wxOK | wxICON_INFORMATION, this);
        else
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
        if (images.empty() || calibration.empty() || stl.empty()) {
            wxMessageBox("Choose the image manifest, camera calibration, and output STL paths.",
                         "Missing input", wxOK | wxICON_WARNING, this);
            return;
        }
        if (!ply.empty() && std::filesystem::path(ply).lexically_normal() ==
                            std::filesystem::path(stl).lexically_normal()) {
            wxMessageBox("STL and PLY output paths must be different.", "Invalid output paths",
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

        RunJob("Multi-view reconstruction", [images, calibration, stl, ply, options]() {
            const auto stats = cvcascade::reconstruct_multiview(images, calibration, stl, options, ply);
            const double consistent = stats.stereo_candidates > 0
                ? 100.0 * static_cast<double>(stats.stereo_consistent) / stats.stereo_candidates : 0.0;
            std::ostringstream text;
            text << "Mesh saved: " << stl
                 << "\nImages: " << stats.images
                 << "\nPairs with triangles: " << stats.registered_pairs
                 << "\nFeature matches: " << stats.feature_matches
                 << "\nPose inliers: " << stats.pose_inliers
                 << "\nMean pose inlier ratio: " << stats.mean_pose_inlier_ratio
                 << "\nMean valid-depth ratio: " << stats.mean_valid_depth_ratio
                 << "\nStereo consistency: " << consistent << "%"
                 << "\nStereo candidates / consistent / rejected: "
                 << stats.stereo_candidates << " / " << stats.stereo_consistent << " / " << stats.stereo_rejected
                 << "\nMesh vertices / triangles: " << stats.vertices << " / " << stats.triangles
                 << "\nScale: " << (stats.metric_scale ? "baseline-scaled (approximate)" : "arbitrary")
                 << "\n";
            if (!ply.empty()) text << "Fused RGB PLY: " << ply << "\nPoints: " << stats.fused_points;
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
