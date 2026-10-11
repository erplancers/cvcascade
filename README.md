# cvcascade

A C++20 command-line photogrammetry prototype using **OpenCV** for image processing, camera calibration, feature matching and dense stereo, and **Open CASCADE Technology (OCCT)** for triangulated geometry and STL export. OpenMP is optional for the relief mesh builder. Includes a wxWidgets desktop workbench as well as the CLI. Builds use CMake, Conan 2 and CTest.

## Capabilities

- **Single-image relief:** brightness-driven height mapping for bas-reliefs and lithophanes.
- **Camera calibration:** chessboard-based intrinsics and lens-distortion calibration, with the SB detector as a fallback and robust median/MAD rejection of grossly inconsistent views.
- **Incremental multi-view reconstruction:** ORB feature matching, RANSAC essential-matrix estimation, relative pose recovery and chained camera poses.
- **Bidirectional stereo consistency:** computes forward and reverse SGBM disparity and rejects ambiguous matches.
- **Speckle/outlier suppression:** disparity speckle filtering, max-depth rejection and relative depth discontinuity rejection before surface triangulation.
- **Dense colored point cloud:** optional RGB PLY export from the rectified source images.
- **Voxel-grid fusion:** deterministic voxel downsampling that averages point coordinates and colors to reduce duplicate samples across adjacent stereo pairs.
- **wxWidgets desktop GUI:** graphical workflows for relief generation, camera calibration, and reconstruction with browse dialogs, image preview, adjustable parameters, background execution, progress indicator, logs, and result summaries.
- **Reconstruction quality report:** console metrics plus optional machine-readable JSON output (`--report report.json`) with feature matches/inliers, pose inlier ratio, valid-depth ratio, stereo consistency counts, mesh size, and scale status.

## Dependencies and cross-platform build

The C++ libraries and CMake build tool are resolved through **Conan 2 / Conan Center**: OpenCV 4.14, Open CASCADE 7.9, wxWidgets 3.2 for the desktop GUI, and CMake. On Linux, wxWidgets' GTK/X11 backend also relies on native graphics/window-system SDK components that Conan Center models with `system` recipes. CI lets Conan install any missing host SDK packages through those recipes; this requires elevated privileges on Linux, so the Linux CI commands run Conan with `sudo`. No manual OS-package list is maintained in CVCascade.

You need Python 3.9+, Conan 2, and a native C++20 compiler/SDK for your operating system (for example, Visual Studio Build Tools on Windows or Xcode Command Line Tools on macOS). The compiler and OS SDK are host toolchain prerequisites; project C++ libraries and the CMake build tool are resolved by Conan.

### Build and test through Conan

```bash
python -m pip install --upgrade "conan>=2.0,<3.0"
conan profile detect --force
conan create . --build=missing -s build_type=Release -c tools.system.package_manager:mode=install
```

`conan create` downloads/builds the required Conan packages, configures and compiles CVCascade, installs the package into the Conan cache, and runs CTest. The default build includes the wxWidgets GUI and all tests.

### Build a local working tree

```bash
conan profile detect --force
conan install . --output-folder=build --build=missing -s build_type=Release -c tools.system.package_manager:mode=install
```

Activate Conan's generated build environment so the CMake executable supplied by Conan is on `PATH`:

- Linux/macOS: `source build/build/Release/generators/conanbuild.sh`
- Windows Command Prompt: `call build\\build\\Release\\generators\\conanbuild.bat`
- Windows PowerShell: `. ./build/build/Release/generators/conanbuild.ps1`

Then configure, compile, test, and install:

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=build/build/Release/generators/conan_toolchain.cmake \
  -DCVCASCADE_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
cmake --install build --config Release --prefix install
```

The installed executables are in `install/bin` (Windows adds the `.exe` extension). To build the CLI without the desktop UI, use a separate output folder and pass `-o "cvcascade/*:with_gui=False"` to `conan install`; wxWidgets is then not required.

## 1. Single-image relief

```bash
./build/cvcascade --input photo.jpg --output relief.stl
./build/cvcascade --input portrait.png --output portrait-relief.stl --width 120 --max-height 8
./build/cvcascade --input heightmap.png --output inverted.stl --invert --max-dimension 768 --threads 8
```

Image intensity maps linearly to Z. The output is an open relief surface, not a watertight printable solid.

## 2. Calibrate the camera

Make a text file with one calibration-image path per line. Paths may be absolute or relative to the list file; blank lines and lines beginning with `#` are ignored.

Example `calibration-images.txt`:

```text
calibration/board_01.jpg
calibration/board_02.jpg
calibration/board_03.jpg
calibration/board_04.jpg
```

Use images from the **same camera and resolution**. Capture 5-15 varied views of a printed chessboard at different distances and tilts. Board dimensions are the number of **inner corners**, not squares.

```bash
./build/cvcascade --calibrate \
  --images calibration-images.txt \
  --calibration camera.yml \
  --board-cols 9 --board-rows 6 --square-size-mm 25
```

The calibration file stores the camera matrix, distortion coefficients, image dimensions, reprojection RMS, and counts of accepted/rejected calibration views. CVCascade tries OpenCV's more robust SB chessboard detector first, then falls back to the classic detector. After an initial fit, it rejects gross per-view reprojection outliers using a median/MAD threshold when at least three views can be retained. Use your board's actual inner-corner layout and square size.

## 3. Multi-view photogrammetry, dense stereo and fusion

Capture a sequence around the object with **substantial overlap (typically 60-80%)**, stable focus/exposure and visible surface texture. Keep image dimensions/aspect ratio consistent and use the same camera settings. List images in capture order in `views.txt`:

```text
photos/view_001.jpg
photos/view_002.jpg
photos/view_003.jpg
photos/view_004.jpg
photos/view_005.jpg
```

Export both a mesh and a colored, voxel-fused PLY point cloud:

```bash
./build/cvcascade --reconstruct \
  --images views.txt --calibration camera.yml \
  --output reconstruction.stl --point-cloud reconstruction.ply \
  --max-dimension 640 --num-disparities 128 --block-size 5 \
  --min-inliers 30 --lr-tolerance 1.5 \
  --speckle-window 100 --speckle-range 2 \
  --fusion-voxel 2.0 --depth-jump 0.20 --report reconstruction-report.json
```

The console report includes the number of feature matches and pose inliers, mean inlier ratio, average valid-depth ratio, left-right stereo consistency, rejected stereo candidates, mesh vertices/triangles and fused point count. `--report reconstruction-report.json` writes these metrics and the output paths to JSON for automated quality tracking.

### Reconstruction options

- `--num-disparities N`: SGBM search range; a positive multiple of 16 and smaller than image width.
- `--block-size N`: odd matching-window size, at least 3.
- `--min-inliers N`: minimum geometrically consistent feature matches per adjacent pair.
- `--lr-tolerance PX`: allowed forward/reverse disparity disagreement; lower is stricter, but can remove more points.
- `--speckle-window N`: remove disparity islands smaller than this size; set 0 to disable speckle removal.
- `--speckle-range N`: disparity difference threshold for speckle filtering.
- `--max-depth N`: reject points deeper than this value in the pair's depth frame. Values are only in millimetres when a valid metric baseline is used.
- `--depth-jump RATIO`: reject triangles that cross a large relative depth discontinuity (default 0.20).
- `--baseline-mm MM`: assumed translation length for each adjacent image pair. Without it, monocular pose recovery produces arbitrary scale.
- `--point-cloud cloud.ply`: additionally export an ASCII PLY with RGB colors.
- `--fusion-voxel N`: voxel size for point-cloud downsampling/fusion. Units are millimetres only when the supplied baseline genuinely defines metric scale; otherwise they are arbitrary model units.

The pipeline remains an incremental SfM + pairwise MVS prototype. It chains relative poses and appends pairwise triangulated surfaces into an OCCT compound. Voxel fusion currently applies to the optional point cloud, not to the STL mesh. It does **not** yet include global bundle adjustment, loop closure, global depth-map fusion, texture baking, or watertight mesh repair. Pairwise poses can accumulate drift; setting `--baseline-mm` is metrically meaningful only when each adjacent pair really uses that baseline. Glossy, textureless, moving, or poorly lit scenes remain difficult.

## Tests

CTest covers relief mapping and STL export, image-list parsing, calibration/CLI validation, voxel fusion/color averaging, invalid voxel sizes, and PLY header/content export. CI configures a Conan profile before resolving dependencies, then builds and runs CTest.
