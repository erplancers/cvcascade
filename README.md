# cvcascade

A C++20 command-line photogrammetry prototype using **OpenCV** for image processing, camera calibration, feature matching and dense stereo, and **Open CASCADE Technology (OCCT)** for triangulated geometry and STL export. OpenMP is optional for the relief mesh builder. Builds use CMake, Conan 2 and CTest.

## Capabilities

- **Single-image relief:** brightness-driven height mapping for bas-reliefs and lithophanes.
- **Camera calibration:** chessboard-based intrinsics and lens-distortion calibration.
- **Incremental multi-view reconstruction:** ORB feature matching, RANSAC essential-matrix estimation, relative pose recovery and chained camera poses.
- **Bidirectional stereo consistency:** computes forward and reverse SGBM disparity and rejects ambiguous matches.
- **Speckle/outlier suppression:** disparity speckle filtering, max-depth rejection and relative depth discontinuity rejection before surface triangulation.
- **Dense colored point cloud:** optional RGB PLY export from the rectified source images.
- **Voxel-grid fusion:** deterministic voxel downsampling that averages point coordinates and colors to reduce duplicate samples across adjacent stereo pairs.
- **Reconstruction quality report:** reports feature matches/inliers, pose inlier ratio, valid-depth ratio, stereo consistency counts and mesh size.

## Dependencies and build (Linux)

Requires a C++20 compiler, CMake 3.24+, Conan 2.x, OpenCV modules core/imgcodecs/imgproc/calib3d/features2d, and OCCT development libraries.

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build libocct-foundation-dev libocct-modeling-data-dev libocct-modeling-algorithms-dev libocct-data-exchange-dev
python -m pip install "conan>=2.0,<3.0"
conan profile detect --force
conan install . --output-folder=build --build=missing -s build_type=Release
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/build/Release/generators/conan_toolchain.cmake \
  -DCVCASCADE_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

If Conan reports another generated toolchain path, use the path printed by Conan. CI installs OCCT and OpenCV from Ubuntu packages to avoid building unrelated optional GUI/media dependencies through Conan. The Conan recipe still supports Conan-managed OpenCV for normal builds.

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

The calibration file stores the camera matrix, distortion coefficients, image dimensions and reprojection RMS. Use your board's actual inner-corner layout and square size.

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
  --fusion-voxel 2.0 --depth-jump 0.20
```

The console report includes the number of feature matches and pose inliers, mean inlier ratio, average valid-depth ratio, left-right stereo consistency, rejected stereo candidates, mesh vertices/triangles and fused point count.

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
