# cvcascade

A C++20 command-line photogrammetry prototype using **OpenCV** for image processing, camera calibration, feature matching and dense stereo, and **Open CASCADE Technology (OCCT)** for triangulated geometry and STL export. OpenMP remains optional for the original image-relief mesh builder. Builds use CMake, Conan 2 and CTest.

## Modes

- **Single-image relief:** maps grayscale brightness to height. Useful for bas-reliefs and lithophanes, not full-object reconstruction.
- **Camera calibration:** estimates intrinsics and lens distortion from chessboard images.
- **Multi-view reconstruction:** matches ORB features between consecutive images, uses RANSAC essential-matrix estimation and pose recovery, rectifies each pair, computes a semi-global block-matching (SGBM) disparity map, triangulates depth and exports the pairwise meshes through OCCT.

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

If Conan reports another generated toolchain path, use the path printed by Conan. OCCT is installed as a system development dependency on the CI image; package names can differ by Linux distribution.

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

Use images from the **same camera and resolution**. Capture at least 5-15 varied views of a printed chessboard with the board at different distances and tilts. Board dimensions are the number of **inner corners**, not the number of squares.

```bash
./build/cvcascade --calibrate \
  --images calibration-images.txt \
  --calibration camera.yml \
  --board-cols 9 --board-rows 6 --square-size-mm 25
```

The calibration file stores the camera matrix, distortion coefficients, image dimensions and reprojection RMS. Use your board's actual inner-corner layout and square size.

## 3. Multi-view photogrammetry and dense stereo

Capture a sequence that moves around the object while keeping **substantial overlap (typically 60-80%)**, stable focus/exposure and visible surface texture. Use the calibrated camera and keep image dimensions/aspect ratio consistent. List the images in capture order in `views.txt`:

```text
photos/view_001.jpg
photos/view_002.jpg
photos/view_003.jpg
photos/view_004.jpg
photos/view_005.jpg
```

Then reconstruct:

```bash
./build/cvcascade --reconstruct \
  --images views.txt --calibration camera.yml --output reconstruction.stl \
  --max-dimension 640 --num-disparities 128 --block-size 5 --min-inliers 30
```

Important options:

- `--num-disparities N`: SGBM search range; must be a multiple of 16 and smaller than the image width.
- `--block-size N`: odd matching-window size, at least 3.
- `--min-inliers N`: minimum geometrically consistent feature matches for each adjacent pair.
- `--max-depth MM`: reject points deeper than this value in the pair's depth frame.
- `--depth-jump RATIO`: reject triangles crossing a large relative depth discontinuity (default 0.20).
- `--baseline-mm MM`: set the assumed translation length for **each adjacent pair**. If omitted, monocular pose recovery leaves the reconstruction at arbitrary scale.

The pipeline is deliberately an initial incremental SfM + pairwise MVS implementation: it chains relative poses and appends valid dense stereo surfaces into an OCCT triangulated compound. It does **not** yet include global bundle adjustment, loop closure, globally fused depth maps, texture baking, or watertight surface repair. Pairwise poses are approximate; using `--baseline-mm` is only metrically meaningful when the assumed consecutive baseline matches the capture setup. For unconstrained handheld sequences, scale and accumulated drift will remain uncertain.

## Tests

CTest covers brightness-to-height mapping, inversion, dimension limits, CLI help, image-list parsing, and photogrammetry argument/file validation. CI configures a Conan profile before resolving dependencies, then builds and runs CTest.
