# cvcascade

A C++20 command-line toolkit for converting a photograph into a 3D relief mesh. OpenCV decodes and resamples the image; Open CASCADE creates the triangulated surface and exports STL. OpenMP is detected optionally.

**Limitation:** one 2D photograph does not uniquely encode hidden geometry or true depth. This implementation creates a brightness-driven height field, suitable for bas-reliefs, lithophanes, terrain-like meshes and image-derived surfaces—not metrically accurate full-object reconstruction.

## Features
- Grayscale preprocessing and area downsampling with configurable maximum resolution.
- Brightness-to-height mapping, optionally inverted.
- Open CASCADE triangulation and STL export.
- Width and maximum relief height in millimetres.
- CMake, Conan 2 and CTest unit/CLI tests.

## Dependencies
C++20 compiler, CMake 3.24+, Conan 2.x, OpenCV 4.x, and the Open CASCADE Technology (OCCT) development SDK. OCCT packaging varies by platform; install its development package if it is not available through Conan.

## Build on Linux
```bash
python -m pip install "conan>=2.0,<3.0"
sudo apt-get install cmake ninja-build libocct-foundation-dev libocct-modeling-data-dev libocct-modeling-algorithms-dev libocct-data-exchange-dev
conan profile detect --force
conan install . --output-folder=build --build=missing -s build_type=Release
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/build/Release/generators/conan_toolchain.cmake
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
If Conan reports a different toolchain path, use the generated path printed in its output.

## Usage
```bash
./build/cvcascade --input photo.jpg --output relief.stl
./build/cvcascade --input portrait.png --output portrait-relief.stl --width 120 --max-height 8
./build/cvcascade --input heightmap.png --output inverted.stl --invert --max-dimension 768 --threads 8
./build/cvcascade --help
```
Options: `--width MM`, `--max-height MM`, `--max-dimension N`, `--invert`, `--threads N`. Image intensity maps linearly to Z. The output is an open relief surface, not a watertight printable solid; a base and side walls are future enhancements.

## Tests
CTest checks brightness mapping, inversion, parameter validation, downsampling and the CLI help command.
