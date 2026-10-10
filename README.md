# cvcascade

A C++ command-line toolkit for turning a single photograph into a scaled 3D height-field mesh. OpenCV handles image preprocessing and sampling; Open CASCADE (OCCT) builds and exports the CAD mesh.

> A single 2D photograph does not uniquely determine real-world depth. The initial reconstruction mode interprets image intensity (or its inverse) as height; it is useful for reliefs, bas-reliefs, terrain-like meshes, and image-derived surfaces rather than metrically accurate object reconstruction.

The implementation, CMake/Conan build, automated tests, and CI are being added on a feature branch.