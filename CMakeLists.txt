cmake_minimum_required(VERSION 3.24)
project(cvcascade VERSION 0.3.0 LANGUAGES CXX)

option(CVCASCADE_BUILD_TESTS "Build unit and CLI tests" ON)
option(CVCASCADE_ENABLE_OPENMP "Enable OpenMP parallel mesh construction" ON)
option(CVCASCADE_BUILD_GUI "Build the wxWidgets desktop application" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Third-party libraries are resolved by Conan and exposed through CMakeDeps.
find_package(OpenCV CONFIG REQUIRED)
find_package(OpenCASCADE CONFIG REQUIRED)

add_library(cvcascade_core
  src/mesh_builder.cpp
  src/photogrammetry.cpp
)
target_include_directories(cvcascade_core PUBLIC include)
target_link_libraries(cvcascade_core PUBLIC
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
  opencv_calib3d
  opencv_features2d
  OpenCASCADE::TKernel
  OpenCASCADE::TKG2d
  OpenCASCADE::TKG3d
  OpenCASCADE::TKGeomBase
  OpenCASCADE::TKBRep
  OpenCASCADE::TKMesh
  OpenCASCADE::TKSTL
)

if(CVCASCADE_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cvcascade_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cvcascade_core PUBLIC CVCASCADE_HAS_OPENMP=1)
  endif()
endif()

add_executable(cvcascade src/main.cpp)
target_link_libraries(cvcascade PRIVATE cvcascade_core)

set(CVCASCADE_INSTALL_TARGETS cvcascade)
if(CVCASCADE_BUILD_GUI)
  find_package(wxWidgets CONFIG REQUIRED)
  add_executable(cvcascade_gui src/gui_main.cpp)
  target_link_libraries(cvcascade_gui PRIVATE cvcascade_core wxWidgets::wxWidgets)
  list(APPEND CVCASCADE_INSTALL_TARGETS cvcascade_gui)
endif()

include(GNUInstallDirs)
install(TARGETS ${CVCASCADE_INSTALL_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CVCASCADE_BUILD_TESTS)
  enable_testing()
  add_executable(cvcascade_unit_tests tests/mesh_builder_tests.cpp)
  target_link_libraries(cvcascade_unit_tests PRIVATE cvcascade_core)
  add_test(NAME mesh_builder_unit COMMAND cvcascade_unit_tests)

  add_executable(cvcascade_photogrammetry_tests tests/photogrammetry_tests.cpp)
  target_link_libraries(cvcascade_photogrammetry_tests PRIVATE cvcascade_core)
  add_test(NAME photogrammetry_unit COMMAND cvcascade_photogrammetry_tests)

  add_test(NAME cli_help COMMAND cvcascade --help)
  set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "--point-cloud")
endif()
