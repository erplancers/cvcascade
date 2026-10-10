cmake_minimum_required(VERSION 3.24)
project(cvcascade VERSION 0.3.0 LANGUAGES CXX)

option(CVCASCADE_BUILD_TESTS "Build unit and CLI tests" ON)
option(CVCASCADE_ENABLE_OPENMP "Enable OpenMP parallel mesh construction" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc calib3d features2d)
find_package(OpenCASCADE REQUIRED)
if(CVCASCADE_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

add_library(cvcascade_core
  src/mesh_builder.cpp
  src/photogrammetry.cpp
)
target_include_directories(cvcascade_core PUBLIC include)
target_include_directories(cvcascade_core SYSTEM PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${OpenCASCADE_INCLUDE_DIR}
)
target_link_libraries(cvcascade_core PUBLIC
  ${OpenCV_LIBS}
  TKernel TKG2d TKG3d TKGeomBase TKBRep TKMesh TKSTL
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(cvcascade_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(cvcascade_core PUBLIC CVCASCADE_HAS_OPENMP=1)
endif()

add_executable(cvcascade src/main.cpp)
target_link_libraries(cvcascade PRIVATE cvcascade_core)

include(GNUInstallDirs)
install(TARGETS cvcascade RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CVCASCADE_BUILD_TESTS)
  enable_testing()
  add_executable(cvcascade_unit_tests tests/mesh_builder_tests.cpp)
  target_link_libraries(cvcascade_unit_tests PRIVATE cvcascade_core)
  add_test(NAME mesh_builder_unit COMMAND cvcascade_unit_tests)

  add_executable(cvcascade_photogrammetry_tests tests/photogrammetry_tests.cpp)
  target_link_libraries(cvcascade_photogrammetry_tests PRIVATE cvcascade_core)
  add_test(NAME photogrammetry_unit COMMAND cvcascade_photogrammetry_tests)

  add_test(NAME cli_help COMMAND cvcascade --help)
  set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "--calibrate")
endif()
