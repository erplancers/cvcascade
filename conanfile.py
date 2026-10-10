from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class CvcascadeRecipe(ConanFile):
    name = "cvcascade"
    version = "0.3.0"
    package_type = "application"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_openmp": [True, False],
        "with_gui": [True, False],
        "with_tests": [True, False],
    }
    default_options = {
        "with_openmp": True,
        "with_gui": True,
        "with_tests": True,
    }
    exports_sources = "CMakeLists.txt", "include/**", "src/**", "tests/**", "README.md"

    def requirements(self):
        # Keep the modules used by CVCascade and avoid optional media/GUI backends.
        # JPEG and PNG support remain enabled for normal image input.
        self.requires("opencv/4.14.0", options={
            "with_ipp": False,
            "with_eigen": False,
            "with_opencl": False,
            "with_cuda": False,
            "with_cublas": False,
            "with_cufft": False,
            "with_cudnn": False,
            "world": False,
            "nonfree": False,
            "with_flatbuffers": False,
            "with_protobuf": False,
            "with_vulkan": False,
            "with_openvino": False,
            "dnn_cuda": False,
            "with_jpeg": "libjpeg-turbo",
            "with_png": True,
            "with_tiff": False,
            "with_jpeg2000": False,
            "with_openexr": False,
            "with_webp": False,
            "with_avif": False,
            "with_gdal": False,
            "with_gdcm": False,
            "with_quirc": False,
            "with_ffmpeg": False,
            "with_tesseract": False,
            "dnn": False,
            "highgui": False,
            "videoio": False,
            "photo": False,
            "objdetect": False,
            "video": False,
            "stitching": False,
            "ml": False,
            "gapi": False,
        })
        # Only modeling and STL toolkits are needed from OCCT.
        self.requires("opencascade/7.9.1", options={
            "with_ffmpeg": False,
            "with_freeimage": False,
            "with_openvr": False,
            "with_rapidjson": False,
            "with_draco": False,
            "with_tk": False,
            "with_tbb": False,
            "with_opengl": False,
        })
        if self.options.with_gui:
            self.requires("wxwidgets/3.2.11")

    def build_requirements(self):
        self.tool_requires("cmake/3.30.5")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CVCASCADE_ENABLE_OPENMP"] = bool(self.options.with_openmp)
        tc.variables["CVCASCADE_BUILD_GUI"] = bool(self.options.with_gui)
        tc.variables["CVCASCADE_BUILD_TESTS"] = bool(self.options.with_tests)
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def test(self):
        if self.options.with_tests:
            CMake(self).test()
