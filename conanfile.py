from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps


class CvcascadeRecipe(ConanFile):
    name = "cvcascade"
    version = "0.3.0"
    package_type = "application"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_openmp": [True, False],
        "with_conan_opencv": [True, False],
    }
    default_options = {
        "with_openmp": True,
        "with_conan_opencv": True,
    }
    exports_sources = "CMakeLists.txt", "include/**", "src/**", "tests/**", "README.md"

    def requirements(self):
        if self.options.with_conan_opencv:
            self.requires("opencv/4.10.0", options={
                "with_gtk": False,
                "with_ffmpeg": False,
                "with_gstreamer": False,
                "with_v4l": False,
            })

    def build_requirements(self):
        self.tool_requires("cmake/3.30.5")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CVCASCADE_ENABLE_OPENMP"] = bool(self.options.with_openmp)
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def test(self):
        CMake(self).test()
