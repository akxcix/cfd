cmake_minimum_required(VERSION 3.22)

project(
    cfd
    VERSION 0.1.0
    DESCRIPTION "Automatic Airfoil Discovery"
    HOMEPAGE_URL "cfd.com"
    LANGUAGES CXX
)

# ---- Declare library ----

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

file(GLOB cxx_source "src/cxx/*.cxx")


# pybind11 method:
pybind11_add_module(_cfd ${cxx_source})
target_compile_definitions(_cfd
                           PRIVATE VERSION_INFO=${VERSION_INFO})

add_library(cfd::cfd ALIAS _cfd)

set_target_properties(
    _cfd PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME _cfd
    OUTPUT_NAME _cfd
)

target_include_directories(
    _cfd ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>"
)

target_include_directories(
    _cfd SYSTEM
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(_cfd PUBLIC cxx_std_17)

find_package(fmt REQUIRED)
target_link_libraries(_cfd PRIVATE fmt::fmt)
