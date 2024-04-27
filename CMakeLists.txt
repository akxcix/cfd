cmake_minimum_required(VERSION 3.22)

include(cmake/prelude.cmake)

project(
    cfd
    VERSION 0.1.0
    DESCRIPTION "Automatic Airfoil Discovery"
    HOMEPAGE_URL "cfd.com"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    cfd_cfd
    source/cfd.cxx
)
add_library(cfd::cfd ALIAS cfd_cfd)

include(GenerateExportHeader)
generate_export_header(
    cfd_cfd
    BASE_NAME cfd
    EXPORT_FILE_NAME export/cfd/cfd_export.hxx
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(cfd_cfd PUBLIC CFD_STATIC_DEFINE)
endif()

set_target_properties(
    cfd_cfd PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME cfd
    OUTPUT_NAME cfd
)

target_include_directories(
    cfd_cfd ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    cfd_cfd SYSTEM
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(cfd_cfd PUBLIC cxx_std_17)

find_package(fmt REQUIRED)
target_link_libraries(cfd_cfd PRIVATE fmt::fmt)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(cfd_py source/python.cxx)
target_compile_definitions(cfd_py
                           PRIVATE VERSION_INFO=${VERSION_INFO})

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${cfd_DEVELOPER_MODE}")
  if(BUILD_EXAMPLES)
    add_subdirectory(example)
  endif()
endif()

# ---- Developer mode ----

if(NOT cfd_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of cfd"
  )
endif()

include(cmake/dev-mode.cmake)
