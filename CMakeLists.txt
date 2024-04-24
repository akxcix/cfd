cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    cfd
    VERSION 0.1.0
    DESCRIPTION "Automated Airfoil Design"
    HOMEPAGE_URL "cfd.com"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    cfd_lib OBJECT
    source/lib.cxx
)

target_include_directories(
    cfd_lib ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(cfd_lib PUBLIC cxx_std_17)

find_package(fmt REQUIRED)
target_link_libraries(cfd_lib PRIVATE fmt::fmt)

# ---- Declare executable ----

add_executable(cfd_exe source/main.cxx)
add_executable(cfd::exe ALIAS cfd_exe)

set_property(TARGET cfd_exe PROPERTY OUTPUT_NAME cfd)

target_compile_features(cfd_exe PRIVATE cxx_std_17)

target_link_libraries(cfd_exe PRIVATE cfd_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
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
