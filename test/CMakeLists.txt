cmake_minimum_required(VERSION 3.14)

project(cfdTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(cfd REQUIRED)
  enable_testing()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(cfd_test source/cfd_test.cxx)
target_link_libraries(
    cfd_test PRIVATE
    cfd::cfd
    Catch2::Catch2WithMain
)
target_compile_features(cfd_test PRIVATE cxx_std_17)

catch_discover_tests(cfd_test)

# ---- End-of-file commands ----

add_folders(Test)
