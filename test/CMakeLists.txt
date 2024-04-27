# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(cfdTests LANGUAGES CXX)

# ---- Dependencies ----

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(cfd_test source/cfd_test.cxx)
target_link_libraries(
    cfd_test PRIVATE
    cfd_lib
    Catch2::Catch2WithMain
)
target_compile_features(cfd_test PRIVATE cxx_std_17)

catch_discover_tests(cfd_test)

# ---- End-of-file commands ----

add_folders(Test)
