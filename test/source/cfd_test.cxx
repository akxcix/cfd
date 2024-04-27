#include <catch2/catch_test_macros.hxx>

#include "lib.hxx"

TEST_CASE("Name is cfd", "[library]") {
  auto const lib = library {};
  REQUIRE(lib.name == "cfd");
}
