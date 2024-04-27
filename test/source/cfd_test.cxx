#include <string>

#include "cfd/cfd.hxx"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Name is cfd", "[library]") {
  auto const exported = exported_class {};
  REQUIRE(std::string("cfd") == exported.name());
}
