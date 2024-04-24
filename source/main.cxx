#include <iostream>
#include <string>

#include "lib.hxx"

auto main() -> int {
  auto const lib = Library {};
  auto const message = "Hello from " + lib.name + "!";
  std::cout << message << '\n';
  return 0;
}
