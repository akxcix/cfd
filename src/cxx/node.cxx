#include <string>

#include "cfd/node.hxx"

#include <fmt/core.h>

Cell::Cell() {}

auto Cell::name() const -> const char* {
    return "Cell";
}
