#include <string>

#include "cfd/cfd.hxx"

#include <fmt/core.h>

Cfd::Cfd()
    : m_name {fmt::format("{}", "cfd")} {}

auto Cfd::name() const -> char const* {
    return m_name.c_str();
}
