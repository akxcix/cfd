#include <string>

#include "cfd/cfd.hxx"

#include <fmt/core.h>

exported_class::exported_class()
    : m_name {fmt::format("{}", "cfd")} {}

auto exported_class::name() const -> char const* {
  return m_name.c_str();
}
