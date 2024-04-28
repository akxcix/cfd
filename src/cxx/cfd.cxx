#include <string>

#include "cfd/cfd.hxx"

#include <fmt/core.h>

Cfd::Cfd()
    : m_name {fmt::format("{}", "cfd")} {}

const char* Cfd::name() const {
    return m_name.c_str();
}
