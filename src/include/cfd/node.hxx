#pragma once

#include <string>

class Cell {
  public:
    /**
     * @brief Initializes the name field to the name of the project
     */
    Cell();

    /**
     * @brief Returns a non-owning pointer to the string stored in this class
     */
    auto name() const -> char const*;

  private:
    std::string m_name;
};
