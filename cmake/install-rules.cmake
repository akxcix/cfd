if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/cfd-${PROJECT_VERSION}"
      CACHE STRING ""
  )
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package cfd)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT cfd_Development
)

install(
    TARGETS cfd_cfd
    EXPORT cfdTargets
    RUNTIME #
    COMPONENT cfd_Runtime
    LIBRARY #
    COMPONENT cfd_Runtime
    NAMELINK_COMPONENT cfd_Development
    ARCHIVE #
    COMPONENT cfd_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    cfd_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE cfd_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(cfd_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${cfd_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT cfd_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${cfd_INSTALL_CMAKEDIR}"
    COMPONENT cfd_Development
)

install(
    EXPORT cfdTargets
    NAMESPACE cfd::
    DESTINATION "${cfd_INSTALL_CMAKEDIR}"
    COMPONENT cfd_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
