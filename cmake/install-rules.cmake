install(
    TARGETS cfd_exe
    RUNTIME COMPONENT cfd_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
