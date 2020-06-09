add_executable(helmholtz_CU helmholtz_CU.cpp)
target_link_libraries(helmholtz_CU ${LINK_LIBS})
install(TARGETS helmholtz_CU RUNTIME DESTINATION bin)
