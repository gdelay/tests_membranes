add_executable(membranes membranes.cpp)
target_link_libraries(membranes ${LINK_LIBS})
install(TARGETS membranes RUNTIME DESTINATION bin)
