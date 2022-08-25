function(BUILD_TARGET SUBDIR)
    # ------------------------------------------------------------------------------
    # Configure Main Projects
    # ------------------------------------------------------------------------------
    set(TARGET "${SUBDIR}_APP")
    message(STATUS "Target: ${TARGET}")
    add_executable(${TARGET})
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${TARGET})

    add_subdirectory(${SUBDIR})

    target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})

    # ------------------------------------------------------------------------------
    # Set compiler options
    # ------------------------------------------------------------------------------
    target_compile_options(${TARGET} PUBLIC
        # warnings
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic>
        $<$<CXX_COMPILER_ID:GCC>:-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>

        # flags for each build type
        $<$<CXX_COMPILER_ID:Clang>:
            $<$<CONFIG:Release>:-O3>                  # Release
            $<$<CONFIG:Debug>:-O0 -g>                 # Debug
            $<$<CONFIG:RelWithDebgInfo>:-O3 -g>       # RelWithDebInfo
        >
        $<$<CXX_COMPILER_ID:GCC>:
            $<$<CONFIG:Release>:-O3>                  # Release
            $<$<CONFIG:Debug>:-O0 -g>                 # Debug
            $<$<CONFIG:RelWithDebgInfo>:-O3 -g>       # RelWithDebInfo
        >

        $<$<CXX_COMPILER_ID:MSVC>: 
            $<$<CONFIG:Release>:/O2 /Zi /MP>          # Release
            $<$<CONFIG:Debug>:/Od /ZI /MP>            # Debug
            $<$<CONFIG:RelWithDebgInfo>:/O2 /ZI /MP>  # RelWithDebInfo
        >
    )
endfunction(BUILD_TARGET)

file(GLOB SUBDIR_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR} "*")
foreach(SUBDIR ${SUBDIR_LIST})
  if (IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${SUBDIR}")
    BUILD_TARGET(${SUBDIR})
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Post Process (copy shared files)
# ------------------------------------------------------------------------------

# These are run every time the main project is built:
add_custom_target(${PROJECT_NAME} ALL)
