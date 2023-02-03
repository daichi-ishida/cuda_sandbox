project(example_find_max LANGUAGES CUDA CXX)

add_executable(${PROJECT_NAME})

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SOURCE FILES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
file(GLOB_RECURSE CXX_SOURCES *.cpp)
set(CXX_FILES ${CXX_SOURCES})
target_sources(${PROJECT_NAME} PRIVATE ${CXX_FILES})

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# LIBRARIES :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
find_package(CUDAToolkit 11 REQUIRED)
find_package(OpenMP REQUIRED)

set(CUDART_OBJ "${PROJECT_NAME}_CUDART")
add_subdirectory("kernel")

target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cublas ${CUDART_OBJ})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# COMPILER OPTIONS ::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

target_compile_definitions(${PROJECT_NAME} PUBLIC
$<$<CONFIG:Debug>:DEBUG>
$<$<NOT:$<CONFIG:Debug>>:NDEBUG>
_OPENMP 
_USE_MATH_DEFINES
)

target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_14)

target_compile_options(${PROJECT_NAME} PUBLIC
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