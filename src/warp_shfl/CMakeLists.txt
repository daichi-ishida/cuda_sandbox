find_package(CUDAToolkit 11 REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE CXX_HEADERS *.h *.hpp)
file(GLOB_RECURSE CXX_SOURCES *.cpp)
set(CXX_FILES ${CXX_HEADERS} ${CXX_SOURCES})
target_sources(${TARGET} PRIVATE ${CXX_FILES})

set(PROJECT_LIBRARY CUDA::cudart CUDA::cublas)
set(CUDART_OBJ "${TARGET}_CUDART")

add_subdirectory("kernel")

target_link_libraries(${TARGET} ${PROJECT_LIBRARY} ${CUDART_OBJ})
# target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})

target_compile_features(${TARGET} PUBLIC cuda_std_14)
target_compile_definitions(${TARGET} PUBLIC
    $<$<CONFIG:Debug>:DEBUG>          # Debug
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>  # else
    _OPENMP 
    _USE_MATH_DEFINES
)
