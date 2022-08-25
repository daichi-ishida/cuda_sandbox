file(GLOB _CUDA_FILES *.cu)

add_library(${CUDART_OBJ} OBJECT ${_CUDA_FILES})
target_compile_features(${CUDART_OBJ} PUBLIC cuda_std_14)
target_compile_options(${CUDART_OBJ} PRIVATE 
--use_fast_math
"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_52,code=sm_52>"
)
target_link_libraries(${CUDART_OBJ} ${PROJECT_LIBRARY})
target_include_directories(${CUDART_OBJ}
    PRIVATE $<TARGET_PROPERTY:INTERFACE_INCLUDE_DIRECTORIES> ${COMMON_INCLUDE_DIRS})
