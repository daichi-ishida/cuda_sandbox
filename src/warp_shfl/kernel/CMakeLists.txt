file(GLOB _CUDA_FILES *.cu)

add_library(${CUDART_OBJ} OBJECT ${_CUDA_FILES})
target_compile_features(${CUDART_OBJ} PUBLIC cuda_std_14)
set_target_properties(${CUDART_OBJ} PROPERTIES CUDA_ARCHITECTURES "61")
target_compile_options(${CUDART_OBJ} PRIVATE 
--use_fast_math
)
target_link_libraries(${CUDART_OBJ} ${PROJECT_LIBRARY})
target_include_directories(${CUDART_OBJ}
    PRIVATE $<TARGET_PROPERTY:INTERFACE_INCLUDE_DIRECTORIES> ${COMMON_INCLUDE_DIRS})
