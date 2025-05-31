set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_tool(mlir-scan
    mlir-scan.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-scan
  PRIVATE
    ${dialect_libs}
    )