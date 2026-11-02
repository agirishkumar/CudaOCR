cmake_minimum_required(VERSION 3.24)
# Must precede project(): enabling CUDA defines CMAKE_CUDA_ARCHITECTURES from the
# compiler default (sm_75), after which a NOT DEFINED guard never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures (default: RTX 3060 Laptop)")
endif()
project(cudaocr LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")

find_package(CUDAToolkit REQUIRED)

add_library(cudaocr_lib STATIC
  src/core/logger.cpp
  src/core/host_image.cpp
  src/core/device_image.cu
  src/core/device_info.cu
  src/stages/grayscale.cu
  src/stages/filters.cu
  src/stages/binarize.cu
  src/stages/enhance.cu
  src/stages/deskew.cu
  src/stages/segment.cu
  src/stages/segment_types.cpp
  src/components/params.cpp
  src/components/preprocess.cpp
  src/components/hough_deskewer.cpp
  src/components/projection_detector.cpp
  src/components/builtin.cpp
  src/pipeline.cu
  src/legacy_pipeline.cu
)
target_include_directories(cudaocr_lib PUBLIC include)
target_include_directories(cudaocr_lib SYSTEM PUBLIC third_party)
target_link_libraries(cudaocr_lib PUBLIC CUDA::cudart)
set_target_properties(cudaocr_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cudaocr apps/cudaocr_cli/main.cpp)
target_link_libraries(cudaocr PRIVATE cudaocr_lib)

include(GNUInstallDirs)
install(TARGETS cudaocr RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---- tests ----
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(unit_tests
  tests/oracles/oracles.cpp
  tests/unit/status_test.cpp
  tests/unit/logger_test.cpp
  tests/unit/host_image_test.cpp
  tests/unit/types_test.cpp
  tests/unit/params_test.cpp
  tests/unit/registry_test.cpp
  tests/unit/components_test.cpp
  tests/unit/device_test.cu
  tests/unit/stream_ordering_test.cu
  tests/unit/grayscale_test.cu
  tests/unit/filters_test.cu
  tests/unit/binarize_test.cu
  tests/unit/enhance_test.cu
  tests/unit/deskew_test.cu
  tests/unit/segment_test.cu
  tests/unit/stage_adapters_test.cu
  tests/integration/pipeline_test.cu
  tests/integration/pipeline_equivalence_test.cu
  tests/integration/pipeline_v2_test.cu
)
target_link_libraries(unit_tests PRIVATE cudaocr_lib gtest gtest_main)
target_include_directories(unit_tests PRIVATE tests)
target_compile_definitions(unit_tests PRIVATE
  CUDAOCR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
set_target_properties(unit_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60)

# Tooling tests (verify gate helpers).
add_test(NAME Scripts.ChangedFiles
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/changed_files_test.sh)

# ---- benchmarks (custom kernels vs NPP _Ctx) ----
add_executable(bench benchmarks/bench.cu)
target_link_libraries(bench PRIVATE cudaocr_lib CUDA::nppif CUDA::nppicc CUDA::nppig CUDA::nppc)
