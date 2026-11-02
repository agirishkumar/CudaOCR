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
  src/eval/metrics.cpp
  src/pipeline.cu
)
target_include_directories(cudaocr_lib PUBLIC include)
target_include_directories(cudaocr_lib SYSTEM PUBLIC third_party)
target_link_libraries(cudaocr_lib PUBLIC CUDA::cudart)
set_target_properties(cudaocr_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cudaocr apps/cudaocr_cli/main.cpp)
target_link_libraries(cudaocr PRIVATE cudaocr_lib)
add_executable(cudaocr-eval apps/cudaocr_eval/main.cpp)
target_link_libraries(cudaocr-eval PRIVATE cudaocr_lib)

include(GNUInstallDirs)
install(TARGETS cudaocr cudaocr-eval RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---- ONNX Runtime (recognizer backend, M3; optional) ----
# Prebuilt CUDA 13 release, SHA-256 pinned. Set CUDAOCR_ORT_ROOT to an already
# extracted tree (include/, lib/) to skip the 233 MB download, e.g. for extra
# build directories. Only src/components/ and tests may include ORT headers.
option(CUDAOCR_WITH_ORT "Build ONNX Runtime-backed components" ON)
set(CUDAOCR_ORT_VERSION 1.31.1)
set(CUDAOCR_ORT_ROOT "" CACHE PATH "Existing ONNX Runtime tree (skips download)")
include(FetchContent)
if(CUDAOCR_WITH_ORT)
  if(NOT CUDAOCR_ORT_ROOT)
    FetchContent_Declare(
      onnxruntime
      URL https://github.com/microsoft/onnxruntime/releases/download/v${CUDAOCR_ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${CUDAOCR_ORT_VERSION}.tgz
      URL_HASH SHA256=b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR no-cmake  # prebuilt binaries: populate only, no add_subdirectory
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(CUDAOCR_ORT_ROOT ${onnxruntime_SOURCE_DIR} CACHE PATH "" FORCE)
  endif()
  if(NOT EXISTS ${CUDAOCR_ORT_ROOT}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "CUDAOCR_ORT_ROOT=${CUDAOCR_ORT_ROOT} has no include/onnxruntime_cxx_api.h")
  endif()
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION ${CUDAOCR_ORT_ROOT}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${CUDAOCR_ORT_ROOT}/include)
  message(STATUS "ONNX Runtime: ${CUDAOCR_ORT_ROOT}")
endif()

# ---- tests ----
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
  tests/unit/eval_test.cpp
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
  tests/integration/pipeline_golden_test.cu
  tests/integration/synth_fixtures_test.cu
  tests/integration/gray_deskew_test.cu
)
target_link_libraries(unit_tests PRIVATE cudaocr_lib gtest gtest_main)
if(CUDAOCR_WITH_ORT)
  target_sources(unit_tests PRIVATE tests/unit/ort_smoke_test.cpp)
  target_link_libraries(unit_tests PRIVATE onnxruntime::onnxruntime)
endif()
target_include_directories(unit_tests PRIVATE tests)
target_compile_definitions(unit_tests PRIVATE
  CUDAOCR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
  CUDAOCR_SYNTH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth"
  CUDAOCR_ONNX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/onnx")
set_target_properties(unit_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60)

# CLI smoke tests: pin the end-to-end counts on the real sample scan
# (same numbers as PipelineGolden, through argument parsing and printing).
add_test(NAME Cli.SampleDefault COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png)
set_tests_properties(Cli.SampleDefault PROPERTIES
  PASS_REGULAR_EXPRESSION "lines: 35   words: 348   chars: 903")
add_test(NAME Cli.SampleOtsuNoDeskew
         COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png --otsu --no-deskew)
set_tests_properties(Cli.SampleOtsuNoDeskew PROPERTIES
  PASS_REGULAR_EXPRESSION "otsu thresh: +[0-9]+.*lines: 35   words: 353   chars: 1436")
add_test(NAME Cli.BadParamIsConfigError
         COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png --C abc)
set_tests_properties(Cli.BadParamIsConfigError PROPERTIES WILL_FAIL TRUE)

# cudaocr-eval on a tiny fixture: exact (0/11 chars, 0/2 words), one-char typo
# (1/19, 1/4), missing prediction (5/5, 1/1) -> aggregate CER 6/35, WER 2/7.
add_test(NAME Eval.ToolAggregate
         COMMAND cudaocr-eval ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/pred
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/gt)
set_tests_properties(Eval.ToolAggregate PROPERTIES
  PASS_REGULAR_EXPRESSION "total.*17[.]14%.*6/35.*28[.]57%.*2/7.*missing: 1")
add_test(NAME Eval.ToolMaxCerFails
         COMMAND cudaocr-eval ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/pred
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/gt --max-cer 0.10)
set_tests_properties(Eval.ToolMaxCerFails PROPERTIES WILL_FAIL TRUE)
add_test(NAME Eval.ToolMaxCerPasses
         COMMAND cudaocr-eval ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/pred
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/eval/gt --max-cer 0.20)

# Tooling tests (verify gate helpers).
add_test(NAME Scripts.GenFixtures
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/gen_fixtures_test.sh)
set_tests_properties(Scripts.GenFixtures PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME Scripts.ChangedFiles
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/changed_files_test.sh)

# ---- benchmarks (custom kernels vs NPP _Ctx) ----
add_executable(bench benchmarks/bench.cu)
target_link_libraries(bench PRIVATE cudaocr_lib CUDA::nppif CUDA::nppicc CUDA::nppig CUDA::nppc)
