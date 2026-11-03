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
  src/stages/line_prep.cu
  src/components/params.cpp
  src/components/preprocess.cpp
  src/components/hough_deskewer.cpp
  src/components/projection_detector.cpp
  src/components/builtin.cpp
  src/components/ctc_greedy_decoder.cpp
  src/components/charset.cpp
  src/components/txt_formatter.cpp
  src/components/json_formatter.cpp
  src/components/hocr_formatter.cpp
  src/components/words.cpp
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

# toml++ (header-only, MIT) for config files; only src/config.cpp includes it.
FetchContent_Declare(
  tomlplusplus
  URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
  URL_HASH SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR no-cmake  # header-only: populate only
)
FetchContent_MakeAvailable(tomlplusplus)
target_sources(cudaocr_lib PRIVATE src/config.cpp)
target_include_directories(cudaocr_lib SYSTEM PRIVATE ${tomlplusplus_SOURCE_DIR}/include)

# The ORT-backed recognizer is part of the library when ORT is enabled.
if(CUDAOCR_WITH_ORT)
  target_sources(cudaocr_lib PRIVATE src/components/onnx_line_recognizer.cpp)
  target_link_libraries(cudaocr_lib PUBLIC onnxruntime::onnxruntime)
  target_compile_definitions(cudaocr_lib PUBLIC CUDAOCR_WITH_ORT=1)
endif()
target_compile_definitions(cudaocr_lib PRIVATE
  CUDAOCR_DEFAULT_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

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
  tests/unit/ctc_decoder_test.cpp
  tests/unit/txt_formatter_test.cpp
  tests/unit/json_formatter_test.cpp
  tests/unit/hocr_formatter_test.cpp
  tests/unit/config_test.cpp
  tests/unit/words_from_ctc_test.cpp
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
  tests/unit/line_prep_test.cu
  tests/integration/pipeline_test.cu
  tests/integration/pipeline_golden_test.cu
  tests/integration/synth_fixtures_test.cu
  tests/integration/gray_deskew_test.cu
)
target_link_libraries(unit_tests PRIVATE cudaocr_lib gtest gtest_main)
if(CUDAOCR_WITH_ORT)
  target_sources(unit_tests PRIVATE tests/unit/ort_smoke_test.cpp tests/unit/rec_model_test.cpp
                 tests/unit/onnx_recognizer_test.cpp
                 tests/integration/ocr_end_to_end_test.cpp)
  target_link_libraries(unit_tests PRIVATE onnxruntime::onnxruntime)
endif()
target_include_directories(unit_tests PRIVATE tests)
target_compile_definitions(unit_tests PRIVATE
  CUDAOCR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
  CUDAOCR_SYNTH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth"
  CUDAOCR_ONNX_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/onnx"
  CUDAOCR_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
set_target_properties(unit_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60)

# CLI smoke tests: pin the end-to-end counts on the real sample scan through
# argument parsing and printing. Default counts changed in M3.11 (adaptive C
# 10 -> 40): 348 words / 903 glyph boxes -> 353 / 1233.
add_test(NAME Cli.SampleDefault COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png -v --no-recognize)
set_tests_properties(Cli.SampleDefault PROPERTIES
  PASS_REGULAR_EXPRESSION "lines: 35   words: 353   chars: 1233")
add_test(NAME Cli.SampleOtsuNoDeskew
         COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png --otsu --no-deskew -v
                 --no-recognize)
set_tests_properties(Cli.SampleOtsuNoDeskew PROPERTIES
  PASS_REGULAR_EXPRESSION "otsu thresh: +[0-9]+.*lines: 35   words: 353   chars: 1436")
add_test(NAME Cli.BadParamIsConfigError
         COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/resources/sample.png --C abc --no-recognize)
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

if(CUDAOCR_WITH_ORT)
  add_test(NAME Cli.TextOutput COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/cli_text_test.sh
                                       $<TARGET_FILE:cudaocr>)
  set_tests_properties(Cli.TextOutput PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME Cli.MultiPage
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/cli_multipage_test.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval>)
  set_tests_properties(Cli.MultiPage PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(CUDAOCR_WITH_ORT)
  # Regression: exit-time heap corruption when a static Ort::Env was destroyed
  # after the CUDA EP (~15% of runs). 25 fresh processes must all exit cleanly.
  add_test(NAME Ort.ExitIsClean
           COMMAND bash -c "for i in $(seq 25); do $<TARGET_FILE:unit_tests> --gtest_filter=OnnxRuntime.RunsAddOnCudaWithoutCpuFallback >/dev/null 2>&1 || { echo \"run $i crashed\"; exit 1; }; done")
  # Accuracy ratchets (benchmarks/ACCURACY.md): baseline + 1 CER point.
  # Clean synthetic pages: baseline 0.51% (M3.10) -> max 1.51%.
  add_test(NAME Accuracy.CleanSyntheticRatchet
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/accuracy_ratchet.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval> 0.0151
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth
                   page_sans_clean page_serif_clean page_mono_clean)
  set_tests_properties(Accuracy.CleanSyntheticRatchet PROPERTIES SKIP_RETURN_CODE 77)
  # Clean real page (resources/sherlock.png): baseline 0.89% (M3.10) -> max 1.89%.
  add_test(NAME Accuracy.SherlockRatchet
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/accuracy_ratchet.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval> 0.0189
                   ${CMAKE_CURRENT_SOURCE_DIR}/resources
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/real sherlock)
  set_tests_properties(Accuracy.SherlockRatchet PROPERTIES SKIP_RETURN_CODE 77)
  # M3.11 levels (adaptive C = 40): all committed synthetic fixtures 0.35% -> max
  # 1.35%; all real scans 5.01% -> max 6.01%.
  add_test(NAME Accuracy.SynthCiRatchet
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/accuracy_ratchet.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval> 0.0135
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/synth
                   page_sans_clean page_serif_clean page_mono_clean page_sans_skew_p3
                   page_serif_skew_m4 page_sans_noise page_serif_inverted line_pangram_sans
                   line_digits_serif line_punct_mono)
  set_tests_properties(Accuracy.SynthCiRatchet PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME Accuracy.RealScansRatchet
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/accuracy_ratchet.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval> 0.0601
                   ${CMAKE_CURRENT_SOURCE_DIR}/resources
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/real
                   sherlock noisy_text noisy_text2 vigorous-before)
  set_tests_properties(Accuracy.RealScansRatchet PROPERTIES SKIP_RETURN_CODE 77)
  # UNLV/ISRI sample (scripts/fetch_unlv.sh, default 40 pages): M4.6 baseline
  # zoned 6.84% -> max 7.84%, full page 19.64% -> max 20.64%.
  add_test(NAME Accuracy.UnlvRatchet
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/unlv_ratchet.sh
                   $<TARGET_FILE:cudaocr> $<TARGET_FILE:cudaocr-eval> 0.0784 0.2064)
  set_tests_properties(Accuracy.UnlvRatchet PROPERTIES SKIP_RETURN_CODE 77)
endif()
add_test(NAME Cli.Config COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/cli_config_test.sh
                                 $<TARGET_FILE:cudaocr>)
add_test(NAME Cli.BlankPageDetectOnlyExits3
         COMMAND cudaocr ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/blank.png --no-recognize)
set_tests_properties(Cli.BlankPageDetectOnlyExits3 PROPERTIES WILL_FAIL TRUE)

# Tooling tests (verify gate helpers).
add_test(NAME Scripts.GenFixtures
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/gen_fixtures_test.sh)
set_tests_properties(Scripts.GenFixtures PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME Scripts.ChangedFiles
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/changed_files_test.sh)

# ---- benchmarks (custom kernels vs NPP _Ctx) ----
add_executable(bench benchmarks/bench.cu)
target_link_libraries(bench PRIVATE cudaocr_lib CUDA::nppif CUDA::nppicc CUDA::nppig CUDA::nppc)
