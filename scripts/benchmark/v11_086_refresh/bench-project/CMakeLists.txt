cmake_minimum_required(VERSION 3.20)
project(v11_086_bench_only LANGUAGES CXX)
# Standalone project: no add_subdirectory of llama/ggml; dependencies are immutable
# COPIES in CANDIDATE/bin and CANDIDATE/common, checked before and after building.
foreach(required PINNED_SOURCE CANDIDATE LIBLLAMA_VERSION)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "missing ${required}")
  endif()
endforeach()
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CANDIDATE}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CANDIDATE}/bin")
set(deps
  "${CANDIDATE}/bin/libllama-common.so.0.0.${LIBLLAMA_VERSION}"
  "${CANDIDATE}/bin/libllama.so.0.0.${LIBLLAMA_VERSION}"
  "${CANDIDATE}/bin/libggml.so.0.16.0"
  "${CANDIDATE}/bin/libggml-cpu.so.0.16.0"
  "${CANDIDATE}/bin/libggml-base.so.0.16.0"
  "${CANDIDATE}/common/libllama-common-base.a")
foreach(dep IN LISTS deps)
  if(NOT EXISTS "${dep}")
    message(FATAL_ERROR "missing copied dependency ${dep}")
  endif()
endforeach()
add_library(llama-bench-impl SHARED "${PINNED_SOURCE}/tools/llama-bench/llama-bench.cpp")
add_executable(llama-bench "${PINNED_SOURCE}/tools/llama-bench/main.cpp")
foreach(target llama-bench-impl llama-bench)
  target_compile_definitions(${target} PRIVATE GGML_BACKEND_SHARED GGML_SHARED GGML_USE_CPU LLAMA_SHARED)
  target_include_directories(${target} PRIVATE
    "${PINNED_SOURCE}/tools/llama-bench" "${PINNED_SOURCE}/common"
    "${PINNED_SOURCE}/vendor" "${PINNED_SOURCE}/include" "${PINNED_SOURCE}/ggml/include")
  target_compile_options(${target} PRIVATE -Wmissing-declarations -Wmissing-noreturn
    -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi)
endforeach()
set_target_properties(llama-bench-impl PROPERTIES DEFINE_SYMBOL llama_bench_impl_EXPORTS POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llama-bench-impl PRIVATE ${deps})
target_link_libraries(llama-bench PRIVATE llama-bench-impl
  "${CANDIDATE}/bin/libllama-common.so.0.0.${LIBLLAMA_VERSION}"
  "${CANDIDATE}/common/libllama-common-base.a"
  "${CANDIDATE}/bin/libllama.so.0.0.${LIBLLAMA_VERSION}"
  "${CANDIDATE}/bin/libggml.so.0.16.0"
  "${CANDIDATE}/bin/libggml-cpu.so.0.16.0"
  "${CANDIDATE}/bin/libggml-base.so.0.16.0")
