# Out-of-tree build of df2_feature_dump against a champion kernel store. Nothing in a llama.cpp tree is modified.
#
#   LLAMA_SRC   a checkout or `git archive` of the champion source commit (headers only: include/, src/llama-ext.h,
#               ggml/include/). The headers MUST come from the same commit as the store (C14 = eee3da838).
#   LLAMA_STORE the kernel store whose bin/ holds libllama.so (e.g. /mnt/raid0/llm/kernels/builds/gpu-20261118-eee3da838-c14)
#
#   cmake -S . -B build -DLLAMA_SRC=<src> -DLLAMA_STORE=<store> && cmake --build build
#
# Run with LD_LIBRARY_PATH=<store>/bin:/opt/rocm/lib and prove ggml linkage with
# epyc-inference-research/scripts/utils/verify_ggml_linkage.sh before trusting a dump.
cmake_minimum_required(VERSION 3.16)
project(df2_feature_dump CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT LLAMA_SRC OR NOT LLAMA_STORE)
  message(FATAL_ERROR "set -DLLAMA_SRC and -DLLAMA_STORE")
endif()
add_executable(df2_feature_dump df2_feature_dump.cpp)
target_include_directories(df2_feature_dump PRIVATE ${LLAMA_SRC}/include ${LLAMA_SRC}/src ${LLAMA_SRC}/ggml/include)
target_link_libraries(df2_feature_dump PRIVATE ${LLAMA_STORE}/bin/libllama.so ${LLAMA_STORE}/bin/libggml.so ${LLAMA_STORE}/bin/libggml-base.so)
set_target_properties(df2_feature_dump PROPERTIES BUILD_RPATH "${LLAMA_STORE}/bin")
