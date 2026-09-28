add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine ninfer_product_logging)

# Only meaningful on Turing, where mma_tf32_bits() and mma_bf16() are bridged onto the fp16
# tensor cores instead of issuing the real instruction. The bridge packs the fragment registers
# by hand, which no amount of SASS inspection can verify -- so it gets a standalone probe that
# re-derives the products on the CUDA cores and demands bit-exact equality. Run it on the
# target GPU before trusting prefill PPL.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "75")
  add_executable(ninfer-sm75-mma-selftest sm75_mma_selftest.cu)
  target_include_directories(ninfer-sm75-mma-selftest PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
endif()
