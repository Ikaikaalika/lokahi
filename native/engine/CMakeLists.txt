cmake_minimum_required(VERSION 3.20)
project(lokahi_engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
  set(LOKAHI_METAL_DEFAULT ON)
else()
  set(LOKAHI_METAL_DEFAULT OFF)
endif()
option(LOKAHI_METAL "Build the Metal GPU backend (macOS only)" ${LOKAHI_METAL_DEFAULT})
option(LOKAHI_CHECK_METAL_SHADERS "Compile shaders with xcrun metal at build time as a syntax check" OFF)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(LOKAHI_CORE_SOURCES
  src/c_api.cpp
  src/chat_template.cpp
  src/decoder.cpp
  src/json.cpp
  src/model.cpp
  src/quant.cpp
  src/regex.cpp
  src/safetensors.cpp
  src/sampling.cpp
  src/service.cpp
  src/text_stream.cpp
  src/thread_pool.cpp
  src/tokenizer.cpp
  src/unicode.cpp
  src/wire.cpp
)

if(LOKAHI_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "LOKAHI_METAL requires macOS")
  endif()
  enable_language(OBJCXX)
  # Embed the Metal source so the library has no build-time dependency on the
  # Metal toolchain; pipelines are compiled once at model load.
  set(LOKAHI_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/decoder_kernels.metal)
  set(LOKAHI_SHADER_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/decoder_kernels_source.h)
  add_custom_command(
    OUTPUT ${LOKAHI_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${LOKAHI_SHADER} -DOUTPUT=${LOKAHI_SHADER_HEADER}
            -DSYMBOL=kDecoderKernelSource -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
    DEPENDS ${LOKAHI_SHADER} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake
    COMMENT "Embedding Metal kernels")
  list(APPEND LOKAHI_CORE_SOURCES src/metal/metal_decoder.mm ${LOKAHI_SHADER_HEADER})
  if(LOKAHI_CHECK_METAL_SHADERS)
    add_custom_target(lokahi_shader_check ALL
      COMMAND xcrun -sdk macosx metal -std=metal3.0 -Werror -c ${LOKAHI_SHADER}
              -o ${CMAKE_CURRENT_BINARY_DIR}/decoder_kernels.air
      DEPENDS ${LOKAHI_SHADER}
      COMMENT "Checking Metal kernels with xcrun metal")
  endif()
endif()

add_library(lokahi_core STATIC ${LOKAHI_CORE_SOURCES})
target_include_directories(lokahi_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
find_package(Threads REQUIRED)
target_link_libraries(lokahi_core PUBLIC Threads::Threads)
if(LOKAHI_METAL)
  target_compile_definitions(lokahi_core PUBLIC LOKAHI_WITH_METAL=1)
  set_source_files_properties(src/metal/metal_decoder.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(lokahi_core PUBLIC "-framework Foundation" "-framework Metal")
endif()

# Shared library exposing the C ABI (include/lokahi/lokahi.h).
add_library(lokahi SHARED src/c_api_export.cpp)
target_link_libraries(lokahi PRIVATE lokahi_core)
if(APPLE)
  target_link_options(lokahi PRIVATE -Wl,-all_load)
else()
  target_link_options(lokahi PRIVATE -Wl,--whole-archive $<TARGET_FILE:lokahi_core> -Wl,--no-whole-archive)
endif()

add_executable(lokahi_cli tools/lokahi_cli.cpp)
set_target_properties(lokahi_cli PROPERTIES OUTPUT_NAME lokahi)
target_link_libraries(lokahi_cli PRIVATE lokahi_core)

enable_testing()
add_executable(lokahi_engine_tests tests/engine_tests.cpp)
target_link_libraries(lokahi_engine_tests PRIVATE lokahi_core)
add_test(NAME lokahi_engine_tests COMMAND lokahi_engine_tests)
