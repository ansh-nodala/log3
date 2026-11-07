cmake_minimum_required(VERSION 3.16)



project(log3

  VERSION 0.4.0

  DESCRIPTION "log3 C++17 Core, N-D Tensor, BitNet b1.58 SIMD Kernels, GQA Transformer, Zero-Copy .l3 I/O, Reverse-Mode Autograd & Optimizers"

  LANGUAGES CXX

)



# ------------------------------------------------------------------------------

# Build Options

# ------------------------------------------------------------------------------

option(L3_BUILD_SHARED    "Build log3 as a shared library (.so/.dll/.dylib)" OFF)

option(L3_BUILD_MAIN      "Build root main.cpp sandbox executable"           ON)

option(L3_BUILD_TESTS     "Build unit tests in tests/ directory"             ON)

option(L3_MEMORY_TRACKING "Enable default allocator memory tracking"         ON)

option(L3_ENABLE_NATIVE   "Enable -march=native / host SIMD tuning"          ON)

option(L3_ENABLE_PCH      "Precompile umbrella header include/log3/log3.h"   OFF)

option(L3_ENABLE_LTO      "Enable Link-Time Optimization (IPO/LTO)"          OFF)

option(L3_ENABLE_ASAN     "Enable AddressSanitizer + UBSan"                  OFF)

option(L3_ENABLE_TSAN     "Enable ThreadSanitizer"                           OFF)



set(L3_LOG_MIN_LEVEL "" CACHE STRING "Compile-time min log level (0=TRACE..5=FATAL)")



if(L3_ENABLE_ASAN AND L3_ENABLE_TSAN)

  message(FATAL_ERROR "log3: L3_ENABLE_ASAN and L3_ENABLE_TSAN cannot be enabled simultaneously.")

endif()



# Default to Release build if not specified

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

  message(STATUS "log3: Setting build type to 'Release' as none was specified.")

  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

endif()



# ------------------------------------------------------------------------------

# C++17 Standard Setup

# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# Always build position-independent code (.a can be linked into Python .so/.pyd bindings)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)



# Threads (required on Linux, Android, macOS, and Windows for ThreadPool)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)



# Optional Link-Time Optimization (LTO)

if(L3_ENABLE_LTO)

  include(CheckIPOSupported)

  check_ipo_supported(RESULT L3_IPO_SUPPORTED OUTPUT L3_IPO_ERROR)

  if(L3_IPO_SUPPORTED)

    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

  else()

    message(WARNING "log3: LTO/IPO requested but not supported: ${L3_IPO_ERROR}")

  endif()

endif()



# ------------------------------------------------------------------------------

# Common Compiler Flags Function (No Exceptions, No RTTI, Relative __FILE__)

# ------------------------------------------------------------------------------

function(l3_apply_target_flags target_name scope)

  if(MSVC)

    target_compile_options(${target_name} ${scope}

      /W4

      /MP

      /Zc:__cplusplus

      /EHs-c-    # Disable C++ exceptions

      /GR-       # Disable RTTI

      "/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=."

    )

    target_compile_definitions(${target_name} ${scope}

      _HAS_EXCEPTIONS=0

      _CRT_SECURE_NO_WARNINGS

      NOMINMAX

      WIN32_LEAN_AND_MEAN

    )

    if(L3_ENABLE_NATIVE AND NOT CMAKE_CROSSCOMPILING)

      if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x86")

        target_compile_options(${target_name} ${scope} /arch:AVX2)

      endif()

    endif()

  else()

    target_compile_options(${target_name} ${scope}

      -Wall

      -Wextra

      -Wpedantic

      -Wno-unused-parameter

      -fno-exceptions

      -fno-rtti

      "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/="

      "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."

    )

    # Apply -march=native only when not cross-compiling (e.g. skip on Android NDK)

    if(L3_ENABLE_NATIVE AND NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)

      target_compile_options(${target_name} ${scope} -march=native)

    endif()

  endif()



  # Sanitizers

  if(L3_ENABLE_ASAN AND NOT MSVC)

    target_compile_options(${target_name} ${scope} -fsanitize=address,undefined -fno-omit-frame-pointer)

    target_link_options(${target_name} ${scope} -fsanitize=address,undefined)

  endif()



  if(L3_ENABLE_TSAN AND NOT MSVC)

    target_compile_options(${target_name} ${scope} -fsanitize=thread -fno-omit-frame-pointer)

    target_link_options(${target_name} ${scope} -fsanitize=thread)

  endif()



  # Memory tracking macro

  if(L3_MEMORY_TRACKING)

    target_compile_definitions(${target_name} ${scope} L3_MEMORY_TRACKING=1)

  else()

    target_compile_definitions(${target_name} ${scope} L3_MEMORY_TRACKING=0)

  endif()



  # Compile-time minimum log level (if explicitly overridden)

  if(NOT "${L3_LOG_MIN_LEVEL}" STREQUAL "")

    target_compile_definitions(${target_name} ${scope} L3_LOG_MIN_LEVEL=${L3_LOG_MIN_LEVEL})

  endif()

endfunction()



# ------------------------------------------------------------------------------

# Library Sources & Headers (Stages 1–4: core, tensor, kernels, io, nn, autograd, optim)

# ------------------------------------------------------------------------------

file(GLOB_RECURSE L3_SOURCES CONFIGURE_DEPENDS

  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/optim/*.cpp"

  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"

)

list(REMOVE_DUPLICATES L3_SOURCES)



file(GLOB_RECURSE L3_HEADERS CONFIGURE_DEPENDS

  "${CMAKE_CURRENT_SOURCE_DIR}/include/log3/*.h"

)



if(NOT L3_SOURCES)

  message(FATAL_ERROR "log3: No .cpp files found under ${CMAKE_CURRENT_SOURCE_DIR}/src/")

endif()



if(L3_BUILD_SHARED)

  add_library(log3 SHARED ${L3_SOURCES} ${L3_HEADERS})

  target_compile_definitions(log3 PRIVATE L3_BUILDING=1)

  set_target_properties(log3 PROPERTIES

    VERSION   ${PROJECT_VERSION}

    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}

    CXX_VISIBILITY_PRESET hidden

    VISIBILITY_INLINES_HIDDEN ON

  )

else()

  add_library(log3 STATIC ${L3_SOURCES} ${L3_HEADERS})

  target_compile_definitions(log3 PUBLIC L3_STATIC=1)

endif()



target_include_directories(log3

  PUBLIC

    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

    $<INSTALL_INTERFACE:include>

)



l3_apply_target_flags(log3 PRIVATE)



# Optional precompiled umbrella header (include/log3/log3.h) for faster rebuilds

if(L3_ENABLE_PCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/log3/log3.h")

  target_precompile_headers(log3 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/log3/log3.h")

endif()



# Link threading across all platforms (Linux, Android, macOS, Windows)

target_link_libraries(log3 PUBLIC Threads::Threads)



# Link libm on POSIX platforms for fast math functions in kernels/RoPE/softmax

if(UNIX AND NOT APPLE)

  target_link_libraries(log3 PUBLIC m)

endif()



# Android logcat sink support

if(ANDROID)

  target_link_libraries(log3 PUBLIC log)

endif()



add_library(log3::log3 ALIAS log3)



# ------------------------------------------------------------------------------

# Root Executable: main.cpp

#   Automatically verifies that main.cpp contains an int main(...) entry point

#   so an empty or header-only scratch main.cpp never breaks the build.

# ------------------------------------------------------------------------------

if(L3_BUILD_MAIN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" L3_MAIN_CONTENT)

  if(L3_MAIN_CONTENT MATCHES "main[ \t\r\n]*\\(")

    add_executable(log3_main "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

    target_link_libraries(log3_main PRIVATE log3::log3)

    l3_apply_target_flags(log3_main PRIVATE)

  else()

    message(STATUS "log3: Skipping log3_main target because main.cpp has no main() entry point.")

  endif()

endif()



# ------------------------------------------------------------------------------

# Unit Tests (tests/*.cpp)

# ------------------------------------------------------------------------------

if(L3_BUILD_TESTS)

  enable_testing()

  file(GLOB L3_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

  foreach(test_src ${L3_TEST_SOURCES})

    get_filename_component(test_name "${test_src}" NAME_WE)

    add_executable(${test_name} "${test_src}")

    target_link_libraries(${test_name} PRIVATE log3::log3)

    l3_apply_target_flags(${test_name} PRIVATE)

    add_test(NAME ${test_name} COMMAND ${test_name})

  endforeach()

endif()



# ------------------------------------------------------------------------------

# Install Rules (Headers + Library Target)

# ------------------------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS log3

  EXPORT log3Targets

  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}

  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}

  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}

)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/log3"

  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

)

