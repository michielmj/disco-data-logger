cmake_minimum_required(VERSION 3.18)

project(data_logger LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-fvisibility=hidden)
endif()

if (MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# -------------------- ZSTD --------------------
option(USE_SYSTEM_ZSTD "Use system-provided zstd instead of vendoring" OFF)
set(ZSTD_TAG v1.5.6 CACHE STRING "zstd version/tag when vendoring")

if (USE_SYSTEM_ZSTD)
  find_package(ZSTD QUIET)
  if (TARGET ZSTD::ZSTD)
    set(ZSTD_TARGET ZSTD::ZSTD)
  else()
    find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
    find_library(ZSTD_LIBRARY NAMES zstd)
    if (NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
      message(FATAL_ERROR "System zstd not found. Provide ZSTD_INCLUDE_DIR/ZSTD_LIBRARY or set USE_SYSTEM_ZSTD=OFF.")
    endif()
    add_library(zstd_system UNKNOWN IMPORTED)
    set_target_properties(zstd_system PROPERTIES
      IMPORTED_LOCATION "${ZSTD_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
    set(ZSTD_TARGET zstd_system)
  endif()
else()
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  # IMPORTANT: use zstd's CMake project under build/cmake
  FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/archive/refs/tags/${ZSTD_TAG}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR build/cmake
    # URL_HASH SHA256=<add checksum for reproducible builds>
  )
  # Build static lib only; no CLI/tests
  set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(zstd)

  # Targets exported by zstd's CMake (varies by version/toolchain)
  if (TARGET zstd::libzstd_static)
    set(ZSTD_TARGET zstd::libzstd_static)
  elseif (TARGET libzstd_static)
    set(ZSTD_TARGET libzstd_static)
  elseif (TARGET zstd::libzstd_shared)
    set(ZSTD_TARGET zstd::libzstd_shared)
  elseif (TARGET libzstd_shared)
    set(ZSTD_TARGET libzstd_shared)
  elseif (TARGET ZSTD::ZSTD) # rare, but handle it
    set(ZSTD_TARGET ZSTD::ZSTD)
  else()
    message(FATAL_ERROR "Vendored zstd did not export a known target (looked for zstd::libzstd_static/libzstd_static/…)")
  endif()
endif()

# -------------------- Sources --------------------
set(DATA_LOGGER_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/data_logger/cpp")
set(DATA_LOGGER_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/src/data_logger/_core.cpp")

pybind11_add_module(_core ${DATA_LOGGER_SRC})
target_include_directories(_core PRIVATE "${DATA_LOGGER_CPP_DIR}")
target_link_libraries(_core PRIVATE ${ZSTD_TARGET})

include(CheckSymbolExists)
check_symbol_exists(fdatasync "unistd.h" HAVE_FDATASYNC)
if (HAVE_FDATASYNC)
  target_compile_definitions(_core PRIVATE HAVE_FDATASYNC=1)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if (ipo_ok)
  set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS _core DESTINATION data_logger)

message(STATUS "== data_logger build configuration ==")
message(STATUS "Use system zstd       : ${USE_SYSTEM_ZSTD}")
message(STATUS "Vendored zstd tag     : ${ZSTD_TAG}")
message(STATUS "ZSTD target           : ${ZSTD_TARGET}")
