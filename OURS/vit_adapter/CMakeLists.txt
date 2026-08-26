cmake_minimum_required(VERSION 3.16)
project(vit_smoke_e2e LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BLB_SCI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../SCI" CACHE PATH
    "SCI source tree inside BLB")
set(BLB_SCI_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../build_vit_smoke_e2e/sci"
    CACHE PATH "SCI build tree inside BLB")
set(BLB_SEAL_DIR "${BLB_SCI_BUILD}/extern/seal/lib/cmake/SEAL-4.1"
    CACHE PATH "SEAL package built inside BLB")
set(BLB_GMP_ROOT "" CACHE PATH "Optional GMP installation prefix")

if(NOT EXISTS "${BLB_SCI_BUILD}/lib/libSCI-FloatingPoint.a")
  message(FATAL_ERROR "Build the bundled SCI targets first: ${BLB_SCI_BUILD}")
endif()

find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(SEAL 4.1.1 EXACT REQUIRED CONFIG PATHS "${BLB_SEAL_DIR}"
             NO_DEFAULT_PATH)
if(BLB_GMP_ROOT AND
   EXISTS "${BLB_GMP_ROOT}/include/x86_64-linux-gnu/gmp.h" AND
   EXISTS "${BLB_GMP_ROOT}/lib/x86_64-linux-gnu/libgmp.a" AND
   EXISTS "${BLB_GMP_ROOT}/lib/x86_64-linux-gnu/libgmpxx.a")
  # Ubuntu's extracted -dev package contains dangling shared-library symlinks.
  # Pin the self-contained archives so no host/project library is selected.
  set(BLB_GMP_INCLUDE_DIR "${BLB_GMP_ROOT}/include/x86_64-linux-gnu")
  set(BLB_GMP_LIBRARY "${BLB_GMP_ROOT}/lib/x86_64-linux-gnu/libgmp.a")
  set(BLB_GMPXX_LIBRARY "${BLB_GMP_ROOT}/lib/x86_64-linux-gnu/libgmpxx.a")
else()
  find_path(BLB_GMP_INCLUDE_DIR NAMES gmp.h HINTS "${BLB_GMP_ROOT}"
            PATH_SUFFIXES include include/x86_64-linux-gnu)
  find_library(BLB_GMP_LIBRARY NAMES gmp HINTS "${BLB_GMP_ROOT}"
               PATH_SUFFIXES lib lib/x86_64-linux-gnu)
  find_library(BLB_GMPXX_LIBRARY NAMES gmpxx HINTS "${BLB_GMP_ROOT}"
               PATH_SUFFIXES lib lib/x86_64-linux-gnu)
endif()
if(NOT BLB_GMP_INCLUDE_DIR OR NOT BLB_GMP_LIBRARY OR NOT BLB_GMPXX_LIBRARY)
  message(FATAL_ERROR "GMP not found; install libgmp-dev or set BLB_GMP_ROOT")
endif()

add_executable(vit_smoke_e2e
  blb_vit_e2e.cpp
  blb_ckks_linear.cpp
  "${BLB_SCI_ROOT}/src/GC/emp-tool.cpp"
  "${BLB_SCI_ROOT}/src/GC/halfgate_gen.cpp"
  "${BLB_SCI_ROOT}/src/GC/halfgate_eva.cpp")
target_include_directories(vit_smoke_e2e PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${BLB_SCI_ROOT}/src"
  "${BLB_SCI_ROOT}/extern/eigen"
  "${BLB_GMP_INCLUDE_DIR}")
target_compile_definitions(vit_smoke_e2e PRIVATE SCI_HE=1)
target_compile_options(vit_smoke_e2e PRIVATE
  -O2 -maes -msse4.1 -mavx -mavx2 -mrdseed
  -Wno-deprecated-declarations)
set(SCI_LIB "${BLB_SCI_BUILD}/lib")
target_link_libraries(vit_smoke_e2e PRIVATE
  "-Wl,--start-group"
  "${SCI_LIB}/libSCI-FloatingPoint.a"
  "${SCI_LIB}/libSCI-Math.a"
  "${SCI_LIB}/libSCI-BuildingBlocks.a"
  "${SCI_LIB}/libSCI-LinearOT.a"
  "-Wl,--end-group"
  SEAL::seal OpenSSL::SSL OpenSSL::Crypto
  OpenMP::OpenMP_CXX Threads::Threads
  "${BLB_GMP_LIBRARY}" "${BLB_GMPXX_LIBRARY}" ${CMAKE_DL_LIBS})
set_target_properties(vit_smoke_e2e PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
