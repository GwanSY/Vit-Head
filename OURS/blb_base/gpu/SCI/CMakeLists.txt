# cmake_minimum_required (VERSION 3.3.2)
cmake_minimum_required (VERSION 3.11)
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)            # 使用C++17
set(CMAKE_CXX_STANDARD_REQUIRED True) # 强制使用指定的C++标准
set(CMAKE_CXX_EXTENSIONS OFF)        # 禁用编译器特定的扩展（如GNU扩展）
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_PREFIX_PATH})
include(CMakeFindDependencyMacro)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_HE_GPU "Use HE-GPU backend" ON)
message(STATUS "Option: USE_HE_GPU = ${USE_HE_GPU}")
if(NOT USE_HE_GPU)
    project (SCI LANGUAGES C CXX)
else()
    project (SCI LANGUAGES C CXX CUDA)
endif()
set(NAME "SCI")

add_definitions(-DUSE_VOLE=1)

option(NO_REVEAL_OUTPUT "Only output secret shares after 2PC" OFF)
message(STATUS "Option: NO_REVEAL_OUTPUT = ${NO_REVEAL_OUTPUT}")

option(BUILD_TESTS "Build tests" ON)
message(STATUS "Option: BUILD_TESTS = ${BUILD_TESTS}")

option(BUILD_NETWORKS "Build networks" OFF)
message(STATUS "Option: BUILD_NETWORKS = ${BUILD_NETWORKS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
## Dependencies
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
# find_package(Eigen3 3.3.3 REQUIRED)
set(EMP-TOOL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/emp-tool")
set(EMP-TOOL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/deps/emp-tool")
set(EMP-OT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/emp-ot")
message(STATUS "EMP-TOOL Directory: ${EMP-TOOL_INCLUDE_DIR}")
find_package(emp-tool REQUIRED PATHS ${EMP-TOOL_INCLUDE_DIR})
find_package(emp-ot REQUIRED PATHS ${EMP-OT_INCLUDE_DIR})
include_directories(${EMP-OT_INCLUDE_DIRS} 
                    ${EMP-TOOL_INCLUDE_DIRS})
message(STATUS "EMP-OT Directory: ${EMP-OT_INCLUDE_DIRS}")
message(STATUS "EMP-TOOL Directory: ${EMP-OT_INCLUDE_DIRS}")

if (NO_REVEAL_OUTPUT)
    add_compile_definitions(NO_REVEAL_OUTPUT=1)
endif()

add_subdirectory(src)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

# if (BUILD_NETWORKS)
#     add_subdirectory(networks)
# endif()

set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(BUILD_DIR "${PROJECT_SOURCE_DIR}/build")
include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake.in"
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake"
	INSTALL_DESTINATION lib/cmake/SCI
    PATH_VARS INSTALL_DIR BUILD_DIR)
install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake"
        DESTINATION lib/cmake/SCI)
