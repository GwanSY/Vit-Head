add_library(SCI-BOLTTransformer
    bolt_he.cpp
    bolt_evaluator.cpp
    bolt_conversion.cpp
    bolt_linear.cpp
    bolt_attention.cpp)

target_link_libraries(SCI-BOLTTransformer
    PUBLIC
    SCI-LinearHE
    SCI-utils)

target_include_directories(SCI-BOLTTransformer
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:./>)

if(BOLT_USE_HE_GPU)
    if(NOT BOLT_PHANTOM_DIR)
        message(FATAL_ERROR
            "BOLT_USE_HE_GPU=ON requires -DBOLT_PHANTOM_DIR=<Phantom cmake config directory> "
            "from the BLB GPU build")
    endif()
    unset(Phantom_DIR CACHE)
    find_package(Phantom CONFIG REQUIRED
        PATHS "${BOLT_PHANTOM_DIR}" NO_DEFAULT_PATH)
    # Phantom's generated export from newer CMake versions advertises
    # `cuda_std_17`; CMake 3.16 (used by the original BOLT build) does not
    # know that feature even though nvcc is available.  The target itself is
    # still compiled with CUDA enabled below; remove only the unsupported
    # imported feature marker.
    set_property(TARGET phantom::Phantom PROPERTY
        INTERFACE_COMPILE_FEATURES cxx_std_17)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        find_package(CUDAToolkit REQUIRED)
        set(BOLT_CUDART_TARGET CUDA::cudart)
    else()
        find_package(CUDA REQUIRED)
        set(BOLT_CUDART_TARGET ${CUDA_CUDART_LIBRARY})
    endif()
    target_compile_definitions(SCI-BOLTTransformer PUBLIC BOLT_USE_HE_GPU=1)
    target_link_libraries(SCI-BOLTTransformer PUBLIC phantom::Phantom
        ${BOLT_CUDART_TARGET})
    find_path(BOLT_CUDA_HEADER_ROOT cuda/std/type_traits
        HINTS ENV CUDA_HOME /usr/local/cuda
        PATH_SUFFIXES include)
    if(NOT BOLT_CUDA_HEADER_ROOT)
        message(FATAL_ERROR
            "BOLT_USE_HE_GPU=ON requires CUDA headers (cuda/std/type_traits)")
    endif()
    target_include_directories(SCI-BOLTTransformer PRIVATE
        "${BOLT_PHANTOM_DIR}/../../../include"
        "${BOLT_PHANTOM_DIR}/../../../include/phantom"
        "${BOLT_CUDA_HEADER_ROOT}")
endif()
