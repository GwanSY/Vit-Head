find_package(OpenMP REQUIRED)
# find_package(OpenFHE)
# set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Wno-error")
set(SCI_EXTERN_INSTALL_DIR "${PROJECT_BINARY_DIR}/extern")
set(SCI_SEAL_INSTALL_DIR "${SCI_EXTERN_INSTALL_DIR}/seal")
set(SCI_PHANTOM_INSTALL_DIR "${SCI_EXTERN_INSTALL_DIR}/phantom")
set(SCI_SEAL_BUILD_DIR "${PROJECT_BINARY_DIR}/deps-build/seal")
set(SCI_PHANTOM_BUILD_DIR "${PROJECT_BINARY_DIR}/deps-build/phantom")
set(SCI_SEAL_CONFIG_DIR "${SCI_SEAL_INSTALL_DIR}/lib/cmake/SEAL-4.1")
set(SCI_PHANTOM_CONFIG_DIR "${SCI_PHANTOM_INSTALL_DIR}/lib/cmake/phantom")

# Do not let a SEAL_DIR cached by an earlier build tree override this build's
# private dependency installation.
unset(SEAL_DIR CACHE)
find_package(SEAL 4.1.1 EXACT CONFIG QUIET
    PATHS "${SCI_SEAL_CONFIG_DIR}" NO_DEFAULT_PATH)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 4.1.1 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND}
        -S "${PROJECT_SOURCE_DIR}/extern/SEAL"
        -B "${SCI_SEAL_BUILD_DIR}"
        -DCMAKE_INSTALL_PREFIX=${SCI_SEAL_INSTALL_DIR}
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DSEAL_BUILD_DEPS=OFF
        -DSEAL_USE_MSGSL=OFF
        -DSEAL_USE_ZLIB=OFF
        -DSEAL_USE_ZSTD=OFF
        RESULT_VARIABLE SCI_SEAL_CONFIGURE_RESULT)
    if (NOT SCI_SEAL_CONFIGURE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to configure SEAL (exit ${SCI_SEAL_CONFIGURE_RESULT})")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND}
        --build "${SCI_SEAL_BUILD_DIR}" --target install --parallel
        RESULT_VARIABLE SCI_SEAL_BUILD_RESULT)
    if (NOT SCI_SEAL_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to build/install SEAL (exit ${SCI_SEAL_BUILD_RESULT})")
    endif()
    find_package(SEAL 4.1.1 EXACT CONFIG REQUIRED
        PATHS "${SCI_SEAL_CONFIG_DIR}" NO_DEFAULT_PATH)
endif()

if(USE_HE_GPU)
    # Phantom_DIR is also cached globally by find_package. Keep it tied to the
    # selected SCI binary directory for relocatable/reproducible builds.
    unset(Phantom_DIR CACHE)
    find_package(Phantom CONFIG QUIET
        PATHS "${SCI_PHANTOM_CONFIG_DIR}" NO_DEFAULT_PATH)
    if(NOT Phantom_FOUND)
        message(STATUS "Phantom was not found: faninstall locally")
        execute_process(COMMAND ${CMAKE_COMMAND}
            -S "${PROJECT_SOURCE_DIR}/extern/phantom"
            -B "${SCI_PHANTOM_BUILD_DIR}"
            -DCMAKE_INSTALL_PREFIX=${SCI_PHANTOM_INSTALL_DIR}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            RESULT_VARIABLE SCI_PHANTOM_CONFIGURE_RESULT)
        if (NOT SCI_PHANTOM_CONFIGURE_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to configure Phantom (exit ${SCI_PHANTOM_CONFIGURE_RESULT})")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND}
            --build "${SCI_PHANTOM_BUILD_DIR}" --target install --parallel
            RESULT_VARIABLE SCI_PHANTOM_BUILD_RESULT)
        if (NOT SCI_PHANTOM_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to build/install Phantom (exit ${SCI_PHANTOM_BUILD_RESULT})")
        endif()
        find_package(Phantom CONFIG REQUIRED
            PATHS "${SCI_PHANTOM_CONFIG_DIR}" NO_DEFAULT_PATH)
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        find_package(CUDAToolkit REQUIRED)
        set(SCI_CUDART_TARGET CUDA::cudart)
    else()
        find_package(CUDA REQUIRED)
        set(SCI_CUDART_TARGET ${CUDA_CUDART_LIBRARY})
    endif()
endif()
            
# link_directories( ${OpenFHE_LIBDIR} )
# link_directories( ${OPENMP_LIBRARIES} )
# set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
# link_libraries( ${OpenFHE_SHARED_LIBRARIES} )

if(NOT USE_HE_GPU)
    find_path(SCI_CUDA_HEADER_ROOT cuda/std/type_traits
        HINTS ENV CUDA_HOME /usr/local/cuda
        PATH_SUFFIXES include)
    if(NOT SCI_CUDA_HEADER_ROOT)
        message(FATAL_ERROR
            "Bundled Phantom declarations require CUDA C++ headers even for the SEAL host build")
    endif()
    add_library(SCI-LinearHE
        conv-field.cpp
        fc-field.cpp
        elemwise-prod-field.cpp
        utils-HE.cpp
        # bert-matmul.cpp
        # bert-matmul-cipher-efficient.cpp
        # bert-matmul-cipher-efficient-seal.cpp
        # bert-matmul-cipher-efficient-seal-ckks.cpp
        # iron-seal.cpp
        # iron-seal-inter1.cpp
        # bert-ct-pt.cpp
        # bert-ct-pt-inter1.cpp
        # utils-HE-openfhe.cpp
    )

    # include_directories( ${OPENMP_INCLUDES} )
    # include_directories( ${OpenFHE_INCLUDE} )
    # include_directories( ${OpenFHE_INCLUDE}/third-party/include )
    # include_directories( ${OpenFHE_INCLUDE}/core )
    # include_directories( ${OpenFHE_INCLUDE}/pke )

    target_link_libraries(SCI-LinearHE
        PUBLIC
        SCI-utils
        SEAL::seal
        ${CMAKE_DL_LIBS}
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
    # Unified CKKS value types are declared in the bundled Phantom headers even
    # when execution is restricted to the SEAL/host branch.  Expose only this
    # repository-local header root; the CPU target does not link Phantom/CUDA.
    target_include_directories(SCI-LinearHE PUBLIC
        "${PROJECT_SOURCE_DIR}/extern/phantom/include"
        "${SCI_CUDA_HEADER_ROOT}"
    )
else()
    add_library(SCI-LinearHE
        conv-field.cpp
        fc-field.cpp
        elemwise-prod-field.cpp
        utils-HE.cpp
        unified-he.cpp
        evaluator-wrapper.cpp
    )
    target_link_libraries(SCI-LinearHE
        PUBLIC
        SCI-utils
        SEAL::seal
        phantom::Phantom
        ${SCI_CUDART_TARGET} # <cuda_runtime_api.h>
        ${CMAKE_DL_LIBS} # fix: undefined reference to symbol 'dlclose@@GLIBC_2.2.5'
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
    target_include_directories(SCI-LinearHE PUBLIC
        "${SCI_PHANTOM_INSTALL_DIR}/include"
        "${SCI_PHANTOM_INSTALL_DIR}/include/phantom"
        ${CUDA_INCLUDE_DIRS}
    )
    set_target_properties(SCI-LinearHE PROPERTIES CUDA_ARCHITECTURES "70;80;90")
endif()
