find_package(OpenMP REQUIRED)
# find_package(OpenFHE)
# set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Wno-error")
find_package(SEAL 4.1.1 EXACT QUIET PATHS "${PROJECT_BINARY_DIR}/extern/seal" NO_DEFAULT_PATH)
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
    execute_process(COMMAND ${CMAKE_COMMAND} -S . -B ${PROJECT_BINARY_DIR}/thirdparty/seal-build -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/extern/seal
        -DFETCHCONTENT_SOURCE_DIR_MSGSL=${PROJECT_SOURCE_DIR}/extern/SEAL/thirdparty/msgsl-src
        -DFETCHCONTENT_SOURCE_DIR_ZLIB=${PROJECT_SOURCE_DIR}/extern/SEAL/thirdparty/zlib-src
        -DFETCHCONTENT_SOURCE_DIR_ZSTD=${PROJECT_SOURCE_DIR}/extern/SEAL/thirdparty/zstd-src
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}/thirdparty/seal-build --target install --parallel
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    find_package(SEAL 4.1.1 EXACT REQUIRED PATHS "${PROJECT_BINARY_DIR}/extern/seal" NO_DEFAULT_PATH)
endif()

if(USE_HE_GPU)
    find_package(Phantom CONFIG PATHS "${PROJECT_BINARY_DIR}/extern/phantom")
    if(NOT Phantom_FOUND)
        message(STATUS "Phantom was not found: faninstall locally")
        execute_process(COMMAND ${CMAKE_COMMAND} -S . -B ${PROJECT_BINARY_DIR}/thirdparty/phantom-build -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/extern/phantom
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/phantom")
        execute_process(COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}/thirdparty/phantom-build --target install --parallel
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/phantom")
        find_package(Phantom REQUIRED PATHS "${PROJECT_BINARY_DIR}/extern/phantom")
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()
            
# link_directories( ${OpenFHE_LIBDIR} )
# link_directories( ${OPENMP_LIBRARIES} )
# set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
# link_libraries( ${OpenFHE_SHARED_LIBRARIES} )

if(NOT USE_HE_GPU)
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
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
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
        CUDA::cudart # <cuda_runtime_api.h>
        ${CMAKE_DL_LIBS} # fix: undefined reference to symbol 'dlclose@@GLIBC_2.2.5'
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
    target_include_directories(SCI-LinearHE PRIVATE
        "${PROJECT_BINARY_DIR}/extern/phantom/include/phantom")
    set_target_properties(SCI-LinearHE PROPERTIES CUDA_ARCHITECTURES "70;80;90")
endif()
