include_directories("${CMAKE_SOURCE_DIR}/src")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
add_library(LIB_BERT_BOLT 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
)

add_library(LIB_BERT_CKKS
    he.cpp
    nonlinear.cpp
    linear_ckks.cpp
    ckks_bert.cpp
    ckks_bert_basic.cpp
    bert_utils.cpp
)
if(USE_HE_GPU)
    target_sources(LIB_BERT_CKKS PRIVATE ckks_plain_cache.cpp)
    target_compile_definitions(LIB_BERT_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()

# End-to-end seven-block MPCViT-CIFAR10 target.  It shares the audited
# 65/128 Phantom geometry with the single-block heterogeneous acceptance
# binary, while loading every trained block and the real 512-wide FFN.
add_library(LIB_VIT65_FULL_CKKS
    he.cpp
    nonlinear.cpp
    linear_ckks.cpp
    ckks_bert.cpp
    ckks_bert_basic.cpp
    bert_utils.cpp
)
target_compile_definitions(LIB_VIT65_FULL_CKKS PUBLIC
    PACKING_NUM=4
    INPUT_DIM=128
    COMMON_DIM=256
    OUTPUT_DIM=64
    INTER_DIM=512
    ATTENTION_LAYERS=7
    NUM_CLASS=10
    BLB_QKV_WEIGHT_FRACTIONAL_BITS=9
    BLB_QKV_BIAS_FRACTIONAL_BITS=11
    BLB_VIT65_A0=1
    BLB_VIT65_HETERO=1
    BLB_VIT65_FULL=1
)
if(USE_HE_GPU)
    target_sources(LIB_VIT65_FULL_CKKS PRIVATE ckks_plain_cache.cpp)
    target_compile_definitions(LIB_VIT65_FULL_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()

# A0 is a separate build of the original BLB CKKS attention donor.  Keeping it
# in its own target prevents the 65-logical-token ViT dimensions from leaking
# into the original 128-token BERT executable.  The physical sequence remains
# 128 because that is the donor's full-slot Phantom layout; the public mask
# marks tokens [65, 128) as padding at runtime.
add_library(LIB_VIT65_BLB_CKKS
    he.cpp
    nonlinear.cpp
    linear_ckks.cpp
    ckks_bert.cpp
    ckks_bert_basic.cpp
    bert_utils.cpp
)
target_compile_definitions(LIB_VIT65_BLB_CKKS PUBLIC
    PACKING_NUM=4
    INPUT_DIM=128
    COMMON_DIM=256
    OUTPUT_DIM=64
    INTER_DIM=1024
    ATTENTION_LAYERS=1
    BLB_QKV_WEIGHT_FRACTIONAL_BITS=9
    BLB_QKV_BIAS_FRACTIONAL_BITS=11
    BLB_VIT65_A0=1
)
if(USE_HE_GPU)
    target_sources(LIB_VIT65_BLB_CKKS PRIVATE ckks_plain_cache.cpp)
    target_compile_definitions(LIB_VIT65_BLB_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()

# Same physical Phantom layout as A0, with a separately compiled fixed public
# heterogeneous assignment.  Keeping a distinct library prevents selective
# boundary/depth code from changing the A0 binary used for comparison.
add_library(LIB_VIT65_HETERO_CKKS
    he.cpp
    nonlinear.cpp
    linear_ckks.cpp
    ckks_bert.cpp
    ckks_bert_basic.cpp
    bert_utils.cpp
)
target_compile_definitions(LIB_VIT65_HETERO_CKKS PUBLIC
    PACKING_NUM=4
    INPUT_DIM=128
    COMMON_DIM=256
    OUTPUT_DIM=64
    INTER_DIM=1024
    ATTENTION_LAYERS=1
    BLB_QKV_WEIGHT_FRACTIONAL_BITS=9
    BLB_QKV_BIAS_FRACTIONAL_BITS=11
    BLB_VIT65_A0=1
    BLB_VIT65_HETERO=1
)
if(USE_HE_GPU)
    target_sources(LIB_VIT65_HETERO_CKKS PRIVATE ckks_plain_cache.cpp)
    target_compile_definitions(LIB_VIT65_HETERO_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()
# target_compile_definitions(LIB_BERT_BOLT PUBLIC SCI_HE=1 SCI-OT=0 USE_CHEETAH=0)
# target_compile_definitions(LIB_BERT_CKKS PUBLIC SCI_HE=1 SCI-OT=0 USE_CHEETAH=0)
target_link_libraries(LIB_BERT_BOLT SCI-LinearHE)
target_link_libraries(LIB_BERT_BOLT SCI-FloatingPoint)
# target_link_libraries(LIB_BERT_BOLT SCI-HE)
target_link_libraries(LIB_BERT_BOLT SCI-Math)


target_link_libraries(LIB_BERT_CKKS SCI-LinearHE)
target_link_libraries(LIB_BERT_CKKS SCI-FloatingPoint)
# target_link_libraries(LIB_BERT_CKKS SCI-HE)
target_link_libraries(LIB_BERT_CKKS SCI-Math)

target_link_libraries(LIB_VIT65_BLB_CKKS SCI-LinearHE)
target_link_libraries(LIB_VIT65_BLB_CKKS SCI-FloatingPoint)
target_link_libraries(LIB_VIT65_BLB_CKKS SCI-Math)

target_link_libraries(LIB_VIT65_HETERO_CKKS SCI-LinearHE)
target_link_libraries(LIB_VIT65_HETERO_CKKS SCI-FloatingPoint)
target_link_libraries(LIB_VIT65_HETERO_CKKS SCI-Math)
target_link_libraries(LIB_VIT65_FULL_CKKS SCI-LinearHE)
target_link_libraries(LIB_VIT65_FULL_CKKS SCI-FloatingPoint)
target_link_libraries(LIB_VIT65_FULL_CKKS SCI-Math)

# add_executable(BERT_NONLINEAR "test_nonlinear.cpp")
# target_link_libraries(BERT_NONLINEAR NONLINEAR)

# add_executable(BOLT_BERT "bolt_bert.cpp")
# target_link_libraries(BOLT_BERT LIB_BERT)

macro (add_test_BOLT _name)
	add_executable(${_name} "${_name}.cpp")
    # target_link_libraries(${_name} SCI-FloatingPoint)
    # target_link_libraries(${_name} SCI-Math) 
    # target_link_libraries(${_name} SCI-HE) 
    target_link_libraries(${_name} LIB_BERT_BOLT) 
    # target_compile_definitions(${_name} SCI_HE=1)
endmacro()

macro (add_test_CKKS _name)
	add_executable(${_name} "${_name}.cpp")
    # target_link_libraries(${_name} SCI-FloatingPoint)
    # target_link_libraries(${_name} SCI-Math) 
    # target_link_libraries(${_name} SCI-HE) 
    target_link_libraries(${_name} LIB_BERT_CKKS) 
    # target_compile_definitions(${_name} SCI_HE=1)
endmacro()

add_test_CKKS(ckks_bert_large_main)
add_test_CKKS(ckks_conversion_regression)
add_test_CKKS(ckks_conversion_e2e)
add_test_CKKS(ckks_secure_fft_regression)
add_test_CKKS(ckks_table6_he_only)
if(USE_HE_GPU)
    add_test_CKKS(ckks_plain_cache_test)
endif()
add_executable(ckks_vit65_a0 ckks_vit65_a0.cpp)
target_link_libraries(ckks_vit65_a0 LIB_VIT65_BLB_CKKS)
add_executable(ckks_vit65_hetero ckks_vit65_a0.cpp)
target_link_libraries(ckks_vit65_hetero LIB_VIT65_HETERO_CKKS)
add_executable(ckks_vit65_full ckks_vit65_full.cpp)
target_link_libraries(ckks_vit65_full LIB_VIT65_FULL_CKKS)
add_test_BOLT(bolt_bert_large)
# add_executable(BOLT_WE_BERT "bolt_bert_word_elimination.cpp")
# target_link_libraries(BOLT_WE_BERT LIB_BERT)
