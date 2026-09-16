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

# ViT-B/16 reuses BLB in a separate library so the reproduced BERT CPU/GPU
# sources and executable targets remain unchanged.
set(BLB_VIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vit_b16")
set(BLB_VIT_RUNTIME "${BLB_VIT_ROOT}/runtime")
add_library(LIB_VIT_BLB_CKKS
    "${BLB_VIT_RUNTIME}/he.cpp"
    "${BLB_VIT_RUNTIME}/nonlinear.cpp"
    "${BLB_VIT_RUNTIME}/linear_ckks.cpp"
    "${BLB_VIT_RUNTIME}/ckks_bert.cpp"
    "${BLB_VIT_RUNTIME}/ckks_bert_basic.cpp"
    "${BLB_VIT_RUNTIME}/bert_utils.cpp"
)
target_include_directories(LIB_VIT_BLB_CKKS PUBLIC
    "${BLB_VIT_RUNTIME}"
    "${BLB_VIT_ROOT}"
)
if(USE_HE_GPU)
    target_compile_definitions(LIB_VIT_BLB_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()
target_link_libraries(LIB_VIT_BLB_CKKS
    SCI-LinearHE
    SCI-FloatingPoint
    SCI-Math
)

add_executable(ckks_vit_b16_main
    "${BLB_VIT_RUNTIME}/ckks_vit_b16_main.cpp"
    "${BLB_VIT_ROOT}/vit_model.cpp"
)
target_link_libraries(ckks_vit_b16_main LIB_VIT_BLB_CKKS)

# OURS is a clean fork of the audited ViT-B/16 BLB runtime.  Keep it in a
# separate library/executable so A0 remains buildable from the byte-identical
# donor sources while A1-A4 evolve independently on the same SCI/Phantom
# backend.
set(OURS_VIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../ours_vit_b16")
set(OURS_VIT_RUNTIME "${OURS_VIT_ROOT}/runtime")
add_library(LIB_VIT_OURS_CKKS
    "${OURS_VIT_RUNTIME}/hetero_plan.cpp"
    "${OURS_VIT_RUNTIME}/he.cpp"
    "${OURS_VIT_RUNTIME}/nonlinear.cpp"
    "${OURS_VIT_RUNTIME}/linear_ckks.cpp"
    "${OURS_VIT_RUNTIME}/ckks_bert.cpp"
    "${OURS_VIT_RUNTIME}/ckks_bert_basic.cpp"
    "${OURS_VIT_RUNTIME}/bert_utils.cpp"
)
target_include_directories(LIB_VIT_OURS_CKKS PUBLIC
    "${OURS_VIT_RUNTIME}"
    "${OURS_VIT_ROOT}"
)
if(USE_HE_GPU)
    target_compile_definitions(LIB_VIT_OURS_CKKS PUBLIC BLB_HAS_HE_GPU=1)
endif()
target_link_libraries(LIB_VIT_OURS_CKKS
    SCI-LinearHE
    SCI-FloatingPoint
    SCI-Math
)

add_executable(ours_vit_b16_main
    "${OURS_VIT_RUNTIME}/ckks_vit_b16_main.cpp"
    "${OURS_VIT_ROOT}/vit_model.cpp"
)
target_link_libraries(ours_vit_b16_main LIB_VIT_OURS_CKKS)

add_executable(ours_vit_hetero_plan_test
    "${OURS_VIT_RUNTIME}/hetero_plan_test.cpp"
)
target_link_libraries(ours_vit_hetero_plan_test LIB_VIT_OURS_CKKS)

add_executable(ours_vit_edge_cloud_test
    "${OURS_VIT_RUNTIME}/edge_cloud_runtime_test.cpp"
)
target_link_libraries(ours_vit_edge_cloud_test LIB_VIT_OURS_CKKS)

add_executable(ours_vit_a0_conversion_selftest
    "${OURS_VIT_RUNTIME}/a0_conversion_selftest.cpp"
)
target_link_libraries(ours_vit_a0_conversion_selftest LIB_VIT_OURS_CKKS)

add_executable(ours_vit_conversion_test
    "${OURS_VIT_RUNTIME}/conversion_test.cpp"
)
target_link_libraries(ours_vit_conversion_test LIB_VIT_OURS_CKKS)

add_executable(ours_vit_row_max_test
    "${OURS_VIT_RUNTIME}/secure_row_max_test.cpp"
)
target_link_libraries(ours_vit_row_max_test LIB_VIT_OURS_CKKS)

add_test_BOLT(bolt_bert_large)
# add_executable(BOLT_WE_BERT "bolt_bert_word_elimination.cpp")
# target_link_libraries(BOLT_WE_BERT LIB_BERT)
