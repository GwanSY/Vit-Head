add_executable(BOLT_VIT bolt_vit.cpp bolt_vit_protocol.cpp bolt_vit_mpc.cpp vit_model.cpp)
execute_process(
  COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse HEAD
  OUTPUT_VARIABLE BOLT_PARENT_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT BOLT_PARENT_GIT_COMMIT)
  set(BOLT_PARENT_GIT_COMMIT "unavailable")
endif()
target_compile_definitions(BOLT_VIT PRIVATE BOLT_VIT_PROTOCOL_ENFORCEMENT=1)
target_compile_definitions(BOLT_VIT PRIVATE
  BOLT_BUILD_GIT_COMMIT="${BOLT_PARENT_GIT_COMMIT}")
target_link_libraries(BOLT_VIT SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

if(BOLT_USE_HE_GPU)
  add_executable(BOLT_VIT_GPU bolt_vit.cpp bolt_vit_protocol.cpp bolt_vit_mpc.cpp vit_model.cpp)
  target_compile_definitions(BOLT_VIT_GPU PRIVATE
    BOLT_VIT_PROTOCOL_ENFORCEMENT=1
    BOLT_VIT_GPU_ENTRY=1)
  target_compile_definitions(BOLT_VIT_GPU PRIVATE
    BOLT_BUILD_GIT_COMMIT="${BOLT_PARENT_GIT_COMMIT}")
  target_link_libraries(BOLT_VIT_GPU SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)
endif()

# IRON's public build entry point uses the same protocol-preserving ViT graph.
# This is deliberately a second executable target, rather than a fallback
# implementation: it shares the BOLT packed BFV/MPC adapter and only selects
# the IRON-compatible model/input magic in the loader.  Keeping the target
# here makes `IRON/scripts/build_secure_vit.sh` build the source currently in
# this tree instead of relying on a stale binary from an older checkout.
add_executable(IRON_VIT bolt_vit.cpp bolt_vit_protocol.cpp bolt_vit_mpc.cpp vit_model.cpp)
target_compile_definitions(IRON_VIT PRIVATE
  BOLT_VIT_PROTOCOL_ENFORCEMENT=1
  IRON_VIT_BASELINE=1)
target_compile_definitions(IRON_VIT PRIVATE
  BOLT_BUILD_GIT_COMMIT="${BOLT_PARENT_GIT_COMMIT}")
target_link_libraries(IRON_VIT SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_MODEL_INSPECT vit_model_inspect.cpp vit_model.cpp)

add_executable(BOLT_VIT_KERNEL_TEST bolt_vit_kernel_test.cpp)
target_link_libraries(BOLT_VIT_KERNEL_TEST SCI-BOLTTransformer SCI-HE)

add_executable(BOLT_VIT_FULL_QK_TEST bolt_vit_full_qk_test.cpp)
target_link_libraries(BOLT_VIT_FULL_QK_TEST SCI-BOLTTransformer SCI-HE)

add_executable(BOLT_VIT_LINEAR_TEST bolt_vit_linear_test.cpp)
target_link_libraries(BOLT_VIT_LINEAR_TEST SCI-BOLTTransformer SCI-HE)

add_executable(BOLT_VIT_SOFTMAX_V_TEST bolt_vit_softmax_v_test.cpp)
target_link_libraries(BOLT_VIT_SOFTMAX_V_TEST SCI-BOLTTransformer SCI-HE)

add_executable(BOLT_VIT_SIGNED_CONVERSION_TEST
    bolt_vit_signed_conversion_test.cpp bolt_vit_mpc.cpp)
target_link_libraries(BOLT_VIT_SIGNED_CONVERSION_TEST
    SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_MPC_REFERENCE_TEST bolt_vit_mpc_reference_test.cpp)
target_link_libraries(BOLT_VIT_MPC_REFERENCE_TEST SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_MPC_THREAD_TEST
    bolt_vit_mpc_thread_test.cpp bolt_vit_mpc.cpp)
target_link_libraries(BOLT_VIT_MPC_THREAD_TEST SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_MPC_SHAPE_TEST
    bolt_vit_mpc_shape_test.cpp bolt_vit_mpc.cpp)
target_link_libraries(BOLT_VIT_MPC_SHAPE_TEST SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_ATTENTION_PROTOCOL_TEST
    bolt_vit_attention_protocol_test.cpp)
target_link_libraries(BOLT_VIT_ATTENTION_PROTOCOL_TEST
    SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_STEM_PROTOCOL_TEST
    bolt_vit_stem_protocol_test.cpp vit_model.cpp)
target_link_libraries(BOLT_VIT_STEM_PROTOCOL_TEST
    SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

add_executable(BOLT_VIT_STEM_PROTOCOL_VALIDATION
    bolt_vit_stem_protocol_test.cpp vit_model.cpp)
target_compile_definitions(BOLT_VIT_STEM_PROTOCOL_VALIDATION PRIVATE
    BOLT_STEM_VALIDATION_CHANNEL=1)
target_link_libraries(BOLT_VIT_STEM_PROTOCOL_VALIDATION
    SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)
