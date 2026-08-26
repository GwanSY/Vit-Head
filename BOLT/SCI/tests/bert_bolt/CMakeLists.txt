add_library(LIB_BERT 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
    )

target_link_libraries(LIB_BERT SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

# add_executable(BERT_NONLINEAR "test_nonlinear.cpp")
# target_link_libraries(BERT_NONLINEAR NONLINEAR)

add_executable(BOLT_BERT "bolt_bert.cpp")
target_link_libraries(BOLT_BERT LIB_BERT)

if(BOLT_USE_HE_GPU)
    add_library(LIB_BERT_GPU
        he.cpp
        nonlinear.cpp
        linear.cpp
        bert_utils.cpp
        bert.cpp)
    target_compile_definitions(LIB_BERT_GPU PRIVATE BOLT_BERT_GPU_ENTRY=1)
    target_link_libraries(LIB_BERT_GPU SCI-BOLTTransformer SCI-FloatingPoint SCI-HE)

    add_executable(BOLT_BERT_GPU "bolt_bert.cpp")
    target_compile_definitions(BOLT_BERT_GPU PRIVATE BOLT_BERT_GPU_ENTRY=1)
    target_link_libraries(BOLT_BERT_GPU LIB_BERT_GPU)
endif()

add_executable(BERT_SHARED_BACKEND_REGRESSION
    bert_shared_backend_regression.cpp)
target_link_libraries(BERT_SHARED_BACKEND_REGRESSION LIB_BERT)


# add_executable(BOLT_WE_BERT "bolt_bert_word_elimination.cpp")
# target_link_libraries(BOLT_WE_BERT LIB_BERT)
