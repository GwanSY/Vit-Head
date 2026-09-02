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

target_link_libraries(LIB_BERT_BOLT SCI-FloatingPoint)
target_link_libraries(LIB_BERT_BOLT SCI-HE)
target_link_libraries(LIB_BERT_BOLT SCI-Math)

target_link_libraries(LIB_BERT_CKKS SCI-FloatingPoint)
target_link_libraries(LIB_BERT_CKKS SCI-HE)
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

add_test_CKKS(ckks_bert_main)
add_test_BOLT(bolt_bert)
# add_executable(BOLT_WE_BERT "bolt_bert_word_elimination.cpp")
# target_link_libraries(BOLT_WE_BERT LIB_BERT)
