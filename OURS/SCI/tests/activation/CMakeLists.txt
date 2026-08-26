# cmake_minimum_required (VERSION 3.3.2)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
macro (add_test_IRON _name)
	add_executable(IRON-${_name} "iron_${_name}.cpp")
    target_link_libraries(IRON-${_name} SCI-Math) 
    target_link_libraries(IRON-${_name} SCI-FloatingPoint) 
endmacro()

macro (add_test_BOLT _name)
	add_executable(BOLT-${_name} "bolt_${_name}.cpp")
    target_link_libraries(BOLT-${_name} SCI-FloatingPoint)
    target_link_libraries(BOLT-${_name} SCI-Math) 
    include_directories(../../src)
    target_link_libraries(BOLT-${_name} SCI-Millionaire Threads::Threads OpenMP::OpenMP_CXX) 
	# target_compile_definitions(BOLT-${_name} PUBLIC USE_VOLE=1)
endmacro()

# add_test_IRON(layer_norm)
# add_test_IRON(softmax)
# add_test_IRON(gelu)
# add_test_IRON(tanh)

# add_test_BOLT(softmax)
# add_test_BOLT(gelu)
# add_test_BOLT(tanh)
add_test_BOLT(layer_norm)

