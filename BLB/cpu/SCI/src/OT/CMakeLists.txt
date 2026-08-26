add_library(SCI-OTPrimitive INTERFACE)
target_link_libraries(SCI-OTPrimitive
    INTERFACE SCI-utils
)


# include_directories("/ezpc_dir/EzPC/SCI/src")
# find_package(emp-tool REQUIRED)
# find_package(emp-ot REQUIRED)


# add_library(SCI-OTPrimitive emp-ot.h)
# target_link_libraries(SCI-OTPrimitive emp-tool emp-ot)
# target_link_libraries(SCI-OTPrimitive
#  SCI-utils
# )