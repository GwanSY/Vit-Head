package(default_visibility = ["//visibility:public"])

cc_import(
    name = "phantom_binary",
    shared_library = "lib/libPhantom.so",
)

cc_library(
    name = "phantom_shared",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [":phantom_binary"],
)
